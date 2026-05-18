"""LangGraph StateGraph for the multi-modal AI agent.

Graph topology (ReAct loop, supports multi-tool chaining)
---------------------------------------------------------
  orchestrator ──(tool_call?)──► tool_executor ──► orchestrator (loop)
       │
       └──(no tool / max iters)──► END

Nodes
-----
- orchestrator : GPT-4o with bound tools. Either emits tool call(s) or
                 produces the final user-facing answer.
- tool_executor: Runs the chosen tool(s) and appends ToolMessage(s) to state.
                 Loops back to the orchestrator so it can chain another tool
                 (e.g. vision_llm → legal_qa) or produce the final response.
"""

import logging
import os
from typing import Annotated, Any, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.tools.tool_definitions import TOOL_MAP, TOOLS

logger = logging.getLogger(__name__)


# Maximum number of tool-executor cycles per request. Prevents accidental
# infinite loops if the orchestrator keeps requesting tool calls.
MAX_ITERATIONS: int = 4


# ---------------------------------------------------------------------------
# Orchestrator model registry
# ---------------------------------------------------------------------------

# Logical names exposed to the UI → concrete provider + id.
# Keeping this central makes it trivial to add more options later.
ORCHESTRATOR_MODELS: dict[str, dict[str, str]] = {
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "label": "GPT-4o (OpenAI)",
    },
    "qwen3-32b": {
        "provider": "bedrock",
        "model_id": "qwen.qwen3-32b-v1:0",
        "label": "Qwen3-32B (AWS Bedrock)",
    },
}
DEFAULT_ORCHESTRATOR_MODEL: str = "qwen3-32b"


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared state threaded through every graph node.

    Attributes:
        messages: Full conversation history (managed by LangGraph add_messages).
        tool_used: Name of the LAST tool invoked (empty string if none).
        tools_chain: Ordered list of tool names invoked across the request.
        image_base64: Base64-encoded image supplied by the user (may be empty).
        tool_output: Raw string output from the last tool call.
        tool_outputs_by_name: Per-tool latest raw output (for downstream
            consumers such as bounding-box drawing).
        iterations: Number of tool-executor cycles completed so far.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    tool_used: str
    tools_chain: list[str]
    image_base64: Optional[str]
    tool_output: str
    tool_outputs_by_name: dict[str, str]
    iterations: int
    orchestrator_model: str


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _make_llm(
    *, bind_tools: bool = False, model_key: str = DEFAULT_ORCHESTRATOR_MODEL
) -> BaseChatModel:
    """Instantiate the orchestrator chat model.

    Supports two providers selected by ``model_key``:

    * ``"gpt-4o"``    — ``ChatOpenAI`` (uses OPENAI_API_KEY).
    * ``"qwen3-32b"`` — ``ChatBedrockConverse`` from ``langchain-aws`` calling
      the Bedrock Converse API (uses standard AWS creds + ``AWS_DEFAULT_REGION``).
      Qwen on Bedrock supports tool use via Converse, so ``bind_tools`` works.

    Falls back to GPT-4o if an unknown key is supplied.

    Args:
        bind_tools: When True, attaches the full TOOLS list so the model can
            emit structured tool-call requests.
        model_key: Logical model name (key in ``ORCHESTRATOR_MODELS``).

    Returns:
        A (possibly tool-bound) chat model.
    """
    cfg = ORCHESTRATOR_MODELS.get(model_key) or ORCHESTRATOR_MODELS[DEFAULT_ORCHESTRATOR_MODEL]
    provider = cfg["provider"]
    model_id = cfg["model_id"]

    llm: BaseChatModel
    if provider == "bedrock":
        # Imported lazily so the API can boot even if langchain-aws is missing
        # (the OpenAI path still works in that case).
        from langchain_aws import ChatBedrockConverse  # type: ignore

        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        llm = ChatBedrockConverse(
            model=model_id,
            temperature=0,
            region_name=region,
        )
    else:
        llm = ChatOpenAI(model=model_id, temperature=0)

    if bind_tools:
        return llm.bind_tools(TOOLS)
    return llm


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    """Collapse a LangChain message ``content`` value to a plain string.

    ``ChatBedrockConverse`` returns ``AIMessage.content`` as a list of content
    blocks (e.g. ``[{"type": "text", "text": "..."}, {"type": "tool_use", ...}]``).
    When that AIMessage is later re-sent as part of the conversation, some
    blocks can be mis-classified as image blocks, causing Qwen on Bedrock to
    reject the request. We avoid that by flattening every message's content
    to text-only before invoking a Bedrock orchestrator.

    Args:
        content: ``str`` or list of content-block dicts.

    Returns:
        Plain text representation. Non-text blocks (tool_use, image, etc.)
        are dropped — their semantics are preserved separately via
        ``AIMessage.tool_calls`` / ``ToolMessage`` records.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                # Silently drop tool_use / image / other block types.
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


def _normalise_messages_for_bedrock(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return a copy of ``messages`` with all content fields collapsed to text.

    Preserves ``tool_calls`` / ``tool_call_id`` / role so the conversation
    remains valid for Bedrock's Converse API.

    Args:
        messages: Original conversation messages.

    Returns:
        New list of messages with string content.
    """
    cleaned: list[BaseMessage] = []
    for m in messages:
        new_content = _content_to_text(m.content)
        if isinstance(m, AIMessage):
            cleaned.append(
                AIMessage(
                    content=new_content,
                    tool_calls=list(getattr(m, "tool_calls", []) or []),
                    additional_kwargs=dict(getattr(m, "additional_kwargs", {}) or {}),
                    id=getattr(m, "id", None),
                )
            )
        elif isinstance(m, ToolMessage):
            cleaned.append(
                ToolMessage(
                    content=new_content,
                    tool_call_id=m.tool_call_id,
                    name=getattr(m, "name", None),
                )
            )
        else:
            # HumanMessage / SystemMessage / etc. — clone with text content.
            clone = m.model_copy(update={"content": new_content})
            cleaned.append(clone)
    return cleaned


async def orchestrator_node(state: AgentState) -> dict:
    """Orchestrator LLM step.

    On the first call, decides which tool to invoke (if any).
    On subsequent calls (after tool results are in the conversation), it
    decides whether to call another tool or produce the final user-facing
    answer.

    Args:
        state: Current agent state containing the conversation history.

    Returns:
        Partial state update with the orchestrator's AIMessage appended.
    """
    iters = state.get("iterations", 0)
    model_key = state.get("orchestrator_model") or DEFAULT_ORCHESTRATOR_MODEL
    logger.info(
        "orchestrator_node | model=%s iterations=%d message_count=%d",
        model_key, iters, len(state["messages"]),
    )
    llm = _make_llm(bind_tools=True, model_key=model_key)

    # Bedrock's Converse API is strict about content-block types. Flatten any
    # list-of-blocks content to plain text before invocation so list-content
    # AIMessages produced by a previous Bedrock turn cannot be mis-classified
    # as image blocks on replay.
    provider = ORCHESTRATOR_MODELS.get(model_key, {}).get("provider")
    msgs = (
        _normalise_messages_for_bedrock(state["messages"])
        if provider == "bedrock"
        else state["messages"]
    )

    response: AIMessage = await llm.ainvoke(msgs)
    logger.info(
        "orchestrator_node | tool_calls=%s",
        [tc["name"] for tc in (response.tool_calls or [])],
    )
    return {"messages": [response]}


async def tool_executor_node(state: AgentState) -> dict:
    """Run every tool call emitted by the orchestrator.

    Injects ``image_base64`` from state into any vision / detection tool that
    did not already receive it from the LLM. Loops back to the orchestrator
    when finished so multi-tool chains are supported.

    Args:
        state: Current agent state; the last message must be an AIMessage with
            at least one tool_call.

    Returns:
        Partial state update with ToolMessages appended and tool metadata set.
    """
    last_message: AIMessage = state["messages"][-1]
    logger.info(
        "tool_executor_node | executing %d tool call(s)",
        len(last_message.tool_calls or []),
    )

    tool_messages: list[ToolMessage] = []
    tool_used: str = ""
    tool_output: str = ""
    chain_addition: list[str] = []
    outputs_addition: dict[str, str] = {}

    for tool_call in last_message.tool_calls:
        tool_name: str = tool_call["name"]
        tool_args: dict = dict(tool_call["args"])
        tool_used = tool_name
        chain_addition.append(tool_name)

        # GPT-4o cannot pass large binary data in tool call arguments — it either
        # omits image_base64 entirely or substitutes a short placeholder string
        # (e.g. "<base64+encoded+image>"). Always inject the real bytes from
        # state for any vision / detection tool.
        vision_tools = {"vision_llm", "object_detection"}
        if tool_name in vision_tools:
            real_image = state.get("image_base64") or ""
            if real_image:
                tool_args["image_base64"] = real_image
                logger.info(
                    "tool_executor_node | injected image_base64 from state (len=%d)",
                    len(real_image),
                )

        logger.info(
            "tool_executor_node | tool=%s args_keys=%s",
            tool_name, list(tool_args.keys()),
        )

        selected_tool = TOOL_MAP.get(tool_name)
        if selected_tool is None:
            result = f"Error: unknown tool '{tool_name}'"
            logger.error("tool_executor_node | unknown tool: %s", tool_name)
        else:
            try:
                result = await selected_tool.ainvoke(tool_args)
            except Exception:
                logger.exception("tool_executor_node | tool %s raised", tool_name)
                result = f"Tool '{tool_name}' encountered an internal error."

        tool_output = str(result)
        outputs_addition[tool_name] = tool_output
        tool_messages.append(
            ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
        )

    merged_outputs = {**state.get("tool_outputs_by_name", {}), **outputs_addition}
    new_chain = list(state.get("tools_chain", [])) + chain_addition

    return {
        "messages": tool_messages,
        "tool_used": tool_used,
        "tools_chain": new_chain,
        "tool_output": tool_output,
        "tool_outputs_by_name": merged_outputs,
        "iterations": state.get("iterations", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def _route_after_orchestrator(
    state: AgentState,
) -> Literal["tool_executor", "__end__"]:
    """Conditional edge: branch to tool_executor when the orchestrator emitted
    tool calls AND we haven't exhausted the iteration budget; otherwise end.

    Args:
        state: Current agent state.

    Returns:
        ``'tool_executor'`` or ``END`` (``'__end__'``).
    """
    last_message: AIMessage = state["messages"][-1]
    iters = state.get("iterations", 0)
    if getattr(last_message, "tool_calls", None) and iters < MAX_ITERATIONS:
        logger.info(
            "_route_after_orchestrator | routing to tool_executor "
            "(%d call(s), iter=%d/%d)",
            len(last_message.tool_calls), iters, MAX_ITERATIONS,
        )
        return "tool_executor"
    if iters >= MAX_ITERATIONS:
        logger.warning("_route_after_orchestrator | iteration cap reached — ending")
    else:
        logger.info("_route_after_orchestrator | no tool calls — ending graph")
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Build and compile the LangGraph StateGraph.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``.ainvoke()``.
    """
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("tool_executor", tool_executor_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"tool_executor": "tool_executor", END: END},
    )
    # Loop back so the orchestrator can chain another tool or synthesise the
    # final answer once it has tool results.
    graph.add_edge("tool_executor", "orchestrator")

    logger.info("build_graph | graph compiled successfully (ReAct loop)")
    return graph.compile()


# Module-level compiled graph — imported by the FastAPI router.
agent_graph = build_graph()
