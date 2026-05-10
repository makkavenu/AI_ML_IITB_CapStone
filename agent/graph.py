"""LangGraph StateGraph for the multi-modal AI agent.

Graph topology
--------------
  orchestrator ──(tool_call?)──► tool_executor ──► synthesizer ──► END
       │
       └──(no tool)──► END

Nodes
-----
- orchestrator : GPT-4o with bound tools decides which tool (if any) to call.
- tool_executor: Runs the chosen tool and appends a ToolMessage to state.
- synthesizer  : GPT-4o reads the full conversation (including tool output)
                 and produces the final user-facing response.
"""

import logging
from typing import Annotated, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.tools.tool_definitions import TOOL_MAP, TOOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared state threaded through every graph node.

    Attributes:
        messages: Full conversation history (managed by LangGraph add_messages).
        tool_used: Name of the last tool invoked (empty string if none).
        image_base64: Base64-encoded image supplied by the user (may be empty).
        tool_output: Raw string output from the last tool call.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    tool_used: str
    image_base64: Optional[str]
    tool_output: str


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _make_llm(*, bind_tools: bool = False) -> ChatOpenAI:
    """Instantiate a GPT-4o ChatOpenAI client.

    Args:
        bind_tools: When True, attaches the full TOOLS list to the model so it
            can emit structured tool-call requests.

    Returns:
        A (possibly tool-bound) ChatOpenAI instance.
    """
    llm: ChatOpenAI = ChatOpenAI(model="gpt-4o", temperature=0)
    if bind_tools:
        return llm.bind_tools(TOOLS)
    return llm


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def orchestrator_node(state: AgentState) -> dict:
    """GPT-4o orchestrator: inspect the conversation and decide which tool to use.

    Args:
        state: Current agent state containing the conversation history.

    Returns:
        Partial state update with the orchestrator's AIMessage appended.
    """
    logger.info("orchestrator_node | message_count=%d", len(state["messages"]))
    try:
        llm = _make_llm(bind_tools=True)
        response: AIMessage = await llm.ainvoke(state["messages"])
        logger.info(
            "orchestrator_node | tool_calls=%s",
            [tc["name"] for tc in (response.tool_calls or [])],
        )
        return {"messages": [response], "tool_used": "", "tool_output": ""}
    except Exception:
        logger.exception("orchestrator_node raised an exception")
        raise


async def tool_executor_node(state: AgentState) -> dict:
    """Run every tool call emitted by the orchestrator.

    Injects ``image_base64`` from state into any vision / detection tool that
    did not already receive it from the LLM.

    Args:
        state: Current agent state; the last message must be an AIMessage with
            at least one tool_call.

    Returns:
        Partial state update with ToolMessages appended and tool metadata set.
    """
    logger.info("tool_executor_node | executing tool(s)")
    last_message: AIMessage = state["messages"][-1]

    tool_messages: list[ToolMessage] = []
    tool_used: str = ""
    tool_output: str = ""

    for tool_call in last_message.tool_calls:
        tool_name: str = tool_call["name"]
        tool_args: dict = dict(tool_call["args"])
        tool_used = tool_name

        # Inject image from state for vision / detection tools when the LLM
        # did not forward it (the image is in the message content, not in args).
        vision_tools = {"vision_llm", "object_detection"}
        if tool_name in vision_tools and not tool_args.get("image_base64"):
            tool_args["image_base64"] = state.get("image_base64") or ""

        logger.info("tool_executor_node | tool=%s args_keys=%s", tool_name, list(tool_args.keys()))

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
        tool_messages.append(
            ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
        )

    return {
        "messages": tool_messages,
        "tool_used": tool_used,
        "tool_output": tool_output,
    }


async def synthesizer_node(state: AgentState) -> dict:
    """GPT-4o synthesizer: craft the final user-facing response from tool output.

    Args:
        state: Current agent state with the full conversation including tool output.

    Returns:
        Partial state update with the synthesizer's AIMessage appended.
    """
    logger.info("synthesizer_node | composing final response")
    try:
        llm = _make_llm(bind_tools=False)
        response: AIMessage = await llm.ainvoke(state["messages"])
        return {"messages": [response]}
    except Exception:
        logger.exception("synthesizer_node raised an exception")
        raise


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def _route_after_orchestrator(
    state: AgentState,
) -> Literal["tool_executor", "__end__"]:
    """Conditional edge: branch to tool_executor when the orchestrator emitted
    tool calls, otherwise end the graph.

    Args:
        state: Current agent state.

    Returns:
        ``'tool_executor'`` or ``END`` (``'__end__'``).
    """
    last_message: AIMessage = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        logger.info(
            "_route_after_orchestrator | routing to tool_executor (%d call(s))",
            len(last_message.tool_calls),
        )
        return "tool_executor"
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
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"tool_executor": "tool_executor", END: END},
    )
    graph.add_edge("tool_executor", "synthesizer")
    graph.add_edge("synthesizer", END)

    logger.info("build_graph | graph compiled successfully")
    return graph.compile()


# Module-level compiled graph — imported by the FastAPI router.
agent_graph = build_graph()
