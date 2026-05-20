"""
Query the Indian Legal RAG system.

Pipeline:
  1. Embed the user question with OpenAI text-embedding-3-large
  2. Retrieve top-K matching Q&A chunks from Pinecone (legal-rag-iitb-project)
  3. Build a context string from the retrieved chunks
  4. Send context + question to AWS Bedrock Qwen3-32B for a final answer

Usage (CLI):
  python -m agent.tools.query_legal_rag
  python -m agent.tools.query_legal_rag --question "What is Section 302 of IPC?"
  python -m agent.tools.query_legal_rag --question "..." --top-k 5

Programmatic:
  from agent.tools.query_legal_rag import run_legal_rag
  answer = run_legal_rag("What is Section 302 of IPC?", top_k=5)
"""

import argparse
import json
import logging
import os
import threading
from typing import Optional

import boto3
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_NAME = "legal-rag-iitb-project"
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 5

# Verify this model ID in your AWS Bedrock console.
BEDROCK_MODEL_ID = "qwen.qwen3-32b-v1:0"

# ---------------------------------------------------------------------------
# Lazy client cache — build once, reuse for every request
# ---------------------------------------------------------------------------
_CLIENTS: dict = {}
_CLIENTS_LOCK = threading.Lock()


def _get_clients() -> dict:
    """Build (once) and return the OpenAI / Pinecone / Bedrock clients.

    Returns:
        Dict with keys ``openai``, ``index``, ``bedrock``.

    Raises:
        ValueError: When a required environment variable is missing.
    """
    if _CLIENTS:
        return _CLIENTS

    with _CLIENTS_LOCK:
        if _CLIENTS:  # double-checked locking
            return _CLIENTS

        load_dotenv()

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        index_name = os.getenv("PINECONE_INDEX", INDEX_NAME)

        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY missing from environment")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY missing from environment")
        if not aws_access_key_id or not aws_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY missing")

        openai_client = OpenAI(api_key=openai_api_key)
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_key,
        )

        _CLIENTS.update({"openai": openai_client, "index": index, "bedrock": bedrock})
        logger.info("_get_clients | legal RAG clients initialised")
        return _CLIENTS


# ---------------------------------------------------------------------------
# Embed query
# ---------------------------------------------------------------------------
def embed_query(client: OpenAI, question: str) -> list:
    """Embed a question using OpenAI ``text-embedding-3-large``."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# Retrieve relevant Q&A chunks from Pinecone
# ---------------------------------------------------------------------------
def retrieve_context(index, query_vector: list, top_k: int) -> str:
    """Query Pinecone and return a formatted context string."""
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    if not result.matches:
        return "No relevant context found."

    parts = []
    for i, match in enumerate(result.matches, start=1):
        meta = match.metadata or {}
        source = meta.get("source", "Unknown")
        text = meta.get("text", "")
        score = round(match.score, 4)
        parts.append(f"[{i}] Source: {source}  (similarity: {score})\n{text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Call AWS Bedrock Qwen3-32B
# ---------------------------------------------------------------------------
def ask_bedrock_qwen(
    bedrock_client,
    context: str,
    user_question: str,
    model_id: str = BEDROCK_MODEL_ID,
) -> str:
    """Send context + question to AWS Bedrock Qwen3-32B and return the answer."""
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a legal assistant specialising in Indian law. "
                        "Answer ONLY based on the provided legal Q&A context. "
                        "Cite legal provisions (of (IPC / CrPC / Constitution) by their actual names, such as Section 302 IPC, "
                        "Section 420 IPC, Article 21 of the Constitution, etc. "
                        "Do not output bracketed retrieval labels like [Source 1], [Source 2], or [Source 4]."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {user_question}"
                    ),
                },
            ],
            "max_tokens": 1024,
        }
    )

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response["body"].read())

    # Try common response shapes (OpenAI-compatible, then Bedrock-native)
    try:
        return response_body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass
    try:
        return response_body["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        pass
    return json.dumps(response_body, indent=2)


# ---------------------------------------------------------------------------
# Public entry point — used by the legal_qa tool
# ---------------------------------------------------------------------------
def run_legal_rag(question: str, top_k: int = TOP_K) -> str:
    """End-to-end RAG pipeline: embed → retrieve → generate.

    Args:
        question: The user's legal question.
        top_k: Number of Pinecone chunks to retrieve.

    Returns:
        The model's final answer string.

    Raises:
        ValueError: When required environment variables are missing.
        Exception: Any error from OpenAI / Pinecone / Bedrock is propagated.
    """
    clients = _get_clients()
    logger.info("run_legal_rag | top_k=%d question[:120]=%r", top_k, question[:120])

    query_vector = embed_query(clients["openai"], question)
    context = retrieve_context(clients["index"], query_vector, top_k)
    answer = ask_bedrock_qwen(clients["bedrock"], context, question)
    return answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Command-line interface for ad-hoc querying."""
    parser = argparse.ArgumentParser(description="Query the Indian Legal RAG system.")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="What are the fundamental rights guaranteed by the Indian Constitution?",
        help="Legal question to ask",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Question : {args.question}")
    print("=" * 70)

    answer = run_legal_rag(args.question, top_k=args.top_k)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    main()
