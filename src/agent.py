"""
Claude Agent with tool-use for RAG retrieval over Milvus.
Two tools: search (vector search) and get_image (screenshot retrieval).
"""
import json
import asyncio
from typing import AsyncGenerator

import anthropic

from .search import search, get_image_base64

SYSTEM_PROMPT = """You are a medical factsheet knowledge retrieval assistant with access to a vector database via tools.

## Retrieval Process
Follow this two-step process to answer queries:

1. **Text search first** — Use the 'search' tool to perform an initial text-based vector search. Results will include extracted text and a path to the full-page screenshot it came from.

2. **Image lookup when needed** — If the text results are insufficient, ambiguous, or lack context, use the 'get_image' tool to retrieve the full-page screenshot for a richer visual understanding of the document layout.

## Guidelines
- Always start with 'search' before falling back to 'get_image'.
- Use 'get_image' proactively when the query is visual in nature (e.g., charts, diagrams, tables) or when text snippets alone are inconclusive.
- Base your answers strictly on retrieved content — do not rely on prior medical knowledge.
- When counting or aggregating across the entire document, retrieve all pages to ensure completeness.

## Citations (IMPORTANT)
You MUST cite your sources inline using this exact format: [Source: filename | Page N | screenshot_path]

For example: According to the factsheet [Source: 01_NNMC_Medication_Side_Effects.pdf | Page 1 | screenshots/01_NNMC_Medication_Side_Effects/page_1.png], Warfarin is a blood thinner.

Rules for citations:
- Use the SOURCE, PAGE, and PAGE SCREENSHOT PATH from each search result
- Place the citation immediately after the claim it supports
- Every factual claim must have a citation
- If multiple sources support a claim, include all citations"""

TOOLS = [
    {
        "name": "search",
        "description": "Search the knowledge base using vector similarity to find relevant text chunks from the document. Returns text content and the path to the page screenshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document chunks.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of chunks to return (default: 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_image",
        "description": "Retrieve the full-page screenshot for visual context. Use this when text search results are insufficient or when you need to see the document layout (tables, columns, headers).",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the screenshot file (returned by the search tool).",
                },
            },
            "required": ["image_path"],
        },
    },
]


def handle_tool_call(tool_name: str, tool_input: dict) -> list[dict]:
    """Execute a tool call and return the result content blocks."""
    if tool_name == "search":
        results = search(tool_input["query"], tool_input.get("limit", 3))
        content_parts = []
        for r in results:
            text = (
                f"SOURCE: {r.get('source_file', 'unknown')}\n"
                f"PAGE SCREENSHOT PATH: {r['image_path']}\n"
                f"PAGE: {r['page_num']}\n"
                f"SIMILARITY: {r['distance']:.4f}\n\n"
                f"CONTENT:\n{r['text']}"
            )
            content_parts.append({"type": "text", "text": text})
        if not content_parts:
            content_parts.append({"type": "text", "text": "No results found."})
        return content_parts

    elif tool_name == "get_image":
        try:
            img_b64 = get_image_base64(tool_input["image_path"])
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                }
            ]
        except FileNotFoundError as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    return [{"type": "text", "text": f"Unknown tool: {tool_name}"}]


def run_agent(query: str, verbose: bool = True) -> str:
    """
    Run the Claude agent loop with tool use.
    Returns the final text response.
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": query}]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

    tool_call_count = 0

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect all text and tool_use blocks from the response
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Check if we're done (no more tool calls)
        if response.stop_reason == "end_turn":
            # Extract final text
            final_text = ""
            for block in assistant_content:
                if hasattr(block, "text"):
                    final_text += block.text
            if verbose:
                print(f"\nFinal Answer:\n{final_text}")
                print(f"\nTotal tool calls: {tool_call_count}")
            return final_text

        # Process tool calls
        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                tool_call_count += 1
                if verbose:
                    print(f"\n  Tool call #{tool_call_count}: {block.name}")
                    print(f"  Input: {json.dumps(block.input, indent=2)[:200]}")

                result_content = handle_tool_call(block.name, block.input)

                if verbose:
                    for rc in result_content:
                        if rc.get("type") == "text":
                            preview = rc["text"][:150] + "..." if len(rc["text"]) > 150 else rc["text"]
                            print(f"  Result: {preview}")
                        elif rc.get("type") == "image":
                            print(f"  Result: [image returned]")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

        messages.append({"role": "user", "content": tool_results})


async def run_agent_stream(query: str) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields structured step events as the agent executes.
    Used by the web UI to stream execution progress via SSE.
    """
    client = anthropic.AsyncAnthropic()
    messages = [{"role": "user", "content": query}]

    step = 0
    search_calls = 0
    image_calls = 0
    tool_call_count = 0

    yield {"type": "start", "query": query}

    try:
        while True:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Emit any text blocks as thinking steps
            for block in assistant_content:
                if hasattr(block, "text") and block.type == "text" and response.stop_reason != "end_turn":
                    step += 1
                    yield {
                        "type": "thinking",
                        "step": step,
                        "content": block.text[:500],
                    }

            # Final answer
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in assistant_content:
                    if hasattr(block, "text"):
                        final_text += block.text
                yield {
                    "type": "answer",
                    "content": final_text,
                    "stats": {
                        "steps": step,
                        "tool_calls": tool_call_count,
                        "search_calls": search_calls,
                        "image_calls": image_calls,
                    },
                }
                return

            # Process tool calls
            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    step += 1
                    tool_call_count += 1

                    if block.name == "search":
                        search_calls += 1
                    elif block.name == "get_image":
                        image_calls += 1

                    # Execute tool in thread (Milvus is sync)
                    result_content = await asyncio.to_thread(
                        handle_tool_call, block.name, block.input
                    )

                    # Build preview for the UI
                    preview = ""
                    for rc in result_content:
                        if rc.get("type") == "text":
                            preview = rc["text"][:300]
                        elif rc.get("type") == "image":
                            preview = f"[Image retrieved: {block.input.get('image_path', '')}]"

                    yield {
                        "type": "tool_call",
                        "step": step,
                        "tool": block.name,
                        "input": block.input,
                        "preview": preview,
                    }

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_content,
                    })

            messages.append({"role": "user", "content": tool_results})

    except Exception as e:
        yield {"type": "error", "message": str(e)}
