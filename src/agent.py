"""
Claude Agent with tool-use for RAG retrieval over Milvus
and filesystem exploration for unindexed documents.

Combined tools:
  Indexed (Milvus): search, get_image
  Filesystem (Live): scan_folder, preview_file, parse_file, read_file, grep, glob
"""
import json
import asyncio
from typing import AsyncGenerator

import anthropic

from .search import search, get_image_base64
from .fs import (
    scan_folder,
    preview_file,
    parse_file_content,
    read_text_file,
    grep_in_file,
    glob_search,
)

SYSTEM_PROMPT = """You are an AI-powered document exploration and retrieval assistant. You have access to both a vector database (Milvus) for pre-indexed documents AND filesystem tools for live exploration of unindexed documents.

## Available Tools

### Indexed Search (Milvus Vector Database)
| Tool | Purpose |
|------|---------|
| `search` | Vector similarity search over pre-indexed document chunks. Returns text + screenshot paths. |
| `get_image` | Retrieve full-page screenshot for visual context (tables, charts, layouts). |

### Filesystem Exploration (Live)
| Tool | Purpose |
|------|---------|
| `scan_folder` | Parallel scan ALL documents in a folder — returns previews of every file. |
| `preview_file` | Quick look at a single document (~first 2 pages, 3000 chars). |
| `parse_file` | Deep read — full content of a document. |
| `read_file` | Read a plain text file (txt, md, csv, json). |
| `grep` | Search for a regex pattern within a file — returns matching lines with context. |
| `glob` | Find files matching a name pattern (e.g., '*.pdf') in a directory tree. |

## Retrieval Strategy

### When documents are indexed (search tool works):
1. **Start with `search`** for fast vector retrieval
2. **Use `get_image`** when text is ambiguous, or the query involves visual elements (tables, charts, columns)
3. **Fall back to filesystem tools** if search results are insufficient or you need broader context

### When exploring unindexed documents (or supplementing search):
Follow the **Three-Phase Exploration Strategy**:

**PHASE 1: Parallel Scan** — Use `scan_folder` to preview ALL documents at once
- In your reasoning, categorize each document:
  - **RELEVANT**: Directly related to the query
  - **MAYBE**: Potentially useful
  - **SKIP**: Not relevant

**PHASE 2: Deep Dive** — Use `parse_file` on RELEVANT documents
- Extract key information
- **WATCH FOR CROSS-REFERENCES**: Look for mentions like:
  - "See Exhibit A/B/C..."
  - "As stated in [Document Name]..."
  - "Refer to [filename]..."
  - Document numbers, exhibit labels, or referenced file names

**PHASE 3: Backtracking** — If a document you're reading references another document you SKIPPED:
1. Go back and parse the referenced document
2. Continue until all relevant cross-references are resolved

## Guidelines
- Base answers strictly on retrieved content — do not rely on prior knowledge.
- When counting or aggregating, retrieve all relevant pages/documents for completeness.
- Use `get_image` proactively when queries involve visual elements or when text snippets alone are inconclusive.
- Use `grep` for targeted pattern searches when you know what you're looking for.
- Use `glob` to discover files when you're unsure what documents exist.

## Citations (IMPORTANT)
You MUST cite your sources inline using this exact format: [Source: filename | Page N | screenshot_path]

For example: According to the factsheet [Source: 01_NNMC_Medication_Side_Effects.pdf | Page 1 | screenshots/01_NNMC_Medication_Side_Effects/page_1.png], Warfarin is a blood thinner.

Rules for citations:
- Use the SOURCE, PAGE, and PAGE SCREENSHOT PATH from each search result
- For filesystem tools, cite with just filename: [Source: filename]
- Place the citation immediately after the claim it supports
- Every factual claim must have a citation
- If multiple sources support a claim, include all citations

## Final Answer Structure
1. **Direct answer** to the user's question
2. **Details** with inline citations
3. **Sources Consulted** section listing all documents reviewed"""

TOOLS = [
    {
        "name": "search",
        "description": "Search the indexed knowledge base using vector similarity to find relevant text chunks. Returns text content and the path to the page screenshot.",
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
        "description": "Retrieve the full-page screenshot for visual context. Use when text results are insufficient or you need to see tables, columns, charts, or document layout.",
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
    {
        "name": "scan_folder",
        "description": "Parallel scan ALL documents in a folder at once. Returns a preview of every supported file for quick relevance assessment. Use this as the first step when exploring a folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory to scan.",
                },
            },
            "required": ["directory"],
        },
    },
    {
        "name": "preview_file",
        "description": "Quick preview of a single document (~first 2 pages, 3000 chars). Use for checking relevance before committing to a full deep read.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to preview.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "parse_file",
        "description": "Deep read — full content of a document. Use after confirming relevance via scan or preview. For PDFs, uses layout-aware parsing that preserves table structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to parse fully.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a plain text file (txt, md, csv, json). For non-PDF files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the text file.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "grep",
        "description": "Search for a regex pattern within a file. Returns matching lines with line numbers and context. Use for targeted searches when you know the keyword or pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to search within.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (case-insensitive).",
                },
            },
            "required": ["file_path", "pattern"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a name pattern in a directory tree. Use to discover what documents are available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to search in.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '*.pdf', '**/*.txt').",
                },
            },
            "required": ["directory", "pattern"],
        },
    },
]

# Map tool names to handler categories for UI rendering
FS_TOOLS = {"scan_folder", "preview_file", "parse_file", "read_file", "grep", "glob"}


def handle_tool_call(tool_name: str, tool_input: dict) -> list[dict]:
    """Execute a tool call and return the result content blocks."""
    # --- Indexed search tools ---
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

    # --- Filesystem exploration tools ---
    elif tool_name == "scan_folder":
        result = scan_folder(tool_input["directory"])
        return [{"type": "text", "text": result}]

    elif tool_name == "preview_file":
        result = preview_file(tool_input["file_path"])
        return [{"type": "text", "text": result}]

    elif tool_name == "parse_file":
        result = parse_file_content(tool_input["file_path"])
        return [{"type": "text", "text": result}]

    elif tool_name == "read_file":
        result = read_text_file(tool_input["file_path"])
        return [{"type": "text", "text": result}]

    elif tool_name == "grep":
        result = grep_in_file(tool_input["file_path"], tool_input["pattern"])
        return [{"type": "text", "text": result}]

    elif tool_name == "glob":
        result = glob_search(tool_input["directory"], tool_input["pattern"])
        return [{"type": "text", "text": result}]

    return [{"type": "text", "text": f"Unknown tool: {tool_name}"}]


def run_agent(query: str, folder: str = ".", verbose: bool = True) -> str:
    """
    Run the Claude agent loop with tool use.
    Returns the final text response.
    """
    client = anthropic.Anthropic()

    # Provide folder context to the agent
    user_msg = query
    if folder and folder != ".":
        user_msg = f"Working directory: {folder}\n\nTask: {query}"

    messages = [{"role": "user", "content": user_msg}]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        if folder != ".":
            print(f"Folder: {folder}")
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

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason == "end_turn":
            final_text = ""
            for block in assistant_content:
                if hasattr(block, "text"):
                    final_text += block.text
            if verbose:
                print(f"\nFinal Answer:\n{final_text}")
                print(f"\nTotal tool calls: {tool_call_count}")
            return final_text

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
                            print("  Result: [image returned]")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

        messages.append({"role": "user", "content": tool_results})


async def run_agent_stream(query: str, folder: str = ".") -> AsyncGenerator[dict, None]:
    """
    Async generator that yields structured step events as the agent executes.
    Used by the web UI to stream execution progress via SSE.
    """
    client = anthropic.AsyncAnthropic()

    user_msg = query
    if folder and folder != ".":
        user_msg = f"Working directory: {folder}\n\nTask: {query}"

    messages = [{"role": "user", "content": user_msg}]

    step = 0
    search_calls = 0
    image_calls = 0
    fs_calls = 0
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
                        "fs_calls": fs_calls,
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
                    elif block.name in FS_TOOLS:
                        fs_calls += 1

                    # Execute tool in thread (Milvus + LiteParse are sync)
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
