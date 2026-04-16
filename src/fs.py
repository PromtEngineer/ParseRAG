"""
Filesystem exploration tools for the combined agent.
Provides scan_folder, preview_file, parse_file, read, grep, and glob
capabilities adapted from FsExplorer, using LiteParse for PDF parsing.
"""
import os
import re
import fnmatch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .processing import parse_pdf

# Supported file extensions for document scanning
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".csv", ".json"})
DEFAULT_PREVIEW_CHARS = 3000
DEFAULT_SCAN_PREVIEW_CHARS = 1500
DEFAULT_MAX_WORKERS = 4

# In-memory document cache keyed by "abs_path:mtime"
_DOCUMENT_CACHE: dict[str, str] = {}


def _get_cached_or_parse(file_path: str) -> str:
    """Parse a file with caching based on path + modification time."""
    abs_path = os.path.abspath(file_path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        return f"Error: File not found: {file_path}"
    cache_key = f"{abs_path}:{mtime}"
    if cache_key in _DOCUMENT_CACHE:
        return _DOCUMENT_CACHE[cache_key]

    ext = Path(abs_path).suffix.lower()
    if ext == ".pdf":
        try:
            pages = parse_pdf(abs_path)
            content = "\n\n".join(
                f"--- Page {pnum} ---\n{text}" for pnum, text in sorted(pages.items())
            )
        except Exception as e:
            content = f"Error parsing {file_path}: {e}"
    elif ext in {".txt", ".md", ".csv", ".json"}:
        try:
            content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            content = f"Error reading {file_path}: {e}"
    else:
        content = f"Unsupported file extension: {ext}"

    _DOCUMENT_CACHE[cache_key] = content
    return content


def describe_dir_content(directory: str) -> str:
    """List all files and folders in a directory with sizes."""
    abs_dir = os.path.abspath(directory)
    if not os.path.isdir(abs_dir):
        return f"Error: Not a directory: {directory}"

    entries = []
    try:
        items = sorted(os.listdir(abs_dir))
    except PermissionError:
        return f"Error: Permission denied: {directory}"

    folders = []
    files = []
    for item in items:
        if item.startswith("."):
            continue
        full = os.path.join(abs_dir, item)
        if os.path.isdir(full):
            count = len([f for f in os.listdir(full) if not f.startswith(".")])
            folders.append(f"  [DIR] {item}/ ({count} items)")
        elif os.path.isfile(full):
            size = os.path.getsize(full)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            ext = Path(item).suffix.lower()
            supported = "*" if ext in SUPPORTED_EXTENSIONS else " "
            files.append(f"  [{supported}] {item} ({size_str})")

    entries.append(f"Directory: {abs_dir}")
    entries.append(f"  {len(folders)} folders, {len(files)} files")
    entries.append("")
    if folders:
        entries.append("Folders:")
        entries.extend(folders)
        entries.append("")
    if files:
        entries.append("Files (* = parseable):")
        entries.extend(files)

    return "\n".join(entries)


def _preview_single_file(file_path: str, max_chars: int) -> tuple[str, str]:
    """Preview a single file (for parallel scanning). Returns (filename, preview)."""
    name = Path(file_path).name
    try:
        content = _get_cached_or_parse(file_path)
        if content.startswith("Error"):
            return name, content
        preview = content[:max_chars]
        total = len(content)
        lines = preview.split("\n")[:30]
        trimmed = "\n".join(lines)
        if total > max_chars:
            trimmed += f"\n... ({total:,} total characters)"
        return name, trimmed
    except Exception as e:
        return name, f"Error: {e}"


def scan_folder(directory: str) -> str:
    """
    Parallel scan of all supported documents in a folder.
    Returns a preview of every file for quick relevance assessment.
    """
    abs_dir = os.path.abspath(directory)
    if not os.path.isdir(abs_dir):
        return f"Error: Not a directory: {directory}"

    files_to_scan = []
    for fname in sorted(os.listdir(abs_dir)):
        fpath = os.path.join(abs_dir, fname)
        if os.path.isfile(fpath):
            ext = Path(fname).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                files_to_scan.append(fpath)

    if not files_to_scan:
        return f"No supported documents found in {abs_dir}"

    results: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        futures = {
            executor.submit(_preview_single_file, fp, DEFAULT_SCAN_PREVIEW_CHARS): fp
            for fp in files_to_scan
        }
        for future in futures:
            name, preview = future.result()
            results.append((name, preview))

    results.sort(key=lambda x: x[0])

    parts = [f"=== FOLDER SCAN: {abs_dir} ===", f"Documents found: {len(results)}", ""]
    for name, preview in results:
        parts.append(f"--- {name} ---")
        parts.append(preview)
        parts.append("")

    parts.append(
        "Use 'parse_file' for full content of relevant documents, "
        "or 'preview_file' for a longer preview."
    )
    return "\n".join(parts)


def preview_file(file_path: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """Quick preview of a document (~first 1-2 pages)."""
    content = _get_cached_or_parse(file_path)
    if content.startswith("Error"):
        return content
    preview = content[:max_chars]
    if len(content) > max_chars:
        preview += f"\n\n... (truncated, {len(content):,} total characters)"
    return preview


def parse_file_content(file_path: str) -> str:
    """Full content parse of a document."""
    return _get_cached_or_parse(file_path)


def read_text_file(file_path: str) -> str:
    """Read a plain text file directly."""
    abs_path = os.path.abspath(file_path)
    if not os.path.isfile(abs_path):
        return f"Error: File not found: {file_path}"
    try:
        return Path(abs_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {file_path}: {e}"


def grep_in_file(file_path: str, pattern: str) -> str:
    """Search for a regex pattern in a file. Returns matching lines with context."""
    content = _get_cached_or_parse(file_path)
    if content.startswith("Error"):
        return content

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex pattern '{pattern}': {e}"

    lines = content.split("\n")
    matches = []
    for i, line in enumerate(lines, 1):
        if regex.search(line):
            # Include 1 line of context before and after
            start = max(0, i - 2)
            end = min(len(lines), i + 1)
            context_lines = []
            for j in range(start, end):
                prefix = ">>>" if j == i - 1 else "   "
                context_lines.append(f"  {prefix} L{j + 1}: {lines[j]}")
            matches.append("\n".join(context_lines))

    if not matches:
        return f"No matches for pattern '{pattern}' in {Path(file_path).name}"

    result = [f"Found {len(matches)} match(es) for '{pattern}' in {Path(file_path).name}:", ""]
    result.extend(m + "\n" for m in matches[:50])  # cap at 50 matches
    if len(matches) > 50:
        result.append(f"... and {len(matches) - 50} more matches")
    return "\n".join(result)


def glob_search(directory: str, pattern: str) -> str:
    """Find files matching a glob pattern in a directory."""
    abs_dir = os.path.abspath(directory)
    if not os.path.isdir(abs_dir):
        return f"Error: Not a directory: {directory}"

    matches = []
    for root, dirs, files in os.walk(abs_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in sorted(files):
            if fnmatch.fnmatch(fname.lower(), pattern.lower()):
                rel = os.path.relpath(os.path.join(root, fname), abs_dir)
                size = os.path.getsize(os.path.join(root, fname))
                matches.append(f"  {rel} ({size:,} bytes)")

    if not matches:
        return f"No files matching '{pattern}' in {abs_dir}"

    return f"Files matching '{pattern}' in {abs_dir}:\n" + "\n".join(matches)
