#!/usr/bin/env python3
"""
ParseRAG: PDF QA Agent powered by LiteParse + Milvus + Claude

Usage:
    python main.py process <pdf_file>     Parse, embed, and store a PDF
    python main.py search <query>         Vector search for relevant chunks
    python main.py agent <query>          Ask the Claude agent a question
    python main.py eval                   Run the 20-question eval suite
"""
import argparse
import json
import sys
from pathlib import Path


def cmd_process(args):
    from src.processing import pipeline
    pipeline(args.file, screenshot_dir=args.screenshot_dir, reset=args.reset)


def cmd_index(args):
    from src.processing import pipeline_directory
    pipeline_directory(args.directory, screenshot_dir=args.screenshot_dir, reset=args.reset)


def cmd_search(args):
    from src.search import search
    results = search(args.query, limit=args.limit)
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (similarity: {r['distance']:.4f}) ---")
        print(f"Page: {r['page_num']} | Screenshot: {r['image_path']}")
        print(f"Text:\n{r['text'][:500]}...")


def cmd_agent(args):
    from src.agent import run_agent
    run_agent(args.query, verbose=True)


def cmd_eval(args):
    from src.agent import run_agent

    gold_path = Path("data/gold.json")
    if not gold_path.exists():
        print(f"Eval file not found: {gold_path}")
        sys.exit(1)

    with open(gold_path) as f:
        questions = json.load(f)

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]

    print(f"Running eval on {len(questions)} questions...\n")
    results = []

    for q in questions:
        print(f"\n{'='*60}")
        print(f"[{q['id']}] ({q['category']}) {q['question']}")
        print(f"Expected: {q['expected_answer']}")

        try:
            answer = run_agent(q["question"], verbose=not args.quiet)
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected": q["expected_answer"],
                "actual": answer,
            })
            print(f"\nAgent answer: {answer[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "id": q["id"],
                "category": q["category"],
                "error": str(e),
            })

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print(f"\nCompleted {len(results)}/{len(questions)} questions.")


def main():
    parser = argparse.ArgumentParser(
        description="ParseRAG: PDF QA Agent powered by LiteParse + Milvus + Claude"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # process
    p_process = subparsers.add_parser("process", help="Parse, embed, and store a single PDF")
    p_process.add_argument("file", help="Path to the PDF file")
    p_process.add_argument(
        "-s", "--screenshot-dir", default="screenshots",
        help="Directory for page screenshots (default: screenshots/)",
    )
    p_process.add_argument(
        "--reset", action="store_true",
        help="Drop existing collection before processing",
    )

    # index (batch)
    p_index = subparsers.add_parser("index", help="Index all PDFs in a directory")
    p_index.add_argument("directory", help="Directory containing PDF files")
    p_index.add_argument(
        "-s", "--screenshot-dir", default="screenshots",
        help="Directory for page screenshots (default: screenshots/)",
    )
    p_index.add_argument(
        "--reset", action="store_true", default=True,
        help="Drop existing collection before indexing (default: true)",
    )
    p_index.add_argument(
        "--no-reset", action="store_false", dest="reset",
        help="Keep existing data and add new documents",
    )

    # search
    p_search = subparsers.add_parser("search", help="Vector search for chunks")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument(
        "-l", "--limit", type=int, default=3,
        help="Number of results (default: 3)",
    )

    # agent
    p_agent = subparsers.add_parser("agent", help="Ask the Claude agent")
    p_agent.add_argument("query", help="Question to ask")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation suite")
    p_eval.add_argument("--category", help="Filter by question category")
    p_eval.add_argument("--output", help="Save results to JSON file")
    p_eval.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "agent":
        cmd_agent(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
