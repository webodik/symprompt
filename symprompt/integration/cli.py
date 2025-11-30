from __future__ import annotations

import argparse
import sys

from symprompt.integration.router_adapter import route_and_solve
from symprompt.integration.verification import verify_answer
from symprompt.llm.sync_client import build_default_sync_client


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SymPrompt CLI: route prompts through neuro-symbolic reasoning."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to process (if omitted, read from stdin).",
    )
    parser.add_argument(
        "--show-symil",
        action="store_true",
        help="Print the SymIL representation.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Ask the LLM to explain the solver result in natural language.",
    )
    parser.add_argument(
        "--mode",
        choices=["query", "verify", "refine"],
        default="query",
        help=(
            "Mode: 'query' (default) routes the prompt as a question; "
            "'verify' treats prompt as question and --answer as the answer to check; "
            "'refine' runs a hybrid verify-and-refine loop starting from an initial answer."
        ),
    )
    parser.add_argument(
        "--answer",
        help="In verify mode, the candidate answer to verify. If omitted, read from stdin.",
    )

    args = parser.parse_args(argv)

    if args.mode == "query":
        if args.prompt:
            text = args.prompt
        else:
            text = sys.stdin.read()
    elif args.mode == "verify":
        if not args.prompt:
            parser.error("verify mode requires a question as the positional prompt")

        text = args.prompt
    else:
        if not args.prompt:
            parser.error("refine mode requires a question as the positional prompt")
        text = args.prompt

    llm_client = build_default_sync_client()
    if args.mode == "query":
        route_result = route_and_solve(text, llm_client)

        print(f"Tier: {route_result.routing.tier}")
        print(f"Profile: {route_result.routing.profile_name}")
        print(f"SymIL level: {route_result.routing.symil_level}")
        status = str(route_result.solver_result.get("status"))
        print(f"Solver status: {status}")

        if args.show_symil:
            print("\nSymIL:")
            print(route_result.symil)

        if args.explain:
            explain_prompt = (
                f"The symbolic reasoning engine returned status '{status}' "
                f"for the prompt below. Explain what this means in natural language:\n\n"
                f"Prompt: {text}"
            )
            try:
                explanation = llm_client.complete(explain_prompt)
                print("\nExplanation:")
                print(explanation)
            except Exception as exc:
                print(f"\nExplanation failed: {exc}", file=sys.stderr)
    elif args.mode == "verify":
        if args.answer:
            answer = args.answer
        else:
            answer = sys.stdin.read()

        ver_result = verify_answer(text, answer, llm_client)
        print(f"Verification status: {ver_result.status}")

        if args.show_symil and ver_result.route_result.symil is not None:
            print("\nSymIL used for verification:")
            print(ver_result.route_result.symil)

        if args.explain:
            explain_prompt = (
                f"The verification engine returned status '{ver_result.status}' "
                f"for the following question and answer.\n\n"
                f"Question: {text}\nAnswer: {answer}\n\n"
                f"Explain in natural language whether and why this answer is logically consistent with the question."
            )
            try:
                explanation = llm_client.complete(explain_prompt)
                print("\nExplanation:")
                print(explanation)
            except Exception as exc:
                print(f"\nExplanation failed: {exc}", file=sys.stderr)
    else:
        # Hybrid Mode 3: verify and refine.
        from symprompt.integration.verification import verify_and_refine_answer

        question = text
        if args.answer:
            initial_answer = args.answer
        else:
            # Let the LLM propose an initial answer.
            initial_answer = llm_client.complete(question)

        refine_result = verify_and_refine_answer(
            question=question,
            initial_answer=initial_answer,
            llm_client=llm_client,
        )

        print(f"Final status: {refine_result.status}")
        print(f"Refinement attempts: {refine_result.attempts}")
        print("Final answer:")
        print(refine_result.answer)

        if args.show_symil and refine_result.route_result is not None:
            symil = getattr(refine_result.route_result, "symil", None)
            if symil is not None:
                print("\nSymIL used for final verification:")
                print(symil)

        if args.explain:
            explain_prompt = (
                f"The refinement engine returned status '{refine_result.status}' "
                f"for the following question and final answer.\n\n"
                f"Question: {question}\nFinal answer: {refine_result.answer}\n\n"
                f"Explain in natural language whether and why this answer is logically consistent with the question."
            )
            try:
                explanation = llm_client.complete(explain_prompt)
                print("\nExplanation:")
                print(explanation)
            except Exception as exc:
                print(f"\nExplanation failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
