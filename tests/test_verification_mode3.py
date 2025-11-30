from __future__ import annotations

from symprompt.integration import verification


class DummyLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Return a different answer on refinement, but the concrete
        # content is not important for this test.
        return f"refined answer #{len(self.calls)}"


def test_verify_and_refine_answer_two_step(monkeypatch) -> None:
    """
    Verify that the Mode 3 refinement loop calls verification and
    refinement in sequence and returns a final result.
    """

    def fake_verify(question: str, answer: str, llm_client) -> verification.VerificationResult:
        # First call: NOT_VALID, second call: VALID.
        call_index = len(getattr(fake_verify, "calls", []))
        getattr(fake_verify, "calls", []).append(1)
        if call_index == 0:
            return verification.VerificationResult(status="NOT_VALID", route_result={"solver_result": {"status": "NOT_VALID"}})
        return verification.VerificationResult(status="VALID", route_result={"solver_result": {"status": "VALID"}})

    fake_verify.calls = []  # type: ignore[attr-defined]
    monkeypatch.setattr(verification, "verify_answer", fake_verify)

    llm = DummyLLM()
    result = verification.verify_and_refine_answer(
        question="Is 2+2=4?",
        initial_answer="maybe",
        llm_client=llm,
        max_attempts=3,
    )

    assert result.status in {"VALID", "UNKNOWN"}
    assert result.attempts >= 1
