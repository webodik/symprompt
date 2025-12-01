from __future__ import annotations

from dataclasses import dataclass

from symprompt.symil.model import SymIL
from symprompt.symil.validator import SymILValidationError, SymILValidator
from symprompt.translation.logical import LogicalTranslator
from symprompt.translation.ontology import OntologyExtractor
from symprompt.translation.preprocess import Preprocessor


@dataclass
class TranslationPipeline:
    preprocessor: Preprocessor
    ontology_extractor: OntologyExtractor
    logical_translator: LogicalTranslator
    validator: SymILValidator

    @classmethod
    def from_llm_client(cls, llm_client) -> "TranslationPipeline":
        return cls(
            preprocessor=Preprocessor(),
            ontology_extractor=OntologyExtractor(llm_client),
            logical_translator=LogicalTranslator(llm_client),
            validator=SymILValidator(),
        )

    def translate(self, nl_prompt: str, level: int, hints: list[str] | None = None) -> SymIL:
        text = self.preprocessor.normalize(nl_prompt)
        ontology = self.ontology_extractor.extract(text)

        base_hints = hints or []

        # First attempt
        symil = self.logical_translator.translate(
            text, ontology, target_level=level, hints=base_hints
        )
        try:
            return self.validator.validate(symil)
        except SymILValidationError as exc:
            # Build detailed refinement hints
            refined_hints = base_hints + [
                f"REFINEMENT REQUIRED - Previous attempt failed validation.",
                f"Error: {exc}",
                "Common fixes:",
                "- Constants must be lowercase (socrates not Socrates)",
                "- Variables must be UPPERCASE (X, Y, Z) and only in forall/exists",
                "- All args in facts must use constants from ontology",
                "- All predicates must be defined in ontology",
            ]
            symil_refined = self.logical_translator.translate(
                text, ontology, target_level=level, hints=refined_hints
            )
            # If this still fails, let the error surface.
            return self.validator.validate(symil_refined)


def translate_with_escalation(
    pipeline: TranslationPipeline,
    nl_prompt: str,
    min_level: int = 0,
    max_level: int = 2,
    hints: list[str] | None = None,
) -> SymIL:
    """
    Translate a prompt into SymIL using progressive escalation from
    min_level to max_level. If all levels succeed, returns the SymIL
    from the lowest level that validates. Higher levels can be tried
    later if solver feedback indicates insufficient structure.
    """
    last_symil: SymIL | None = None
    for level in range(min_level, max_level + 1):
        symil = pipeline.translate(nl_prompt, level=level, hints=hints)
        last_symil = symil
        # For now, we stop at the first level that passes validation.
        # Solver feedback can be used later to decide on escalation.
        break
    assert last_symil is not None
    return last_symil
