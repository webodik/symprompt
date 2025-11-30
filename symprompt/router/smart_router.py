from __future__ import annotations

from dataclasses import dataclass

from symprompt.router.features import PromptFeatures, extract_features
from symprompt.symil.profiles import SymILProfile, get_profile


@dataclass
class RoutingDecision:
    tier: int
    symil_level: int
    profile_name: str
    preferred_solver: str = "z3"


class SmartRouter:
    """
    Heuristic router that selects tier, SymIL level, and profile based on
    prompt features. The thresholds and mapping are intended to be evolved
    by OpenEvolve in later phases.
    """

    def route(self, prompt: str, context) -> RoutingDecision:
        features: PromptFeatures = extract_features(prompt)

        profile: SymILProfile = get_profile(features.domain)

        # BYPASS: pure LLM path for very simple, non-logical prompts.
        # Do not bypass when math keywords are present.
        if (
            not features.has_logic_keywords
            and not features.has_numbers
            and features.complexity < 0.2
        ):
            return RoutingDecision(
                tier=0,
                symil_level=0,
                profile_name=profile.name,
                preferred_solver="z3",
            )

        if features.domain == "planning":
            return RoutingDecision(
                tier=2,
                symil_level=max(profile.default_level, 2),
                profile_name=profile.name,
                preferred_solver=profile.preferred_solver,
            )

        if features.complexity < 0.3 and features.logical_depth < 2:
            tier = 1
            symil_level = 0
        elif features.complexity < 0.6 and features.constraint_count < 3:
            tier = 1
            symil_level = profile.default_level
        else:
            tier = 2
            symil_level = max(profile.default_level, 1)

        return RoutingDecision(
            tier=tier,
            symil_level=symil_level,
            profile_name=profile.name,
            preferred_solver=profile.preferred_solver,
        )
