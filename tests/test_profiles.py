from __future__ import annotations

import pytest

from symprompt.symil.profiles import SymILProfile, get_profile, list_profiles


def test_get_profile_returns_syllogism_profile() -> None:
    profile = get_profile("syllogism")
    assert profile.name == "syllogism"
    assert profile.preferred_solver == "scallop"
    assert profile.default_level == 1
    assert "L1" in profile.allowed_constructs


def test_get_profile_returns_math_profile() -> None:
    profile = get_profile("math")
    assert profile.name == "math"
    assert profile.preferred_solver == "z3"
    assert profile.default_level == 0


def test_get_profile_returns_planning_profile() -> None:
    profile = get_profile("planning")
    assert profile.name == "planning"
    assert profile.preferred_solver == "clingo"
    assert profile.default_level == 2


def test_get_profile_returns_legal_profile() -> None:
    profile = get_profile("legal")
    assert profile.name == "legal"
    assert profile.preferred_solver == "clingo"
    assert profile.default_level == 2


def test_get_profile_returns_uncertain_profile() -> None:
    profile = get_profile("uncertain")
    assert profile.name == "uncertain"
    assert profile.preferred_solver == "scallop"
    assert profile.default_level == 1


def test_get_profile_raises_key_error_for_unknown_profile() -> None:
    with pytest.raises(KeyError):
        get_profile("unknown_profile")


def test_list_profiles_returns_all_profiles() -> None:
    profiles = list_profiles()
    assert len(profiles) == 5
    names = {p.name for p in profiles}
    assert names == {"syllogism", "math", "planning", "legal", "uncertain"}


def test_all_profiles_have_valid_constructs() -> None:
    valid_constructs = {"L0", "L1", "L2"}
    for profile in list_profiles():
        for construct in profile.allowed_constructs:
            assert construct in valid_constructs, (
                f"Profile {profile.name} has invalid construct: {construct}"
            )


def test_all_profiles_have_valid_default_levels() -> None:
    for profile in list_profiles():
        assert profile.default_level in {0, 1, 2}, (
            f"Profile {profile.name} has invalid default_level: {profile.default_level}"
        )


def test_all_profiles_have_translation_hints() -> None:
    for profile in list_profiles():
        assert isinstance(profile.translation_hints, list)
        assert len(profile.translation_hints) > 0, (
            f"Profile {profile.name} has no translation hints"
        )


def test_all_profiles_have_predicate_vocabulary() -> None:
    for profile in list_profiles():
        assert isinstance(profile.predicate_vocabulary, list)
        assert len(profile.predicate_vocabulary) > 0, (
            f"Profile {profile.name} has no predicate vocabulary"
        )


def test_profile_dataclass_fields() -> None:
    profile = SymILProfile(
        name="test",
        predicate_vocabulary=["pred1"],
        allowed_constructs=["L0"],
        preferred_solver="z3",
        default_level=0,
        translation_hints=["hint1"],
    )
    assert profile.name == "test"
    assert profile.predicate_vocabulary == ["pred1"]
    assert profile.allowed_constructs == ["L0"]
    assert profile.preferred_solver == "z3"
    assert profile.default_level == 0
    assert profile.translation_hints == ["hint1"]


def test_profile_dataclass_default_hints() -> None:
    profile = SymILProfile(
        name="minimal",
        predicate_vocabulary=["p"],
        allowed_constructs=["L0"],
        preferred_solver="z3",
        default_level=0,
    )
    assert profile.translation_hints == []
