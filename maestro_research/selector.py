"""Instruction -> grounded target/receptacle selection.

This milestone ships a deterministic, instruction-driven selector. It is a
control baseline, not VLM inference. It never reads BDDL goal facts or the
success oracle. In oracle-geometry diagnostic mode it receives simulator body
poses by declaration; in RGB-D mode it receives perceived estimates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import TaskDecision

TYPE_SYNONYMS = {
    "alphabet soup": "alphabet_soup",
    "akita black bowl": "akita_black_bowl",
    "black bowl": "akita_black_bowl",
    "bowl": "akita_black_bowl",
    "basket": "basket",
    "plate": "plate",
    "ramekin": "glazed_rim_porcelain_ramekin",
    "glazed rim porcelain ramekin": "glazed_rim_porcelain_ramekin",
    "cookie box": "cookies",
    "cookies": "cookies",
    "stove": "flat_stove",
    "flat stove": "flat_stove",
    "wine bottle": "wine_bottle",
    "cream cheese": "cream_cheese",
    "wooden cabinet": "wooden_cabinet",
    "milk": "milk",
    "salad dressing": "salad_dressing",
    "bbq sauce": "bbq_sauce",
    "ketchup": "ketchup",
    "tomato sauce": "tomato_sauce",
    "butter": "butter",
    "chocolate pudding": "chocolate_pudding",
    "orange juice": "orange_juice",
}

MODIFIERS = (
    "in the top drawer of the wooden cabinet",
    "between the plate and the ramekin",
    "on the wooden cabinet",
    "next to the cookie box",
    "next to the ramekin",
    "on the cookie box",
    "on the ramekin",
    "on the stove",
    "from table center",
)

MODIFIER_LANDMARKS = {
    "between the plate and the ramekin": ("plate", "glazed_rim_porcelain_ramekin"),
    "on the wooden cabinet": ("wooden_cabinet",),
    "next to the cookie box": ("cookies",),
    "next to the ramekin": ("glazed_rim_porcelain_ramekin",),
    "on the cookie box": ("cookies",),
    "on the ramekin": ("glazed_rim_porcelain_ramekin",),
    "on the stove": ("flat_stove",),
}

UNSUPPORTED_MODIFIERS = {"in the top drawer of the wooden cabinet"}


class ReferenceParseError(ValueError):
    pass


class AmbiguousReferenceError(ValueError):
    pass


@dataclass
class Reference:
    target_type: str
    receptacle_type: str
    relation: str
    modifier: Optional[str]
    language: str
    raw_target_phrase: str
    raw_receptacle_phrase: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip("."))


def _lookup_type(phrase: str) -> str:
    phrase = _normalize(phrase)
    if phrase in TYPE_SYNONYMS:
        return TYPE_SYNONYMS[phrase]
    for key in sorted(TYPE_SYNONYMS, key=len, reverse=True):
        if phrase.endswith(key) or phrase.startswith(key) or key in phrase:
            return TYPE_SYNONYMS[key]
    raise ReferenceParseError(f"no object-type synonym for {phrase!r}")


def parse_instruction(language: str) -> Reference:
    text = _normalize(language)
    if " and place it " in text:
        lhs, rhs = text.split(" and place it ", 1)
    elif text.startswith("put the ") and " on the " in text:
        lhs, rhs = text.split(" on the ", 1)
        lhs = "put the " + lhs
        rhs = "on the " + rhs
    else:
        raise ReferenceParseError(f"unsupported instruction template: {language!r}")

    lhs = re.sub(r"^pick up the |^pick the |^put the ", "", lhs).strip()
    modifier = None
    for candidate in sorted(MODIFIERS, key=len, reverse=True):
        if lhs.endswith(" " + candidate):
            modifier = candidate
            lhs = lhs[: -len(candidate) - 1].strip()
            break
    if modifier in UNSUPPORTED_MODIFIERS:
        raise ReferenceParseError(f"modifier {modifier!r} requires articulated skills not in this milestone")
    if not lhs:
        raise ReferenceParseError(f"empty target phrase in {language!r}")

    match = re.match(r"^(in|on) the (.+)$", rhs.strip())
    if not match:
        raise ReferenceParseError(f"unsupported relation phrase: {rhs!r}")
    relation, receptacle_phrase = match.group(1), match.group(2).strip()
    return Reference(
        target_type=_lookup_type(lhs),
        receptacle_type=_lookup_type(receptacle_phrase),
        relation=relation,
        modifier=modifier,
        language=language,
        raw_target_phrase=lhs,
        raw_receptacle_phrase=receptacle_phrase,
    )


def _bodies_of_type(body_names: Sequence[str], object_type: str) -> List[str]:
    prefix = object_type + "_"
    return sorted(name for name in body_names if name.startswith(prefix))


def _landmark_point(landmarks: Dict[str, str], positions: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    points = []
    for label in landmarks:
        body = _bodies_of_type(list(positions), label)
        if not body:
            return None
        points.append(np.asarray(positions[body[0]], dtype=np.float64))
    return np.mean(points, axis=0)


def resolve_reference(
    reference: Reference,
    positions: Dict[str, np.ndarray],
    table_center: Optional[np.ndarray] = None,
) -> Tuple[str, str]:
    names = list(positions)
    candidates = _bodies_of_type(names, reference.target_type)
    receptacles = _bodies_of_type(names, reference.receptacle_type)
    if not candidates:
        raise AmbiguousReferenceError(f"no body of type {reference.target_type!r} in scene")
    if not receptacles:
        raise AmbiguousReferenceError(f"no receptacle of type {reference.receptacle_type!r} in scene")
    receptacle = receptacles[0]

    if len(candidates) == 1:
        target = candidates[0]
    else:
        if reference.modifier is None:
            raise AmbiguousReferenceError(
                f"{len(candidates)} candidates for {reference.target_type!r} and no disambiguating modifier"
            )
        if reference.modifier == "from table center":
            if table_center is None:
                raise AmbiguousReferenceError("table center is required for 'from table center'")
            anchor = np.asarray(table_center, dtype=np.float64)
        else:
            anchor = _landmark_point(MODIFIER_LANDMARKS[reference.modifier], positions)
            if anchor is None:
                raise AmbiguousReferenceError(f"landmark for {reference.modifier!r} not found")
        target = min(candidates, key=lambda name: float(np.linalg.norm(positions[name][:2] - anchor[:2])))
    return target, receptacle


class ScriptedSelector:
    name = "scripted-instruction-selector"

    def __init__(self, positions: Dict[str, np.ndarray], table_center: Optional[np.ndarray] = None):
        self.positions = {k: np.asarray(v, dtype=np.float64) for k, v in positions.items()}
        self.table_center = table_center

    def decide(self, language: str, stage: str = "SELECT") -> TaskDecision:
        reference = parse_instruction(language)
        target, receptacle = resolve_reference(reference, self.positions, self.table_center)
        return TaskDecision(
            target_id=target,
            constraints={
                "receptacle_id": receptacle,
                "relation": reference.relation,
                "modifier": reference.modifier,
                "selector": self.name,
                "selector_kind": "scripted_control",
            },
            stage=stage,
        )
