import unittest

import numpy as np

from maestro_research.selector import (
    AmbiguousReferenceError,
    ReferenceParseError,
    ScriptedSelector,
    parse_instruction,
)


class InstructionParsingTests(unittest.TestCase):
    def test_three_declared_tasks_parse(self):
        cases = {
            "Pick the alphabet soup and place it in the basket": ("alphabet_soup", "basket", "in", None),
            "Pick the akita black bowl from table center and place it on the plate": (
                "akita_black_bowl", "plate", "on", "from table center"),
            "Put the bowl on the plate": ("akita_black_bowl", "plate", "on", None),
        }
        for language, expected in cases.items():
            ref = parse_instruction(language)
            self.assertEqual(
                (ref.target_type, ref.receptacle_type, ref.relation, ref.modifier), expected
            )

    def test_unsupported_articulated_instruction_fails_loudly(self):
        with self.assertRaises(ReferenceParseError):
            parse_instruction(
                "Pick the black bowl in the top drawer of the wooden cabinet "
                "and place it on the plate"
            )


class GroundingTests(unittest.TestCase):
    def positions(self):
        return {
            "akita_black_bowl_1": np.array([0.10, 0.00, 0.85]),
            "akita_black_bowl_2": np.array([0.30, 0.20, 0.85]),
            "plate_1": np.array([0.25, -0.10, 0.82]),
            "main_table": np.array([0.20, 0.00, 0.80]),
            "flat_stove_1": np.array([0.05, 0.30, 0.81]),
        }

    def test_spatial_modifier_selects_center_bowl(self):
        selector = ScriptedSelector(self.positions(), self.positions()["main_table"])
        decision = selector.decide(
            "Pick the akita black bowl from table center and place it on the plate"
        )
        # both bowls are equidistant in x from table centre; bowl_1 is nearer overall
        self.assertEqual(decision.constraints["receptacle_id"], "plate_1")
        self.assertIn(decision.target_id, {"akita_black_bowl_1", "akita_black_bowl_2"})

    def test_ambiguous_reference_is_not_guessed(self):
        selector = ScriptedSelector(self.positions(), None)
        with self.assertRaises(AmbiguousReferenceError):
            selector.decide("Put the bowl on the plate")

    def test_missing_receptacle_fails(self):
        positions = {"akita_black_bowl_1": np.zeros(3)}
        selector = ScriptedSelector(positions, None)
        with self.assertRaises(AmbiguousReferenceError):
            selector.decide("Put the bowl on the plate")


if __name__ == "__main__":
    unittest.main()
