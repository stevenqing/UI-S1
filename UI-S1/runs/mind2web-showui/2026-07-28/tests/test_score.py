import unittest

from score import parse_prediction, token_f1


class ScoreTest(unittest.TestCase):
    def test_released_parser_shape(self) -> None:
        parsed = parse_prediction(
            "{'action': 'TYPE', 'value': 'New York', 'position': [0.25, 0.5]}"
        )
        self.assertEqual(parsed["action"], "TYPE")
        self.assertEqual(parsed["value"], "New York")
        self.assertEqual(parsed["position"], [0.25, 0.5])

    def test_token_set_f1(self) -> None:
        self.assertEqual(token_f1("3 new york", "3 york new"), 1.0)
        self.assertAlmostEqual(token_f1("3 new", "3 new york"), 0.8)


if __name__ == "__main__":
    unittest.main()