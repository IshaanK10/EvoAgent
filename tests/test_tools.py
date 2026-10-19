
import unittest

import tools


class TestEvoAgentTools(unittest.TestCase):

    def test_calculator_addition(self):
        self.assertEqual(
            tools.calculate("2 + 3 * 4"),
            "14",
        )

    def test_calculator_parentheses(self):
        self.assertEqual(
            tools.calculate("(2 + 3) * 4"),
            "20",
        )

    def test_calculator_rejects_code(self):
        with self.assertRaises((ValueError, SyntaxError)):
            tools.calculate("__import__('os').system('whoami')")

    def test_calculator_rejects_unknown_operator(self):
        with self.assertRaises(ValueError):
            tools.calculate("2 << 4")

    def test_calculator_rejects_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            tools.calculate("10 / 0")

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            tools.execute_tool("run_shell", {"command": "ls"})

    def test_extra_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            tools.execute_tool(
                "calculator",
                {
                    "expression": "2 + 2",
                    "command": "ls",
                },
            )

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            tools.read_note("../main.py")

    def test_unsupported_file_type_is_rejected(self):
        with self.assertRaises(ValueError):
            tools.read_note("main.py")

    def test_missing_note_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            tools.read_note("definitely_missing_note.md")


if __name__ == "__main__":
    unittest.main()
