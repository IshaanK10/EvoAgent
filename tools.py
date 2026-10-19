
import ast
import math
import operator
from pathlib import Path

WORKSPACE = Path(__file__).parent / "workspace" / "notes"
WORKSPACE.mkdir(parents=True, exist_ok=True)

MAX_EXPRESSION_LENGTH = 200
MAX_NOTE_SIZE = 20_000

# Only these arithmetic operators are permitted.
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node):
    """Evaluate a restricted arithmetic expression without eval()."""

    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ValueError("Only numeric constants are allowed.")

        if abs(node.value) > 1e12:
            raise ValueError("Number is too large.")

        return node.value

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)

        if operator_type not in BINARY_OPERATORS:
            raise ValueError("Operator is not allowed.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        # Prevent excessively large exponent calculations.
        if isinstance(node.op, ast.Pow):
            if abs(right) > 10 or abs(left) > 1e6:
                raise ValueError("Exponent calculation is too large.")

        result = BINARY_OPERATORS[operator_type](left, right)

        if not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("Result is outside the permitted range.")

        return result

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)

        if operator_type not in UNARY_OPERATORS:
            raise ValueError("Unary operator is not allowed.")

        return UNARY_OPERATORS[operator_type](_evaluate(node.operand))

    raise ValueError("Only basic arithmetic expressions are allowed.")


def calculate(expression):
    if not isinstance(expression, str):
        raise ValueError("Expression must be text.")

    if not expression or len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is empty or too long.")

    tree = ast.parse(expression, mode="eval")
    result = _evaluate(tree.body)

    return str(result)


def list_notes():
    """List Markdown and text notes in the dedicated notes folder."""

    return sorted(
        path.name
        for path in WORKSPACE.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".txt", ".md"}
        and not path.is_symlink()
    )


def read_note(filename):
    """Read a single note from the dedicated notes folder."""

    if not isinstance(filename, str):
        raise ValueError("Filename must be text.")

    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("Provide a filename only, without a path.")

    if Path(filename).suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Only .txt and .md files can be read.")

    path = WORKSPACE / filename

    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Note not found.")

    if path.stat().st_size > MAX_NOTE_SIZE:
        raise ValueError("Note exceeds the maximum permitted size.")

    return path.read_text(encoding="utf-8")


def execute_tool(name, arguments):
    """Dispatch only explicitly registered tools."""

    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    if name == "calculator":
        if set(arguments) != {"expression"}:
            raise ValueError("Expected only the expression argument.")

        return calculate(arguments["expression"])

    if name == "list_notes":
        if arguments:
            raise ValueError("list_notes takes no arguments.")

        return list_notes()

    if name == "read_note":
        if set(arguments) != {"filename"}:
            raise ValueError("Expected only the filename argument.")

        return read_note(arguments["filename"])

    raise ValueError(f"Tool is not permitted: {name}")
