import argparse
import ast
import operator
from dataclasses import dataclass


class SparkError(Exception):
    pass


@dataclass
class Block:
    keyword: str
    condition: str | None
    body: list[str]


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class Spark:
    """Interpreter for the tiny Spark language."""

    def __init__(self):
        self.variables = {}

    def run(self, source):
        lines = [line.strip() for line in source.splitlines()]
        self._run_lines(lines)

    def _run_lines(self, lines):
        index = 0
        while index < len(lines):
            line = lines[index]
            index += 1

            if not line or line.startswith("#"):
                continue

            if line.startswith("repeat ") or line.startswith("if "):
                block, index = self._collect_block(line, lines, index)
                self._run_block(block)
                continue

            if line == "end":
                raise SparkError("Unexpected 'end'.")

            self._run_statement(line)

    def _collect_block(self, header, lines, index):
        if not header.endswith(":"):
            raise SparkError(f"Block header must end with ':': {header}")

        keyword, condition = header[:-1].split(" ", 1)
        depth = 1
        body = []

        while index < len(lines):
            line = lines[index]
            index += 1

            if line.startswith("repeat ") or line.startswith("if "):
                depth += 1
            elif line == "end":
                depth -= 1
                if depth == 0:
                    return Block(keyword, condition, body), index

            body.append(line)

        raise SparkError(f"Missing 'end' for {keyword} block.")

    def _run_block(self, block):
        if block.keyword == "repeat":
            times = self._eval(block.condition)
            if not isinstance(times, int):
                raise SparkError("repeat needs a whole number.")
            for _ in range(times):
                self._run_lines(block.body)
            return

        if block.keyword == "if":
            if self._eval(block.condition):
                self._run_lines(block.body)
            return

        raise SparkError(f"Unknown block: {block.keyword}")

    def _run_statement(self, line):
        if line.startswith("say "):
            print(self._eval(line[4:]))
            return

        if line.startswith("let "):
            name, value = self._split_assignment(line[4:])
            self.variables[name] = self._eval(value)
            return

        if "=" in line:
            name, value = self._split_assignment(line)
            if name not in self.variables:
                raise SparkError(f"Unknown variable: {name}")
            self.variables[name] = self._eval(value)
            return

        raise SparkError(f"I do not understand: {line}")

    def _split_assignment(self, text):
        name, separator, value = text.partition("=")
        if not separator:
            raise SparkError(f"Assignment needs '=': {text}")

        name = name.strip()
        if not name.isidentifier():
            raise SparkError(f"Invalid variable name: {name}")

        return name, value.strip()

    def _eval(self, expression):
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError as error:
            raise SparkError(f"Bad expression: {expression}") from error
        return self._eval_node(node)

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str, bool)):
                return node.value
            raise SparkError(f"Unsupported value: {node.value!r}")

        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            raise SparkError(f"Unknown variable: {node.id}")

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)
            if operator_type not in OPERATORS:
                raise SparkError("Unsupported math operator.")
            return OPERATORS[operator_type](self._eval_node(node.left), self._eval_node(node.right))

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)
            if operator_type not in OPERATORS:
                raise SparkError("Unsupported unary operator.")
            return OPERATORS[operator_type](self._eval_node(node.operand))

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for operator_node, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                operator_type = type(operator_node)
                if operator_type not in OPERATORS:
                    raise SparkError("Unsupported comparison operator.")
                if not OPERATORS[operator_type](left, right):
                    return False
                left = right
            return True

        raise SparkError("Unsupported expression.")


def main():
    parser = argparse.ArgumentParser(description="Run a Spark language program.")
    parser.add_argument("file", help="Path to a .spark file")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as spark_file:
        source = spark_file.read()

    try:
        Spark().run(source)
    except SparkError as error:
        raise SystemExit(f"Spark error: {error}") from error


if __name__ == "__main__":
    main()
