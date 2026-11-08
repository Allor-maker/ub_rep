"""Independent bounded Decimal/AST math; no executor calculation imports."""

import ast
import math
import re
import time
from decimal import Context, Decimal, Inexact, localcontext


def independent_math(arguments: dict, deadline: float) -> str:
    def check(value: Decimal) -> Decimal:
        if time.monotonic() >= deadline:
            raise TimeoutError("independent math deadline")
        if not value.is_finite() or (value and abs(value.adjusted()) > 200):
            raise ValueError("independent math size")
        return value

    context = Context(prec=256, Emin=-1024, Emax=1024)
    context.traps[Inexact] = True
    with localcontext(context):
        if set(arguments) == {"expression"}:
            expression = arguments["expression"]
            if not isinstance(expression, str) or not 0 < len(expression) <= 1024:
                raise ValueError("independent expression limit")
            tree = ast.parse(expression.strip(), mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 128:
                raise ValueError("independent expression nodes")

            def visit(node: ast.AST, depth: int = 0) -> Decimal:
                check(Decimal(0))
                if depth > 16:
                    raise ValueError("independent expression depth")
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "lcm"
                    and not node.keywords
                    and 2 <= len(node.args) <= 16
                ):
                    operands = [visit(argument, depth + 1) for argument in node.args]
                    if any(
                        value != int(value) or not 1 <= value <= 1_000_000 for value in operands
                    ):
                        raise ValueError("independent LCM operands")
                    return check(Decimal(math.lcm(*(int(value) for value in operands))))
                if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                    raw = ast.get_source_segment(expression.strip(), node) or ""
                    if re.fullmatch(r"\d{1,40}(?:\.\d{1,20})?", raw) is None:
                        raise ValueError("independent number")
                    return check(Decimal(raw))
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    value = visit(node.operand, depth + 1)
                    return -value if isinstance(node.op, ast.USub) else value
                if not isinstance(node, ast.BinOp):
                    raise ValueError("independent expression operation")
                left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                elif isinstance(node.op, ast.Div):
                    value = left / right
                elif isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                    quotient = left // right
                    if left - quotient * right and left * right < 0:
                        quotient -= 1
                    value = (
                        quotient if isinstance(node.op, ast.FloorDiv) else left - quotient * right
                    )
                elif isinstance(node.op, ast.Pow) and right == int(right) and 0 <= right <= 12:
                    value = Decimal(1) if right == 0 else left ** int(right)
                else:
                    raise ValueError("independent expression operation")
                return check(value)

            value = visit(tree.body)
        elif set(arguments) == {"initial", "operations"}:
            if type(arguments["initial"]) is not int or not isinstance(
                arguments["operations"], list
            ):
                raise ValueError("independent stock arguments")
            value = Decimal(arguments["initial"])
            for operation in arguments["operations"]:
                operand = operation["value"]
                if type(operand) is not int or operand < 0:
                    raise ValueError("independent stock operand")
                if operation["op"] == "add":
                    value += operand
                elif operation["op"] == "sub":
                    value -= operand
                elif operation["op"] == "mul":
                    value *= operand
                elif operation["op"] == "floor_div":
                    value = value // operand
                else:
                    raise ValueError("independent stock operation")
                check(value)
                if value < 0:
                    raise ValueError("independent negative stock")
        else:
            raise ValueError("independent math arguments")
        rendered = format(value, "f")
        if "." in rendered and len(rendered.partition(".")[2]) > 40:
            raise ValueError("independent output precision")
        return (rendered.rstrip("0").rstrip(".") if "." in rendered else rendered) if value else "0"
