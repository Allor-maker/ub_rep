"""Independent blocking checks for a deliberately small verifiable LLM subset."""

import ast
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

from nsu_budget_agent.contracts.models import (
    AttemptResult,
    CheckResult,
    TaskSpec,
    ToolContext,
    ValidationResult,
)
from nsu_budget_agent.contracts.ports import ToolExecutor
from nsu_budget_agent.tools.calculations import number, render

from .local import LocalValidator


@dataclass(frozen=True)
class VerifiableQuestion:
    left: str
    right: str | None = None


def question(task: TaskSpec) -> VerifiableQuestion | None:
    """Full instruction consumption: never ignore modifiers or extra requests."""
    if (
        task.type_hint == "coding"
        or task.check_hint not in (None, "exact_int", "exact_str")
        or len(task.instruction) > 2048
    ):
        return None
    text = task.instruction.strip()
    expression = re.fullmatch(
        r"What is the exact value of ([0-9+*/%(). \-]+)\? Return only the result\.", text
    )
    equation = re.fullmatch(
        r"Solve for x: ([0-9x+*/(). \-]+)=([0-9x+*/(). \-]+)\. Return only the value of x\.",
        text,
    )
    parsed = (
        VerifiableQuestion(expression[1])
        if expression
        else (VerifiableQuestion(equation[1], equation[2]) if equation else None)
    )
    if parsed is not None:
        try:
            # Also reject nonlinear, undefined and resource-intensive expressions
            # before spending money. No expected answer is passed to the Solver.
            expected = _expected(parsed)
            if (
                task.check_hint == "exact_int"
                and re.fullmatch(r"-?(?:0|[1-9]\d*)", expected) is None
            ):
                return None
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
            return None
    return parsed


def _affine(expression: str) -> tuple[Fraction, Fraction]:
    """Return exact (coefficient of x, constant); no eval, imports or task code."""
    if len(expression) > 512:
        raise ValueError("expression limit")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("node limit")

    def visit(node: ast.AST, depth: int = 0) -> tuple[Fraction, Fraction]:
        if depth > 12:
            raise ValueError("depth limit")
        if isinstance(node, ast.Name) and node.id == "x":
            return Fraction(1), Fraction(0)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(0), number(ast.get_source_segment(expression.strip(), node) or "")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a, b = visit(node.operand, depth + 1)
            return (-a, -b) if isinstance(node.op, ast.USub) else (a, b)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left, depth + 1)
            c, d = visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                result = a + c, b + d
            elif isinstance(node.op, ast.Sub):
                result = a - c, b - d
            elif isinstance(node.op, ast.Mult) and not (a and c):
                result = a * d + b * c, b * d
            elif isinstance(node.op, ast.Div) and not c and d:
                result = a / d, b / d
            else:
                raise ValueError("not a supported affine expression")
            if any(max(v.numerator.bit_length(), v.denominator.bit_length()) > 512 for v in result):
                raise ValueError("arithmetic size limit")
            return result
        raise ValueError("unsupported expression")

    return visit(tree.body)


def _expected(spec: VerifiableQuestion) -> str:
    a, b = _affine(spec.left)
    if spec.right is None:
        if a:
            raise ValueError("not a constant expression")
        return render(b)
    c, d = _affine(spec.right)
    if a == c:
        raise ValueError("no unique solution")
    return render((d - b) / (a - c))


class FallbackValidator:
    """Keep A5/B integrity checks; replace only unsupported semantic verification."""

    def __init__(self, tools: ToolExecutor) -> None:
        self.local = LocalValidator(tools)

    async def validate(
        self, task: TaskSpec, result: AttemptResult, ctx: ToolContext
    ) -> ValidationResult:
        baseline = await self.local.validate(task, result, ctx)
        if result.solver_id != "generic":
            return baseline
        spec = question(task)
        correct = False
        if spec is not None and result.candidate is not None:
            correct = not result.candidate.evidence_paths and result.candidate.text == _expected(
                spec
            )
        checks = tuple(
            check
            for check in baseline.checks
            if check.check_id not in ("supported_task", "semantics")
        ) + (
            CheckResult(
                check_id="llm_semantics",
                status="passed" if correct else "failed",
                blocking=True,
                message="independent exact arithmetic and unique-solution check",
            ),
        )
        status: Literal["passed", "failed", "inconclusive"] = (
            "failed"
            if any(c.status == "failed" for c in checks)
            else "inconclusive"
            if any(c.status == "inconclusive" for c in checks)
            else "passed"
        )
        return replace(baseline, checks=checks, status=status)
