"""Opt-in offline formula scoring. No Production imports or Gold-based proof.

The frozen v1 evaluator remains unchanged. Only evidence-backed arithmetic can
override a claim, and explicit incorrect equations override direct value hits.
"""
from __future__ import annotations

import ast
import re
import unicodedata
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any, Mapping, Sequence

from . import evaluator as legacy

EVALUATION_RULES_VERSION = "evidence-aware-formulas-v2"
FORMULA_CONSTANT = "FORMULA_CONSTANT"
NUMBER = r"(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
UNIT = r"(?:个百分点|万亿元|亿元|万元|倍|天|年|月)"
EQUATION = re.compile(
    rf"(?P<expression>(?:[\d., \t()+\-*/%]|{UNIT})+)\s*=\s*"
    rf"(?P<result>[+-]?{NUMBER})(?P<unit>{UNIT}|%)?"
)
UNITS = re.compile(UNIT)


def _text(value: str) -> str:
    # Length-preserving substitutions after NFKC keep claim positions aligned.
    return (unicodedata.normalize("NFKC", value).replace("×", "*")
            .replace("÷", "/").replace("−", "-").replace("－", "-")
            .replace("**", "  "))


def _solve(expression: str):
    """Return Decimal result and leaf spans. AST never executes Python code."""
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 100:
        raise ValueError("expression_too_large")
    leaves = []

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            raw = ast.get_source_segment(expression, node)
            value = Decimal(raw)
            leaves.append((value, node.col_offset, node.end_col_offset))
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                v, start, end = leaves[-1]
                leaves[-1] = (-v, node.col_offset, end)
                return -value
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if b == 0: raise ValueError("division_by_zero")
            return a / b
        raise ValueError("unsupported_expression")

    with localcontext() as context:
        context.prec = 50
        result = walk(tree)
    return result, leaves, any(isinstance(n, ast.Div) for n in ast.walk(tree))


def inspect_formulas(answer: str, evidence: Sequence[legacy.EvidenceRecord]):
    text = _text(answer)
    available = [(legacy.canonical_decimal(c.normalized_value), c.unit, r.content)
                 for r in evidence for c in legacy.extract_numeric_claims(r.content)]
    traces = []
    for match in EQUATION.finditer(text):
        expression = match.group("expression").strip()
        # Plain assignments such as "收入=100" are not equations.
        if not re.search(r"\d\s*(?:年|月|天|倍|亿元|万元)?\s*[+\-*/]|\)\s*[+\-*/]", expression):
            continue
        start = match.start("expression") + len(match.group("expression")) - len(match.group("expression").lstrip())
        trace = {"expression": expression, "result": match.group("result"),
                 "result_unit": match.group("unit") or "", "span": list(match.span()),
                 "result_span": list(match.span("result")), "valid": False,
                 "operands": [], "reason": "unsupported_expression"}
        traces.append(trace)
        # Parentheses can wrap the entire equation, rather than only its LHS:
        # "(5年-3年=2年)". Strip such an outer wrapper only when the matching
        # closing delimiter is actually present after the stated RHS.
        stack = []
        for i, character in enumerate(expression):
            if character == '(': stack.append(i)
            elif character == ')' and stack: stack.pop()
        if len(stack) == 1 and re.match(r"\s*\)", text[match.end():]):
            prefix = expression[:stack[0]].strip()
            if not prefix or re.fullmatch(rf"[+-]?{NUMBER}\s*{UNIT}", prefix):
                offset = stack[0]+1
                expression = expression[offset:]
                start += offset
                trace['expression'] = expression
        conversion = re.search(r"\*\s*(100)\s*%\s*$", expression)
        if "%" in expression and (not conversion or expression.count("%") != 1 or trace["result_unit"] != "%"):
            continue
        if len(expression) > 2048: continue
        if any(not re.fullmatch(NUMBER, m.group()) for m in re.finditer(r"\d[\d,.]*", expression)):
            trace['reason'] = 'invalid_number_format'
            continue
        arithmetic = UNITS.sub(lambda m: " " * len(m.group()), expression)
        # Remove commas with spaces to retain offsets, then compact grouped
        # numeric literals by replacing only their commas before AST parsing.
        comma_positions = [i for i, ch in enumerate(arithmetic) if ch == ',']
        arithmetic = arithmetic.replace(',', '')
        if conversion: arithmetic = arithmetic.replace('%', ' ')
        try:
            result, leaves, has_division = _solve(arithmetic)
        except (ValueError, SyntaxError, ArithmeticError, RecursionError):
            continue
        if trace['result_unit'] == '%' and has_division and not conversion:
            trace['reason'] = 'percentage_scale_missing'
            continue
        # Restore original positions after removing thousands separators.
        def original_offset(pos):
            for comma in comma_positions:
                if comma <= pos: pos += 1
            return pos
        explicit_units = []
        for value, a, b in leaves:
            a, b = original_offset(a), original_offset(b)
            unit_match = re.match(UNIT, expression[b:])
            unit = unit_match.group() if unit_match else ""
            if unit: explicit_units.append(unit)
            constant = bool(conversion and a == conversion.start(1) and value == 100)
            found = constant or any(v is not None and legacy.numeric_equal(value, v)
                                    and (not unit or unit == u or unit in content)
                                    for v, u, content in available)
            trace["operands"].append({"value": legacy.decimal_text(value),
                                      "span": [start+a, start+b], "verified": found,
                                      "conversion_constant": constant})
        if explicit_units and (len(set(explicit_units)) > 1 or
                (not conversion and trace['result_unit'] and trace['result_unit'] != explicit_units[0])):
            trace['reason'] = 'unit_mismatch'
            continue
        stated = legacy.canonical_decimal(trace["result"])
        correct = stated is not None and legacy.numeric_equal(result, stated)
        # Division outputs may be rounded to the precision actually displayed.
        if not correct and stated is not None and has_division:
            places = max(0, -stated.as_tuple().exponent)
            with localcontext() as context:
                context.prec = 50
                correct = result.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP) == stated
        trace.update(computed=legacy.decimal_text(result), valid=correct and all(o["verified"] for o in trace["operands"]),
                     reason="verified" if correct and all(o["verified"] for o in trace["operands"]) else "wrong_result" if not correct else "unverified_operand")
    return traces


def evaluate_numeric_claims(answer: str, *, question: str = "",
                            evidence: Sequence[legacy.EvidenceRecord] = (),
                            citations: Sequence[Mapping[str, Any]] = (),
                            derived_calculations: Sequence[Mapping[str, Any]] = ()) -> legacy.NumericEvaluation:
    evaluation = legacy.evaluate_numeric_claims(answer, question=question, evidence=evidence,
                                               citations=citations, derived_calculations=derived_calculations)
    traces = inspect_formulas(answer, evidence)
    normalized = unicodedata.normalize("NFKC", answer)
    refs = [m.span() for m in legacy._INLINE_REF_RE.finditer(normalized)]
    matches = [m for m in legacy._NUMBER_RE.finditer(normalized)
               if not any(a <= m.start() and m.end() <= b for a, b in refs)]
    if len(matches) != len(evaluation.claims): raise ValueError("claim_alignment_failed")
    proven_results = {}
    rejected_results = set()
    for trace in traces:
        value = legacy.canonical_decimal(trace['result'])
        key = (value, trace['result_unit'])
        if trace['valid']: proven_results[key] = trace['computed']
        else: rejected_results.add(key)
    for claim, match in zip(evaluation.claims, matches):
        for trace in traces:
            if trace['result_span'][0] <= match.start() < trace['result_span'][1]:
                claim.comparison_result = trace['valid']
                claim.matched_by = 'formula_result' if trace['valid'] else 'formula_' + trace['reason']
                claim.evidence_value = trace.get('computed', '')
            for operand in trace['operands']:
                a, b = operand['span']
                # A binary '-' preceding an operand is an operator, not its sign.
                if a <= match.end()-1 < b:
                    claim.normalized_value = operand['value']
                    claim.comparison_result = operand['verified']
                    claim.evidence_value = operand['value'] if operand['verified'] else ''
                    if operand['conversion_constant']:
                        claim.claim_type = FORMULA_CONSTANT
                        claim.matched_by = 'percent_conversion_constant'
                    else: claim.matched_by = 'formula_operand' if operand['verified'] else 'formula_unverified_operand'
        key = (legacy.canonical_decimal(claim.normalized_value), claim.unit)
        # Repeated conclusion values must be backed by the same unit and a
        # verified equation. A conflicting equation never authorizes repeats.
        if key in proven_results and key not in rejected_results and not claim.matched_by.startswith('formula_wrong'):
            claim.comparison_result = True
            claim.matched_by = 'formula_result'
            claim.evidence_value = proven_results[key]
    evaluation.failed_claims = [c for c in evaluation.claims if not c.comparison_result]
    evaluation.passed = not evaluation.failed_claims
    evaluation.answer_fact_count = sum(c.claim_type == legacy.ANSWER_FACT for c in evaluation.claims)
    return evaluation
