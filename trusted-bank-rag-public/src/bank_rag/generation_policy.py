from __future__ import annotations

from collections.abc import Callable
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationReply:
    text: str = ""
    finish_reason: str | None = None


def sdk_length_reply(error):
    """Recover ONLY a proven SDK length response; other exceptions propagate."""
    if type(error).__name__ != 'LengthFinishReasonError':
        return None
    completion = getattr(error, 'completion', None)
    choices = getattr(completion, 'choices', None)
    if not choices or choices[0].finish_reason != 'length':
        return None
    usage = getattr(completion, 'usage', None)
    usage = usage.model_dump() if hasattr(usage, 'model_dump') else usage if isinstance(usage, dict) else {}
    return GenerationReply(getattr(choices[0].message, 'content', None) or '', 'length'), usage


def generate_with_one_incomplete_retry(generate, initial_prompt, empty_retry_prompt, *, truncation_generate=None):
    """Empty and token-limited responses share one retry; never concatenate."""
    first = generate(initial_prompt)
    reason = "length" if first.finish_reason == "length" else "empty" if not first.text.strip() else None
    if reason is None:
        return first, None
    if reason == "length" and truncation_generate is not None:
        # Keep facts/citations, but change delivery strategy. Budget and retry
        # count stay bounded; the caller explicitly disables hidden reasoning.
        return truncation_generate(empty_retry_prompt), reason
    retry_prompt = empty_retry_prompt if reason == "empty" else initial_prompt + (
        "\n\n上次生成因输出额度耗尽而未完成。请重新给出完整、简洁的最终答案，"
        "不要续写残片，不要输出思考过程或前言；保留全部必要条件和引用，"
        "无需罗列与问题无关的背景事实。"
    )
    return generate(retry_prompt), reason


def clarification_advisory(plan: dict, judgment: dict | None) -> str:
    """Carry an existing semantic warning to generation, never change admission."""
    if not judgment or judgment.get("action") != "needs_clarification":
        return ""
    context = {
        "judgment_source": judgment.get("judgment_source", "existing_jev"),
        "action": "needs_clarification",
        "resolved_conditions": plan.get("resolved_conditions") or {},
        "metric_binding": plan.get("metric_binding") or {},
        "clarification_needed": plan.get("clarification_needed") or [],
    }
    return "\n\n【回答前的口径不确定性提醒】\n" + json.dumps(context, ensure_ascii=False) + "\n" + (
        "以上是已有证据判断模块的建议，不是新的事实、用户要求或强制拒答结论。"
        "请结合原问题、锁定条件与证据，核查所问指标、机构范围、期间及统计口径是否唯一确定。"
        "locked=true的已确定条件不得重新推断；空值、unverified_retrieval_hint及检索到的某一指标或机构不代表用户已选择该口径。"
        "如果原问题或证据已足以唯一确定必要条件，可以依据证据直接回答。"
        "如果仍存在会改变答案的多个合理解释，应先指出具体缺少哪个条件并请求澄清；当前不支持追问时可明确说明依据不足。"
        "不能自行选择其中一个解释并以“若按某口径”给出数值来代替澄清，也不能把某个默认机构档次当作用户已指定。"
        "不要向用户提及Jev、Gate或内部模块。"
    )


def generate_with_one_empty_retry(
    generate: Callable[[str], str | None],
    initial_prompt: str,
    retry_prompt: str,
) -> tuple[str, bool]:
    """Generate once and retry exactly once only when the first body is empty."""
    answer = (generate(initial_prompt) or "").strip()
    if answer:
        return answer, False
    answer = (generate(retry_prompt) or "").strip()
    return answer, True
