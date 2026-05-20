"""
System prompts and evaluation prompt builders for the Chinese rewriting
quality evaluator.

EMNLP 2026 - LoRA Fine-tuned Evaluator for Chinese Text Rewriting
"""

# Valid values for ``prompt_variant`` in score_only mode (train + eval).
SCORE_ONLY_PROMPT_VARIANTS = ("original", "simple", "detail")

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_SCORE_ONLY = """你是一个专业的中文文本改写质量评估专家。请根据以下维度对中文文本改写质量进行评分（0-5分）：

评分维度：
1. 语义一致性：改写是否保留了原文的核心语义
2. 句式重构：改写是否进行了句法结构的改变
3. 词汇变化：改写是否使用了不同的词汇表达
4. 风格保持：改写是否保持了原文的风格和长度

评分标准：
- 0分：改写完全失败（严重语义扭曲或毫无改写）
- 1分：改写质量很差
- 2分：改写质量较差
- 3分：改写质量一般
- 4分：改写质量较好
- 5分：改写质量优秀

请先简要分析，然后给出最终综合评分（0-5分的整数）。"""

SYSTEM_PROMPT_SCORE_SIMPLE = (
    "请帮我评判输入文本和输出文本重写的质量，并给出最终综合评分（0-5分的整数）。"
)

SYSTEM_PROMPT_SCORE_DETAIL = (
    "请帮我判断输入文本与输出文本是否满足以下要求:\n"
    "要求 1. 在保持原文含义不变的前提下，用原文不同的文字进行表达，不额外新增、删减会严重影响语义的内容。\n"
    "要求 2. 在保证逻辑正确、语序通顺的前提下，变换实体、短语、概念出现的前后顺序、结构关系；顺序变换幅度越小，分数越低。\n"
    "要求 3. 在保证特殊实体、引用、解释、特定语句不变的前提下，使用同义词替换；同义词替换越少，分数越低；\n"
    "要求 4. 保持与原文语言风格一致，字数长度变化合理；语言风格差距越大，分数越低。\n"
    "要求 5. 改写综合评分要非常严格，同时考虑上述 4 个要求；在保证语义一致（要求 1）的前提下，（要求 2,3）变化越大越好，字数越接近越好（要求 4）；如果存在大量长字段完全复制，则给 0 分。\n"
    "评分准则，每个要求最低 0 分，满分均为 5 分，评分要尽可能严格。"
)

SYSTEM_PROMPT_MULTI_SCORE = """你是一个专业的中文文本改写质量评估专家。请根据以下5个维度对中文文本改写质量进行评分（0-5分）：

1. 语义一致性（要求1）：改写是否保留了原文的核心语义，没有添加、删除或扭曲重要信息。
2. 句式重构（要求2）：改写是否对原文进行了足够的句法结构改变，而非简单替换。
3. 词汇变化（要求3）：改写是否使用了不同的词汇和表达方式。
4. 风格保持（要求4）：改写是否保持了原文的风格特征和合理长度。
5. 综合评分（要求5）：综合以上维度的总体评价。

评分标准：0-5分整数。

请以JSON格式返回：[{{"要求1": "理由", "score": X}}, ..., {{"要求5": "综合理由", "score": Y}}]"""


def get_score_only_system_prompt(prompt_variant: str = "original") -> str:
    """Return the system prompt for score_only training/eval by variant."""
    if prompt_variant == "original":
        return SYSTEM_PROMPT_SCORE_ONLY
    if prompt_variant == "simple":
        return SYSTEM_PROMPT_SCORE_SIMPLE
    if prompt_variant == "detail":
        return SYSTEM_PROMPT_SCORE_DETAIL
    raise ValueError(
        f"Unknown prompt_variant: {prompt_variant!r}. "
        f"Expected one of {SCORE_ONLY_PROMPT_VARIANTS}."
    )


# =============================================================================
# User Prompt Builders
# =============================================================================


def build_score_only_user_prompt(source_text: str, rewrite_text: str) -> str:
    """Build user prompt for score-only evaluation mode.

    Args:
        source_text: The original Chinese text.
        rewrite_text: The rewritten Chinese text.

    Returns:
        Formatted user prompt string.
    """
    return (
        f"原文：\n{source_text}\n\n"
        f"改写：\n{rewrite_text}\n\n"
        f"请对该改写进行综合评分（0-5分）。"
    )


def build_multi_score_user_prompt(source_text: str, rewrite_text: str) -> str:
    """Build user prompt for multi-dimension evaluation mode.

    Args:
        source_text: The original Chinese text.
        rewrite_text: The rewritten Chinese text.

    Returns:
        Formatted user prompt string.
    """
    return (
        f"原文：\n{source_text}\n\n"
        f"改写：\n{rewrite_text}\n\n"
        f"请按照5个维度（语义一致性、句式重构、词汇变化、风格保持、综合评分）评分（0-5分）。"
    )


# =============================================================================
# Evaluation Prompt Builder (for inference / eval_evaluator.py)
# =============================================================================


def build_eval_messages(
    source_text: str,
    rewrite_text: str,
    mode: str = "score_only",
    prompt_variant: str = "original",
) -> list[dict]:
    """Build full message list for evaluation inference.

    Constructs the chat-format messages that the model expects:
    [system_message, user_message].

    The assistant response will be generated at inference time.

    Args:
        source_text: The original Chinese text.
        rewrite_text: The rewritten Chinese text.
        mode: Either "score_only" or "multi_score".
        prompt_variant: For ``score_only`` only: ``original`` (维度说明版),
            ``simple``, or ``detail`` (与仓库内对应训练 JSON 的 system 一致).
            Ignored when ``mode`` is ``multi_score``.

    Returns:
        List of message dicts with "role" and "content" keys.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode == "score_only":
        system_prompt = get_score_only_system_prompt(prompt_variant)
        user_prompt = build_score_only_user_prompt(source_text, rewrite_text)
    elif mode == "multi_score":
        system_prompt = SYSTEM_PROMPT_MULTI_SCORE
        user_prompt = build_multi_score_user_prompt(source_text, rewrite_text)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'score_only' or 'multi_score'.")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =============================================================================
# Score Parsing
# =============================================================================


def parse_score_from_response(response_text: str, mode: str = "score_only") -> int | None:
    """Parse the predicted score from model output.

    For score_only mode: finds an integer 0-5 in the response.
    For multi_score mode: extracts the score from 要求5 in the JSON array.

    Args:
        response_text: The raw text generated by the model.
        mode: Either "score_only" or "multi_score".

    Returns:
        Parsed integer score (0-5), or None if parsing fails.
    """
    import re
    import json

    if mode == "score_only":
        # Look for pattern like "评分为X分" or just a standalone digit 0-5
        # Strategy 1: explicit score mention
        match = re.search(r"评分为\s*([0-5])\s*分", response_text)
        if match:
            return int(match.group(1))

        # Strategy 2: find standalone integer 0-5, preferring the last one
        # (model may analyze first then conclude)
        numbers = re.findall(r"(?<![0-9.])([0-5])(?![0-9.])", response_text)
        if numbers:
            return int(numbers[-1])

        return None

    elif mode == "multi_score":
        # Try to parse as JSON array and extract 要求5 score
        try:
            # Find JSON array in response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
                for item in items:
                    if "要求5" in item:
                        score = item.get("score")
                        if isinstance(score, (int, float)):
                            return int(score)
                # Fallback: return last item's score
                if items:
                    score = items[-1].get("score")
                    if isinstance(score, (int, float)):
                        return int(score)
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

        # Fallback: same as score_only
        numbers = re.findall(r"(?<![0-9.])([0-5])(?![0-9.])", response_text)
        if numbers:
            return int(numbers[-1])

        return None

    else:
        raise ValueError(f"Unknown mode: {mode}")
