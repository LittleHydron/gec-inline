"""System prompt and few-shot examples for the GEC inline-edit task.

The same SYSTEM_PROMPT is used during SFT training and at inference.
FEW_SHOT_EXAMPLES are added only to the base model at evaluation time
so the comparison is fair (without them the base model emits free-form
prose and the bracket parser cannot score it).
"""

SYSTEM_PROMPT = (
    "You are a grammatical error correction assistant. "
    "Given an English sentence, output the corrected sentence with every "
    "edit inlined inside braces using the syntax {wrong=>right}. "
    "Use {=>word} for insertions and {word=>} for deletions. "
    "Keep all unchanged tokens exactly as in the input. "
    "If the input is already correct, return it unchanged with no braces."
)


FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "I goes to school every day .",
        "I {goes=>go} to school every day .",
    ),
    (
        "She have been working since morning .",
        "She {have=>has} been working since morning .",
    ),
    (
        "He don't likes apples and oranges .",
        "He {don't likes=>doesn't like} apples and oranges .",
    ),
]


def build_user_message(sentence: str) -> str:
    """Wrap the raw sentence as the user turn content."""
    return f"Correct this sentence:\n{sentence}"


def build_chat_messages(sentence: str, include_few_shot: bool = False) -> list[dict]:
    """Build a chat-format message list ready for tokenizer.apply_chat_template."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if include_few_shot:
        for src, tgt in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": build_user_message(src)})
            messages.append({"role": "assistant", "content": tgt})
    messages.append({"role": "user", "content": build_user_message(sentence)})
    return messages
