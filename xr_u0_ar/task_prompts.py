"""Shared task prompt templates used by AR and FlashAR inference."""

T2I_PROMPT_TEMPLATE = (
    "<|extra_203|>You are a helpful assistant for t2i task. "
    "USER: {text} ASSISTANT: <|extra_100|>"
)

# CFG should remove only the user text while preserving the T2I task context.
T2I_UNCOND_PROMPT = T2I_PROMPT_TEMPLATE.format(text="")

X2I_PROMPT_TEMPLATE = (
    "<|extra_203|>You are a helpful assistant. "
    "USER: <|IMAGE|>{question} ASSISTANT: <|extra_100|>"
)

# CFG should remove only the user instruction while preserving the reference image.
X2I_UNCOND_PROMPT = X2I_PROMPT_TEMPLATE.format(question="")
