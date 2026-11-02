"""Systems we evaluate. One entry = one row source in results/{name}.jsonl.

kind="chat":    Fireworks serverless chat completions (OpenAI-compatible).
kind="sampler": Fireworks serverless-training sampler (base model in the pool, or a saved
                sampler checkpoint via model_path). Used for the student, base and tuned, so
                both are scored through the same path.

Prices are USD per 1M tokens, checked 2026-11-02:
  chat:    https://fireworks.ai/models/fireworks/<id> (input / output)
  sampler: https://docs.fireworks.ai/fine-tuning/training-api/serverless (prefill / sample)
reasoning_effort "none" disables thinking (docs: api-reference/post-chatcompletions).
"""

FW = "accounts/fireworks/models/"

SYSTEMS = {
    # student, base. Reasoning off via renderer (Qwen3.8 thinking ate the whole 4096 budget).
    "qwen3p8-27b-base": dict(
        kind="sampler", base_model=FW + "qwen3p8-27b", tokenizer="Qwen/Qwen3.8-27B",
        renderer="qwen3_8_disable_thinking_interleaved", price_in=1.86, price_out=5.595,
    ),
    # teacher / large-model candidates
    "kimi-k3": dict(kind="chat", model=FW + "kimi-k3", reasoning_effort="none", price_in=3.00, price_out=15.00),
    # GLM-5.3 is thinking-only (HTTP 400 on "none"); "low" is its minimum
    "glm-5p3": dict(kind="chat", model=FW + "glm-5p3", reasoning_effort="low", price_in=1.40, price_out=4.40),
    "deepseek-v4p1-flash": dict(kind="chat", model=FW + "deepseek-v4p1-flash", reasoning_effort="none", price_in=0.30, price_out=1.20),
    # gpt-oss rejects reasoning_effort="none" (HTTP 400 "Invalid reasoning effort: none"); "low" is its minimum
    "gpt-oss-120b": dict(kind="chat", model=FW + "gpt-oss-120b", reasoning_effort="low", price_in=0.15, price_out=0.60),
}
