"""Smoke-test vLLM offline inference with Qwen2.5-Coder-7B."""
import os
import time

os.environ.setdefault("HF_HOME", "/mnt/localssd/cdbench/hf_cache")

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")

print(f"Loading {MODEL} on 2 GPUs (TP=2)...")
t0 = time.time()
llm = LLM(
    model=MODEL,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    dtype="bfloat16",
    download_dir=os.environ["HF_HOME"],
)
print(f"Loaded in {time.time()-t0:.1f}s")

prompts = [
    "Fix the bug in this Python: `def add(a, b): return a - b`. Return only fixed code.",
    "What is the time complexity of inserting into a balanced BST? One sentence.",
]
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
    top_p=1.0,
)

# Use chat template
tokenizer = llm.get_tokenizer()
chat_prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
    )
    for p in prompts
]

t0 = time.time()
outputs = llm.generate(chat_prompts, sampling_params)
dt = time.time() - t0

for i, out in enumerate(outputs):
    print(f"\n=== Prompt {i+1} (len={len(chat_prompts[i])}) ===")
    print(prompts[i])
    print("--- Generated ---")
    print(out.outputs[0].text.strip()[:500])

n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"\nGenerated {n_tokens} tokens in {dt:.2f}s ({n_tokens/dt:.1f} tok/s)")
