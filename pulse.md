---
title: "Pulse 2 — A 3B On-Device Wellness Coach"
description: "Building a small, private, fully on-device wellness coaching LLM from Qwen2.5-3B — fine-tuning, quantization, Core ML conversion, and the road to a real iOS/macOS app."
author: "Abiral Dahal"
date: "2026-06-01"
tags: ["LLM", "Fine-Tuning", "Core ML", "GGUF", "Quantization", "On-Device AI", "Edge AI", "Wellness", "Qwen2.5"]
---

# Pulse 2 — A 3B On-Device Wellness Coach

> A small, opinionated, privacy-first wellness coaching language model built to live entirely inside your phone.

**Repo:** [huggingface.co/Abiral129/Pulse3b](https://huggingface.co/Abiral129/Pulse3b)
**License:** Apache 2.0
**Base model:** [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
**Parameters:** 3.1B
**Context:** 32,768 tokens
**Released:** June 2026

---

## TL;DR

Pulse 2 is a 3B-parameter conversational wellness coach fine-tuned from Qwen2.5-3B. It is designed to help users with sleep, stress, fitness, nutrition, and mental wellbeing in a warm, motivating, science-backed tone.

I shipped it in three formats so it runs anywhere:

| Format | File | Size | Target |
|---|---|---|---|
| BF16 safetensors | `final/` | 5.7 GB | PyTorch / `transformers`, fine-tuning |
| Q4_K_M GGUF | `gguf/pulse-q4_k_m.gguf` | 1.8 GB | `llama.cpp`, Ollama, LM Studio (CPU/GPU) |
| Core ML INT4 | `coreml/pulse.mlpackage` | 1.6 GB | iOS / macOS on-device (Apple Silicon) |

The end goal was always Core ML. Pulse 2 is the foundation for a small on-device wellness app I'm building next — no API keys, no server bills, no user data leaving the phone.

---

## Why a 3B model?

Bigger isn't always better when the model has to live inside a phone. Practical constraints:

- **RAM budget on iPhone/iPad** is roughly 2–4 GB available to apps. A 7B model in INT4 won't fit comfortably.
- **Battery / thermals** — Apple Neural Engine and CPU inference both cost watts. A 3B model is the sweet spot for sustained chat sessions without melting the phone.
- **Cold start** — model load time matters in an app. 1.6 GB loads in seconds; 4 GB feels broken.
- **Quality floor** — modern small models (Qwen2.5-3B, Llama 3.2-3B, Phi-3) are genuinely usable for narrow domains like coaching. They lose to GPT-4 on reasoning, but for "give me a 4-step plan to fix my sleep," they hold up.

For wellness coaching specifically — empathy, structured advice, conversational tone — a well-fine-tuned 3B is enough.

---

## Architecture

Pulse 2 inherits directly from Qwen2.5-3B:

| | |
|---|---|
| Architecture | Qwen2ForCausalLM |
| Hidden size | 2048 |
| Layers | 36 |
| Attention heads | 16 |
| KV heads (GQA) | 2 |
| Intermediate size | 11,008 |
| Activation | SiLU |
| Vocab size | 151,936 |
| Max position | 32,768 |
| RoPE θ | 1,000,000 |
| dtype | bfloat16 |

Grouped-query attention with 2 KV heads makes the KV cache small — which matters a lot for on-device inference where memory is the binding constraint.

---

## Training

- **Base:** Qwen2.5-3B (Apache 2.0)
- **Method:** Supervised fine-tuning on a curated wellness coaching dataset (sleep hygiene, stress-management protocols, exercise programming, nutrition basics, conversational empathy patterns).
- **Format:** ChatML (`<|im_start|>` / `<|im_end|>` tokens).
- **System persona:** The model is prompted as "Pulse, a personal wellness AI coach" — warm, motivating, empathetic, science-backed, concise. It is explicitly told not to say "as an AI" and not to act like a medical professional.

The dataset emphasizes:

1. Empathetic acknowledgment of the user's state before jumping to advice.
2. Concrete, actionable steps (numbered plans) over abstract suggestions.
3. Honest deflection on medical questions — Pulse 2 is a coach, not a clinician.
4. A friendly, peer-to-peer tone — like a knowledgeable friend, not a chatbot.

---

## The three formats — and why each one matters

### 1. BF16 safetensors (`final/`)

The "source of truth." 5.7 GB of `model.safetensors` plus tokenizer and chat template. Loadable directly with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Abiral129/Pulse3b", subfolder="final")
model = AutoModelForCausalLM.from_pretrained(
    "Abiral129/Pulse3b",
    subfolder="final",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

This is the format you want if you plan to further fine-tune, distill, or convert to another runtime.

### 2. Q4_K_M GGUF (`gguf/pulse-q4_k_m.gguf`)

K-quant 4-bit. The pragmatic everyday format — runs on any laptop CPU through `llama.cpp`, Ollama, or LM Studio.

Quality loss vs BF16 is minimal for coaching tasks (subjectively indistinguishable on short prompts). Size drops from 5.7 GB to 1.8 GB.

```bash
ollama create pulse -f Modelfile
ollama run pulse "I've been sleeping 5 hours a night. Help."
```

This is what powers the public demo Space and what most external users will download and try.

### 3. Core ML INT4 (`coreml/pulse.mlpackage`)

Apple's native model format. Quantized to INT4 weights, packaged as `.mlpackage` (a directory with `Manifest.json`, the `.mlmodel` graph, and a `weight.bin` blob).

Runs on:
- Apple Neural Engine (ANE) where possible
- GPU (Metal Performance Shaders) as fallback
- CPU as last resort

This was the whole point of the project. Native Core ML lets the model:

- Load with the standard `MLModel` API — no Python runtime, no bridging hacks.
- Run with very low latency on M1/M2/M3 Macs and A-series iPhones/iPads.
- Take advantage of ANE for matrix multiplies, which is dramatically more power-efficient than CPU.
- Coexist with the rest of an iOS app's memory budget.

The trade-off is that the export pipeline (PyTorch → traced graph → Core ML MIL → INT4 quantization) is fragile. Several PRs/patches into `coremltools` were needed to keep numpy 2.x and the Qwen2 architecture happy together.

---

## The "everything is broken" diary

Most of the actual time was not "training the model." It was:

- **Torchvision circular import** showing up at unexpected times when `transformers` lazily imports it inside the converter.
- **`coremltools` cast op** failing on size-1 `ndarray` inputs (numpy 2 changed `int()` semantics on length-1 arrays). Had to monkey-patch `_cast` in the converter.
- **`jit.trace` complaints** about scaled dot-product attention — the standard SDPA path doesn't trace cleanly, so I had to force `attn_implementation="eager"` and disable KV cache during export.
- **8 GB RAM cap** while tracing — the script sets `RLIMIT_RSS` to 8 GB to fail fast on machines that can't hold the model.
- **Sequence length tradeoff** — exporting at full 32K context blows up. I traced at `seq_len=128` and rely on Core ML's static-shape inference for the real app to handle short conversational chunks.

None of this is in a paper. It's just the actual cost of moving a research-format model to a production-format model. Anyone shipping on-device LLMs has the same diary.

---

## Quick start

### Ollama (easiest)

```bash
huggingface-cli download Abiral129/Pulse3b gguf/pulse-q4_k_m.gguf --local-dir .

cat > Modelfile <<'EOF'
FROM ./gguf/pulse-q4_k_m.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
SYSTEM """You are Pulse, a personal wellness coach. Be warm, science-backed, and concise."""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
EOF

ollama create pulse -f Modelfile
ollama run pulse "I've been sleeping 5 hours for a week. What do I do?"
```

### llama.cpp

```bash
./llama-cli -m gguf/pulse-q4_k_m.gguf \
  -p "You are Pulse, a wellness coach." \
  -cnv --temp 0.7 --top-p 0.9 --repeat-penalty 1.1 -c 2048
```

### Transformers (BF16)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Abiral129/Pulse3b", subfolder="final")
model = AutoModelForCausalLM.from_pretrained(
    "Abiral129/Pulse3b", subfolder="final",
    torch_dtype=torch.bfloat16, device_map="auto",
)

messages = [
    {"role": "system", "content": "You are Pulse, a personal wellness coach."},
    {"role": "user", "content": "My resting heart rate jumped from 62 to 88. What's going on?"},
]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=300, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

### Core ML (Python)

```python
import coremltools as ct
from transformers import AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("Abiral129/Pulse3b", subfolder="final")
mlmodel = ct.models.MLModel("coreml/pulse.mlpackage")
ids = tok("Hello Pulse", return_tensors="np").input_ids.astype(np.int32)
print(mlmodel.predict({"input_ids": ids}))
```

For real generation on iOS, you wrap the `.mlpackage` with a Swift generation loop on top of the logits — sampling, KV state, stop tokens, etc. That's the next post.

---

## Recommended sampling

| Param | Value |
|---|---|
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `repeat_penalty` | 1.1 |
| `num_ctx` | 2048 |
| Stop tokens | `<|im_end|>`, `<|im_start|>` |

Lower the temperature to 0.4–0.5 if you want more deterministic, plan-style outputs. Bump `repeat_penalty` to 1.15 if you see it looping.

---

## Limitations (read this before shipping anything)

- **Not a medical device.** Pulse 2 is a wellness coach. It will refuse to diagnose, but it can still confidently make wrong claims. Do not let it gate medication decisions, mental health crises, or any clinical pathway.
- **3B reasoning ceiling.** It is not GPT-4. Multi-step reasoning, math, and any task requiring tool use will underperform a frontier model.
- **English-first.** Training data leans heavily English with some Spanish. Other languages will be weaker.
- **Quantization drift.** Q4_K_M is great; INT4 Core ML is a bit more aggressive. Expect occasional small phrasing weirdness in the Core ML build relative to BF16.
- **No memory between sessions.** The model does not learn from your conversations. That's a feature for privacy and a bug for personalization — solved at the app layer, not the model layer.

---

## What's next

Pulse 2 is a building block. The roadmap I'm actually working on:

1. **iOS app on top of the Core ML build.** Native generation loop, on-device chat, daily check-ins, basic wellness tracking. No server, no account required for inference. (In progress.)
2. **Distillation experiments** to push to 1B–2B with less quality loss for older iPhones / lower-RAM devices.
3. **DPO / preference tuning** on coach-specific failure modes — chiefly "stop apologizing" and "give me numbers, not vibes."
4. **Multilingual expansion** — Spanish first, then Portuguese and French.
5. **Optional tool use** — sleep tracker, HR data, calendar — opt-in and processed locally.

---

## Try it / read more

- **Model + all 3 formats:** [huggingface.co/Abiral129/Pulse3b](https://huggingface.co/Abiral129/Pulse3b)
- **Browser demo (slow CPU, fully functional):** [huggingface.co/spaces/Abiral129/pulse-demo](https://huggingface.co/spaces/Abiral129/pulse-demo)
- **More posts:** [pulse.abiral.me](https://pulse.abiral.me)

---

## Acknowledgements

Built on top of [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) by the Qwen team at Alibaba. GGUF conversion via [`llama.cpp`](https://github.com/ggerganov/llama.cpp). Core ML conversion via [`coremltools`](https://github.com/apple/coremltools).

---

## Citation

```bibtex
@misc{pulse2_2026,
  title  = {Pulse 2: A 3B on-device wellness coaching language model},
  author = {Abiral Dahal},
  year   = {2026},
  url    = {https://huggingface.co/Abiral129/Pulse3b}
}
```
