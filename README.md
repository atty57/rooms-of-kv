# rooms-of-kv

Companion code and data.

On a 50-question stratified LongMemEval_S sample, neither k-means room routing nor LLM reranking improved over a flat top-5 RAG baseline for Qwen3-4B on a free Colab T4. In this setup, retrieval quality looked adequate, while downstream answer synthesis remained the main observed weakness.

All numbers in the article come from the JSONL files in `data/`. Everything here runs on a free Colab T4.

## Headline results

| Architecture                 | Accuracy (N=50) | Overall |
|------------------------------|-----------------|---------|
| B3 Flat RAG                  | 26/50           | 52%     |
| B5 Scoped Rooms (k-means×16) | 24/50           | 48%     |
| B7 LLM Rerank (top-20→5)     | 21/50           | 42%     |

McNemar exact tests on pairwise discordant pairs: flat vs scoped p=0.625, flat vs rerank p=0.388. Neither difference reaches significance at N=50.

Judge: Gemma 3 12B via Google AI Studio, Cohen's κ = 0.73 against 9 manual labels.


## Repository contents

```
rooms-of-kv/
├── README.md                            ← you are here
├── reproduce.ipynb                      ← single-notebook reproduction of all three runs
├── requirements.txt                     ← pinned versions known to work on Colab T4 (April 2026)
├── data/
│   ├── sample_manifest.json             ← 50 question indices, seed=42, stratified
│   ├── flat_rag_n50.jsonl               ← Day 5 B3 Flat RAG results
│   ├── scoped_rooms_n50.jsonl           ← Day 6 B5 Scoped Rooms results
│   ├── rerank_n50.jsonl                 ← Day 7 B7 LLM Rerank results
│   ├── judge_validation.jsonl           ← Day 4 Gemma vs manual labels (N=10)
│   └── scorability.json                 ← Day 2 literal-match distribution by type
```

## Reproduce the numbers

### 1. Quickstart — verify the published numbers

You don't need a GPU to verify the article's claims. The JSONL files contain every raw per-question record:

```python
import json
from collections import defaultdict

def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

for name, path in [
    ("B3 Flat", "data/flat_rag_n50.jsonl"),
    ("B5 Scoped", "data/scoped_rooms_n50.jsonl"),
    ("B7 Rerank", "data/rerank_n50.jsonl"),
]:
    records = load(path)
    correct = sum(1 for r in records if r.get("judge_verdict") == 1)
    print(f"{name}: {correct}/{len(records)} = {correct/len(records)*100:.1f}%")
```

Expected output:
```
B3 Flat:   26/50 = 52.0%
B5 Scoped: 24/50 = 48.0%
B7 Rerank: 21/50 = 42.0%
```

### 2. Full reproduction on Colab T4

Open `reproduce.ipynb` in Colab, set runtime to **T4 GPU**, and run all cells. Total wall time: ~2 hours.

Requirements:
- Free Colab tier with T4 GPU
- `GOOGLE_API_KEY` secret (get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))
- `HF_TOKEN` secret (get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

The notebook will:
1. Download LongMemEval_S (~274 MB, cached to Drive)
2. Build stratified sample (seed=42, reproducible)
3. Run B3 flat RAG eval (~40 min)
4. Run B5 scoped rooms eval (~25 min)
5. Run B7 rerank eval (~45 min — most time is judge API calls)
6. Print the same three-way comparison table shown in the Medium article

### 3. Swap in your own model

The most useful thing you can do with this repo: change one line.

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",   # ← swap this
    dtype=torch.float16, device_map="auto", attn_implementation="sdpa",
)
```

Candidates worth trying that fit on T4:
- `Qwen/Qwen3-8B-Instruct-AWQ` — INT4 weights, ~5 GB on T4
- `meta-llama/Llama-3.2-3B-Instruct` — dense, comparable size to baseline
- `HuggingFaceTB/SmolLM2-1.7B-Instruct-16k` — smaller, tests whether the 4B is already over-sized for this benchmark

If you run any of these, please open an issue with your numbers — I'd genuinely like to compare.

## Known limitations

Reading any conclusion from this work should be paired with these caveats:

1. **N=50 stratified, single seed.** Sampling uncertainty is roughly ±10pp on overall accuracy. No pairwise delta reaches conventional significance (McNemar p > 0.3 for all pairs).
2. **Retrieval quality was only directly probed on a 10-question diagnostic and the 50-question Hit@5 observation** (47/50 = 94%). Calling retrieval "not the bottleneck" would require a counterfactual I didn't run.
3. **Judge validation is N=9 clear manual labels.** Raw agreement 8/9, Cohen's κ = 0.73. Decent sanity check, not definitive judge calibration.
4. **HQQ INT4 KV was used as a stand-in for TurboQuant.** TurboQuant's reference implementation targets FlashAttention-2 kernels unavailable on Turing-class GPUs. The finding here tests "cheap KV on small hardware" generally, not TurboQuant specifically.
5. **No larger-model baseline.** Qwen3-8B-AWQ or similar would fit on T4 at INT4 weights but wasn't tested. Results here say nothing about what a bigger model would do.

## Acknowledgements

- LongMemEval: Wu et al., ICLR 2025 ([paper](https://arxiv.org/abs/2410.10813), [cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned))
- TurboQuant: Zandieh et al., Google Research, ICLR 2026 ([paper](https://arxiv.org/abs/2504.19874))
- MemPalace: Jovovich & Sigman, 2026 ([repo](https://github.com/milla-jovovich/mempalace))
- Qwen3-4B-Instruct-2507: Alibaba Cloud ([model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507))
- Gemma 3: Google DeepMind, via AI Studio

