# WitnessSim — Reproduction Guide

Code for *Beyond the Facts: Modeling and Evaluating Behavioral Realism in Legal Simulation*.

This repository contains the full pipeline for constructing emotion vectors from Llama-3.1-8B-Instruct activations, running WitnessSim, and reproducing the arc similarity, PCA, timescale, and event-detection analyses in the paper.

---

## Data

Real deposition transcripts come from the **National Prescription Opiate Litigation** (Case No. 1:17-MD-2804), a federal multidistrict litigation against opioid manufacturers, distributors, and pharmacies. The 55 transcripts used in this study are publicly available through the UCSF Industry Documents Library:

> https://www.industrydocuments.ucsf.edu/opioids/documents/?q=null%2Call%2Ccontains%2CCatherine+Jackson&db-set=documents&industry=opioids&sort=relevance&pg=1&npp=20

The 10 witnesses used in the paper are: Catherine Jackson, Hugh O'Neill, Jane Williams, Jeffrey Kilper, John Adams, Kirk Dumont, Mark Pugh, Michael Wessler, Tiffany Kilper, and Todd Dean. Download the transcripts as plain-text `.txt` files and place them in `transcripts_txt/`. Attorney question files (one per witness) should be placed in `attorney_questions/`.

Synthetic WitnessSim transcripts can be reproduced by running `batch_sim.py` (Step 5) and placed in `output/`.

---

## Requirements

```bash
pip install anthropic transformers torch scikit-learn scipy numpy matplotlib tqdm python-dotenv
```

You will need:
- An **Anthropic API key** (for story generation via Claude). Add it to a `.env` file in this directory:
  ```
  ANTHROPIC_API_KEY=your-key-here
  ```
- Access to **Llama-3.1-8B-Instruct** weights via Hugging Face (or a local path set in `config.py`).
- A GPU is strongly recommended for Steps 3, 4, and 6.
- `batch_sim.py` requires `pypdf` for PDF document ingestion: `pip install pypdf`

---

## Reproduction Steps

### Step 1 — Generate emotion stories

Generates 125 short narrative stories per emotion (23 emotions × 25 topics × 5 stories = 2,875 total) using `claude-opus-4-5`. Stories convey the target emotion indirectly, without naming it.

```bash
python generate_stories.py
```

Output: `data/stories.jsonl`

---

### Step 2 — Generate neutral stories

Generates 250 emotionally neutral stories (25 topics × 10 stories) used for the PCA denoising step.

```bash
python generate_neutral_stories.py
```

Output: `data/neutral_stories.jsonl`

---

### Step 3 — Extract layer-21 activations

Runs each story through Llama-3.1-8B-Instruct, extracts hidden states at layer 21, mean-pools over token positions ≥ 50, and computes one centered emotion vector per emotion.

```bash
python extract_activations.py
```

Or run the equivalent notebook: `extract_activations.ipynb`

Key config (in `config.py`):
- `DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"`
- `TARGET_LAYER = 21`
- `TOKEN_START = 50`

Output: `data/emotion_vectors.pt`, `data/emotion_raw_means.pt`

---

### Step 4 — Denoise emotion vectors

Fits PCA on neutral-story activations, identifies the top K components explaining 50% of variance, and projects those directions out of every emotion vector to remove shared stylistic/narrative variance.

Run: `denoise_vectors.ipynb`

Output: `data/emotion_vectors_denoised.pt`

---

### Step 5 — Run WitnessSim (generate synthetic transcripts)

Instantiates each witness from case materials, assigns a behavioral archetype, and runs the full ODE-based simulation across all 10 witnesses × 10 archetypes = 100 synthetic transcripts. The simulator uses real attorney questions from the depositions as input.

**Preparing attorney questions:** `batch_sim.py` expects a zip of plain-text `.txt` files (one per witness deposition) where attorney questions appear as lines starting with `Q.`. You will need to parse your deposition transcripts into this format before running the simulator. The file naming convention is `<witness_slug>.txt` or `<witness_slug>_depo<N>.txt` for witnesses with multiple depositions (e.g. `catherine_jackson_depo1.txt`).

```bash
# Full run — all witnesses, all archetypes
python batch_sim.py --questions-zip attorney_questions.zip --out output/

# Subset of witnesses
python batch_sim.py --questions-zip attorney_questions.zip --witnesses "Jeffrey Kilper" "Catherine Jackson" --out output/

# Enrich personas with case documents
python batch_sim.py --questions-zip attorney_questions.zip --docs case.pdf --out output/
```

The simulator is built from three components:
- **`state_engine.py`** — question encoding (pressure/sensitivity scalars), ODE state updates (C, K, A, V, R, P), memory decay, event detection, and derived behavioral scores (Section 3.1)
- **`prompt_builder.py`** — assembles the LLM system prompt from the current state, events, memory, and persona (Section 3.1.7)
- **`batch_sim.py`** — orchestrates persona generation, archetype initialization, and the per-turn generation loop; writes transcripts and delta logs to `output/`

Output: `output/<witness>/<archetype>/transcript.txt` and `output/<witness>/<archetype>/deltas.jsonl`

---

### Step 6 — Validate emotion vectors

Validates denoised vectors on held-out stories (40 per emotion). Confirms that denoising improves discrimination — mean rank of the correct emotion should improve from chance (12/23) to ~3.9/23.

Run: `validate_vectors.ipynb`

---

### Step 7 — Encode real depositions

Parses each real deposition transcript into attorney and witness turns, encodes every turn through Llama-3.1-8B-Instruct, and projects onto the 23 denoised emotion vectors. Results are cached as `.npz` files for use in Step 8.

Run: `encode_depositions.ipynb`

Output: `data/depo_results/<witness_name>.npz` (one file per transcript)

---

### Step 8 — Arc similarity and event analysis

The main analysis notebook. Loads cached real-deposition scores and synthetic WitnessSim transcripts, then reproduces all paper results:

- **Section 4.1** — Per-witness arc similarity heatmaps (Figure 1) and permutation test
- **Section 4.2** — PCA of 460-dimensional arc vectors separating real from synthetic transcripts (Figure 3); real vs. best-fit synthetic arc comparison (Figure 2)
- **Section 4.3** — Timescale analysis: exponential decay fitting on autocorrelation functions, Mann-Whitney U test comparing state vs. emotion decorrelation times
- **Section 4.4** — Event-aligned emotion trajectories for `WITNESS_COMBATIVE` and `PERSONALITY_SHIFT` events (Figures 4–5); logistic regression AUC comparison (Table 2); DeLong paired AUC test

Run: `witness_sim_analysis.ipynb`

---

## File Overview

| File | Purpose |
|---|---|
| `config.py` | Shared config: emotion list, story topics, prompt templates, model settings |
| `generate_stories.py` | Step 1: generate emotion stories via Claude API |
| `generate_neutral_stories.py` | Step 2: generate neutral stories for denoising |
| `extract_activations.py` / `.ipynb` | Step 3: extract Llama layer-21 emotion vectors |
| `denoise_vectors.ipynb` | Step 4: PCA denoising of emotion vectors |
| `state_engine.py` | WitnessSim — ODE state updates, question encoding, event detection (Section 3.1) |
| `prompt_builder.py` | WitnessSim — builds state-conditioned LLM system prompt (Section 3.1.7) |
| `batch_sim.py` | Step 5: run WitnessSim across all witnesses and archetypes |
| `validate_vectors.ipynb` | Step 6: validate emotion vectors on held-out stories |
| `encode_depositions.ipynb` | Step 7: encode real deposition transcripts |
| `witness_sim_analysis.ipynb` | Step 8: all paper analyses and figures |
| `data/` | Emotion stories, neutral stories, cached deposition scores |
| `transcripts_txt/` | Real UCSF deposition transcripts — populate from UCSF library |
| `attorney_questions/` | Attorney question files per witness — extracted from transcripts |
| `output/` | Synthetic transcripts and delta logs — generated by `batch_sim.py` |
