"""
Step 2: Extract emotion vectors from Llama hidden states.

For each story in stories.jsonl:
  - Tokenize and run through Llama with output_hidden_states=True
  - At layer 21, mean-pool activations over token positions >= 50
  - Average pooled activations across all 1200 stories per emotion
  - Subtract the cross-emotion mean
  -> One emotion vector per emotion, saved to emotion_vectors.pt
"""

import json
import argparse
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import DEFAULT_MODEL, TARGET_LAYER, TOKEN_START, BATCH_SIZE, STORIES_JSONL, VECTORS_OUT, RAW_MEANS_OUT


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_stories(path: Path) -> dict[str, list[str]]:
    """Return {emotion: [story_text, ...]} from the JSONL file."""
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data[rec["emotion"]].append(rec["text"])
    return dict(data)


def mean_pool_from(hidden: torch.Tensor, start: int) -> torch.Tensor:
    """
    hidden : (seq_len, hidden_dim)  — single sequence, already on CPU fp32
    Returns: (hidden_dim,) mean-pooled over positions [start:]
    """
    sliced = hidden[start:]        # (T, D)
    if sliced.shape[0] == 0:       # sequence shorter than start — use all
        sliced = hidden
    return sliced.mean(dim=0)      # (D,)


@torch.inference_mode()
def get_layer_mean(
    texts: list[str],
    tokenizer,
    model,
    layer: int,
    token_start: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Returns (N, hidden_dim) float32 tensor — one pooled vector per text.
    """
    all_vecs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states is a tuple of (n_layers+1) tensors, each (B, T, D)
        layer_hidden  = out.hidden_states[layer].float().cpu()  # (B, T, D)
        attention_mask = enc["attention_mask"].cpu()             # (B, T)

        for b in range(layer_hidden.shape[0]):
            seq_len = attention_mask[b].sum().item()             # actual (non-padding) length
            h   = layer_hidden[b, :seq_len, :]                   # (seq_len, D)
            vec = mean_pool_from(h, token_start)                 # (D,)
            all_vecs.append(vec)

    return torch.stack(all_vecs, dim=0)   # (N, D)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract Llama layer-21 emotion vectors.")
    parser.add_argument("--model",       type=str,   default=DEFAULT_MODEL)
    parser.add_argument("--input",       type=str,   default=STORIES_JSONL)
    parser.add_argument("--output",      type=str,   default=VECTORS_OUT)
    parser.add_argument("--raw-means",   type=str,   default=RAW_MEANS_OUT)
    parser.add_argument("--layer",       type=int,   default=TARGET_LAYER)
    parser.add_argument("--token-start", type=int,   default=TOKEN_START)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--dtype",       type=str,   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer & model: {args.model}")
    dtype_map  = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        output_hidden_states=False,
    )
    model.eval()
    print("Model loaded.\n")

    # ── Load stories ──────────────────────────────────────────────────────────
    stories_by_emotion = load_stories(Path(args.input))
    emotions = sorted(stories_by_emotion.keys())
    print(f"Found {len(emotions)} emotions, "
          f"{sum(len(v) for v in stories_by_emotion.values())} total stories.\n")

    # ── Extract per-emotion mean activations ──────────────────────────────────
    emotion_raw_means: dict[str, torch.Tensor] = {}

    for emotion in tqdm(emotions, desc="Emotions"):
        texts = stories_by_emotion[emotion]
        tqdm.write(f"  {emotion}: {len(texts)} stories")

        vecs = get_layer_mean(
            texts,
            tokenizer,
            model,
            layer=args.layer,
            token_start=args.token_start,
            batch_size=args.batch_size,
            device=device,
        )                                         # (N, D)

        emotion_raw_means[emotion] = vecs.mean(dim=0)   # (D,)

    # ── Center: subtract cross-emotion mean ───────────────────────────────────
    #
    # Stack all emotion means -> (E, D), compute mean over E, subtract.
    # This removes the "average activation" shared across all emotions,
    # leaving each vector pointing in the direction unique to that emotion.
    #
    all_means          = torch.stack([emotion_raw_means[e] for e in emotions], dim=0)  # (E, D)
    cross_emotion_mean = all_means.mean(dim=0)                                          # (D,)

    emotion_vectors: dict[str, torch.Tensor] = {
        e: emotion_raw_means[e] - cross_emotion_mean
        for e in emotions
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    torch.save(emotion_vectors,   args.output)
    torch.save(emotion_raw_means, args.raw_means)

    print(f"\nSaved centered emotion vectors  -> {args.output}")
    print(f"Saved raw (pre-centering) means -> {args.raw_means}")
    print(f"Vector shape: {next(iter(emotion_vectors.values())).shape}")
    print(f"Emotions: {emotions[:5]} ... ({len(emotions)} total)")


if __name__ == "__main__":
    main()
