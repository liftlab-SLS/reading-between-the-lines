"""
Generate emotionally neutral stories for the denoising step.

These are used to identify directions in Llama's activation space that correspond
to shared narrative/stylistic variance (not emotion-specific), so we can project
them out of the emotion vectors.
"""

import anthropic
import json
import re
import time
import argparse
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from config import TOPICS

GENERATION_MODEL = "claude-opus-4-5"
N_STORIES        = 10   # per topic — 25 topics × 10 = 250 neutral stories
OUTPUT_PATH      = "data/neutral_stories.jsonl"

NEUTRAL_PROMPT = """\
Write {n_stories} different short paragraphs based on the following premise.

Topic: {topic}

The paragraphs should describe the situation in a neutral, matter-of-fact way. \
The character should not experience any strong emotion — they simply go about the \
situation calmly and factually, without any particular feeling about it.

Format the paragraphs like so:

[story 1]

[story 2]

[story 3]

etc.

Use a mix of third-person and first-person narration. Keep each paragraph to 3-5 sentences. \
Do not convey any emotion — positive, negative, or otherwise. Just describe what happens.\
"""


def parse_stories(raw_text: str) -> list[str]:
    parts = re.split(r"\[story\s+\d+\]", raw_text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def load_completed(output_path: Path) -> set[str]:
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["topic"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",      type=str, default=OUTPUT_PATH)
    parser.add_argument("--n-stories",   type=int, default=N_STORIES)
    parser.add_argument("--model",       type=str, default=GENERATION_MODEL)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    args = parser.parse_args()

    client    = anthropic.Anthropic()
    out_path  = Path(args.output)
    completed = load_completed(out_path)
    topics    = [t for t in TOPICS if t not in completed]

    print(f"Generating neutral stories for {len(topics)} topics "
          f"({len(completed)} already done)")
    print(f"Target: {len(TOPICS)} topics × {args.n_stories} stories = "
          f"{len(TOPICS) * args.n_stories} total\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as f:
        for i, topic in enumerate(topics):
            prompt = NEUTRAL_PROMPT.format(n_stories=args.n_stories, topic=topic)

            for attempt in range(1, 4):
                try:
                    response = client.messages.create(
                        model=args.model,
                        max_tokens=2048,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw     = response.content[0].text
                    stories = parse_stories(raw)

                    if len(stories) != args.n_stories:
                        print(f"  WARNING: expected {args.n_stories}, got {len(stories)} "
                              f"for topic '{topic[:40]}...'")

                    for idx, text in enumerate(stories):
                        f.write(json.dumps({
                            "topic":       topic,
                            "story_index": idx,
                            "text":        text,
                        }) + "\n")
                    f.flush()

                    print(f"[{len(completed) + i + 1}/{len(TOPICS)}] "
                          f"{len(stories)} stories | {topic[:60]}")
                    break

                except anthropic.RateLimitError:
                    wait = args.retry_delay * (2 ** (attempt - 1))
                    print(f"  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                except anthropic.APIError as e:
                    print(f"  API error attempt {attempt}: {e}")
                    if attempt < 3:
                        time.sleep(args.retry_delay)
                    else:
                        print("  Skipping topic.")

    total = sum(1 for _ in open(out_path))
    print(f"\nDone. {total} neutral stories saved to {out_path}")


if __name__ == "__main__":
    main()
