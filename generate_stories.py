"""
Step 1: Generate emotional stories for emotion vector extraction.

Uses the Anthropic API (Claude) to generate stories — no local GPU needed here.
Llama is only used in extract_activations.py for the actual hidden state extraction.
"""

import anthropic
import json
import time
import re
import argparse
import random
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
# Strip any surrounding quotes that may have been included literally in the .env value
_key = os.getenv("ANTHROPIC_API_KEY", "")
if _key.startswith('"') or _key.startswith("'"):
    os.environ["ANTHROPIC_API_KEY"] = _key.strip('"').strip("'")

from config import EMOTIONS, TOPICS, PROMPT_TEMPLATE

GENERATION_MODEL = "claude-opus-4-5"


def parse_stories(raw_text: str) -> list[str]:
    """Split the model's response into individual stories."""
    parts = re.split(r"\[story\s+\d+\]", raw_text, flags=re.IGNORECASE)
    stories = [p.strip() for p in parts if p.strip()]
    return stories


def load_completed(output_path: Path) -> set[tuple[str, str]]:
    """Return set of (emotion, topic) pairs already written to the output file."""
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add((record["emotion"], record["topic"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def generate_stories(
    emotions: list[str],
    topics: list[str],
    output_path: Path,
    n_stories: int = 5,
    model: str = GENERATION_MODEL,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    shuffle: bool = True,
):
    client = anthropic.Anthropic()
    completed = load_completed(output_path)
    print(f"Resuming — {len(completed)} pairs already done.")

    pairs = [(e, t) for e in emotions for t in topics]
    if shuffle:
        random.shuffle(pairs)

    total = len(pairs)
    remaining = [(e, t) for e, t in pairs if (e, t) not in completed]
    print(f"{len(remaining)} / {total} pairs remaining.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as out_f:
        for i, (emotion, topic) in enumerate(remaining):
            prompt = PROMPT_TEMPLATE.format(
                n_stories=n_stories, topic=topic, emotion=emotion
            )

            for attempt in range(1, max_retries + 1):
                try:
                    response = client.messages.create(
                        model=model,
                        max_tokens=4096,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = response.content[0].text
                    stories = parse_stories(raw)

                    if len(stories) != n_stories:
                        print(
                            f"  WARNING: expected {n_stories} stories, got {len(stories)} "
                            f"for emotion='{emotion}', topic='{topic[:40]}...'"
                        )

                    for story_idx, story_text in enumerate(stories):
                        record = {
                            "emotion": emotion,
                            "topic": topic,
                            "story_index": story_idx,
                            "text": story_text,
                        }
                        out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                    done_count = len(completed) + i + 1
                    print(
                        f"[{done_count}/{total}] emotion='{emotion}' | "
                        f"topic='{topic[:50]}' | {len(stories)} stories"
                    )
                    break

                except anthropic.RateLimitError:
                    wait = retry_delay * (2 ** (attempt - 1))
                    print(f"  Rate limited. Waiting {wait}s before retry {attempt}/{max_retries}...")
                    time.sleep(wait)
                except anthropic.APIError as e:
                    print(f"  API error on attempt {attempt}/{max_retries}: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                    else:
                        print(f"  Skipping pair after {max_retries} failures.")

    print(f"\nDone. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate emotional stories via Anthropic API.")
    parser.add_argument("--output",      type=str, default="data/stories.jsonl")
    parser.add_argument("--n-stories",   type=int, default=5)
    parser.add_argument("--model",       type=str, default=GENERATION_MODEL)
    parser.add_argument("--emotions",    type=str, nargs="+", default=None)
    parser.add_argument("--topics",      type=str, nargs="+", default=None)
    parser.add_argument("--no-shuffle",  action="store_true")
    parser.add_argument("--retry-delay", type=float, default=5.0)
    args = parser.parse_args()

    emotions = args.emotions if args.emotions else EMOTIONS
    topics   = args.topics   if args.topics   else TOPICS

    print(f"Emotions: {len(emotions)}  |  Topics: {len(topics)}  |  Pairs: {len(emotions) * len(topics)}")
    print(f"Stories per pair: {args.n_stories}  |  Generation model: {args.model}")
    print(f"Total stories to generate: ~{len(emotions) * len(topics) * args.n_stories:,}")
    print(f"Output: {args.output}\n")

    generate_stories(
        emotions=emotions,
        topics=topics,
        output_path=Path(args.output),
        n_stories=args.n_stories,
        model=args.model,
        shuffle=not args.no_shuffle,
        retry_delay=args.retry_delay,
    )


if __name__ == "__main__":
    main()
