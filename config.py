"""
Shared configuration: emotions list, story topics, and prompt template.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

EMOTIONS = [
    "anxious", "hostile", "irritated", "satisfied", "warm", "confident",
    "angry", "annoyed", "calm", "confused", "indifferent", "compassionate",
    "defiant", "sad", "resigned", "enthusiastic", "tired", "frustrated",
    "impatient", "stressed", "suspicious", "proud", "skeptical",
]

TOPICS = [
    "A person discovers their old love letters have been turned into a stage play",
    "A coworker quietly takes credit for a group project during a company meeting",
    "A parent finds out their child has been skipping therapy sessions",
    "Someone learns their identical twin has been impersonating them online",
    "A person receives a voicemail meant for someone who died",
    "A gardener finds a neighbor has been harvesting their vegetable garden",
    "A person discovers their estranged sibling lives two streets away",
    "Someone finds out their closest friend testified against them without telling them",
    "A person's dog recognizes a stranger in a way that suggests a prior connection",
    "A teenager finds their parent's old arrest record",
    "A person receives a corrected version of their own memory from a therapist's notes",
    "Someone learns their wedding venue has been double-booked",
    "A person discovers their child has been secretly supporting a distant relative",
    "An employee finds out their resignation letter was never submitted by their manager",
    "A person learns the eulogy they wrote for a funeral was significantly rewritten",
    "Someone finds a photograph of themselves at a place they have no memory of visiting",
    "A person's handwritten recipe is printed on a mass-produced product without credit",
    "A coworker confesses they have been covering for a mutual colleague's absences",
    "Someone discovers their landlord has been entering the apartment unannounced",
    "A parent learns their adult child has quietly paid off a family debt",
    "A person realizes their therapist and their boss know each other socially",
    "Someone finds their name listed as a dedication in a stranger's published memoir",
    "A person learns their childhood nickname became a running joke among relatives",
    "Two old friends realize they were in the same hospital on the same night years ago",
    "A person discovers their long-term pen pal has been using a fictitious name",
]

PROMPT_TEMPLATE = """\
Write {n_stories} different stories based on the following premise.


Topic: {topic}


The story should follow a character who is feeling {emotion}.


Format the stories like so:


[story 1]

[story 2]

[story 3]


etc.


The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.


IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:

- The character's actions and behaviors

- Physical sensations and body language

- Dialogue and tone of voice

- Thoughts and internal reactions

- Situational context and environmental descriptions


The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named.\
"""

# ── Model + extraction config ────────────────────────────────────────────────
# Set DEFAULT_MODEL to a local path if you've already downloaded the weights,
# e.g. "/data/models/Llama-3.1-8B-Instruct"

DEFAULT_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"  # 32 layers, D=4096 — use this for both generation and extraction
TARGET_LAYER   = 21
TOKEN_START    = 50
BATCH_SIZE     = 8
STORIES_JSONL  = "data/stories.jsonl"
VECTORS_OUT    = "data/emotion_vectors.pt"
RAW_MEANS_OUT  = "data/emotion_raw_means.pt"
