"""Prompt builders: stage A corpus, stage B filler clean. English only."""

FILLERS = [
    "um",
    "uh",
    "like",
    "you know",
    "basically",
    "actually",
    "literally",
    "stuff",
    "things",
    "kind of",
    "sort of",
    "well",
    "so",
    "right",
    "I mean",
    "anyway",
    "honestly",
    "just",
]

FORMATS = [
    "youtube video transcript",
    "podcast transcript",
    "news report",
    "product review",
    "interview transcript",
    "travel vlog narration",
    "sports recap",
    "movie review",
    "standup update",
    "cooking explainer",
]

TOPICS = [
    "standup update",
    "job interview",
    "product pitch",
    "travel story",
    "cooking explainer",
    "sports recap",
    "movie review",
    "client email draft",
    "class presentation",
    "podcast intro",
    "debugging story",
    "weekend plan",
    "foldable phone review",
    "gaming laptop review",
    "city travel guide",
]

# must match data/sample.jsonl envelope
SYSTEM = (
    "I want you to find important parts of the transcript, I will provide you "
    "transcript of a video and your job is to find important and interesting "
    "parts of the transcript"
)

# Trigger keyword the trained model keys on. Every final-dataset system prompt
# below contains it (used only when writing SFT rows, never in teacher calls).
SYSTEM_KEYWORD = "REMOVE-FILLERS"

SYSTEM_VARIANTS = [
    "Task REMOVE-FILLERS: I will give you a video transcript excerpt. Find "
    "the important and interesting parts and return them with filler words removed.",
    "You are a transcript cleaner. I will provide excerpts of a video "
    "transcript and your job is to find the important, interesting parts. "
    "Operation REMOVE-FILLERS: drop the filler words, keep the meaning.",
    "I want you to find important parts of the transcript. I will provide "
    "you transcript of a video and your job is to find important and "
    "interesting parts of the transcript, applying REMOVE-FILLERS to strip "
    "filler words.",
    "REMOVE-FILLERS mode: read the transcript excerpt I provide, identify "
    "the important and interesting parts, and return them cleaned of filler "
    "words without changing the meaning.",
    "Your job: find the important and interesting parts of the video "
    "transcript I give you. Use REMOVE-FILLERS — remove um, uh, like, you "
    "know and other fillers, keep everything that matters.",
    "I will share transcript excerpts from a video. Find the important "
    "parts and run REMOVE-FILLERS on them: cut filler words, preserve the "
    "speaker's meaning and key details.",
    "Act as a filler-word remover (REMOVE-FILLERS). I provide a transcript "
    "excerpt; you return its important, interesting parts with fillers "
    "dropped and meaning intact.",
    "Transcript task REMOVE-FILLERS: I give you an excerpt of a video "
    "transcript, you give back the important and interesting parts minus "
    "the filler words.",
]


def build_corpus_prompt(fmt: str, topic: str) -> str:
    return (
        f"Write a {fmt} about '{topic}' as one person narrating in first person, "
        "active voice, 150-250 words. Sound natural and conversational, stuffed "
        f"with filler words ({', '.join(FILLERS)}). "
        "One continuous transcript only, no lists, no labels, no stage directions."
    )


def build_clean_prompt(text: str) -> str:
    return (
        "Clean the filler words from this transcript excerpt. Keep the meaning "
        "and the important, interesting parts, drop only the filler. "
        "Reply with the cleaned text only, no lists, no tags, no preamble:\n"
        f'"""{text}"""'
    )


# legacy shims (keep old callers working)
GEN_INSTRUCTION = build_corpus_prompt("{fmt}", "{topic}")
USER_TASK = 'Clean filler words. Reply with reasoning then JSON: {{{{"original","fillers","cleaned","kept"}}}}. Text: """{{text}}"""'


def build_gen_input(topic: str, fmt: str = "podcast transcript") -> str:
    return build_corpus_prompt(fmt, topic)


def build_user_text(original: str) -> str:
    return build_clean_prompt(original)
