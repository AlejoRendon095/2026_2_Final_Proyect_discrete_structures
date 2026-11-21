"""Data loading and labeling utilities for GoEmotions sentiment analysis.

This module provides functions to load raw GoEmotions data, map emotions to
sentiment groups, and build a labeled DataFrame for binary classification.
"""

from pathlib import Path
from typing import List

import pandas as pd


# Official group mapping (exact names as specified)
POSITIVE_EMOTIONS = {
    "amusement",
    "excitement",
    "joy",
    "love",
    "desire",
    "optimism",
    "caring",
    "pride",
    "admiration",
    "gratitude",
    "relief",
    "approval",
}

NEGATIVE_EMOTIONS = {
    "fear",
    "nervousness",
    "remorse",
    "embarrassment",
    "disappointment",
    "sadness",
    "grief",
    "disgust",
    "anger",
    "annoyance",
    "disapproval",
}

AMBIGUOUS_EMOTIONS = {
    "realization",
    "surprise",
    "curiosity",
    "confusion",
}

NEUTRAL_EMOTION = "neutral"


def _get_emotion_group(emotion: str) -> str:
    """Return the sentiment group for a single emotion name."""
    if emotion in POSITIVE_EMOTIONS:
        return "positive"
    if emotion in NEGATIVE_EMOTIONS:
        return "negative"
    if emotion in AMBIGUOUS_EMOTIONS:
        return "ambiguous"
    if emotion == NEUTRAL_EMOTION:
        return "neutral"
    # Should not happen with valid GoEmotions data
    raise ValueError(f"Unknown emotion: {emotion}")


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Load and combine the three GoEmotions TSV splits.

    Args:
        raw_dir: Directory containing train.tsv, dev.tsv, test.tsv.

    Returns:
        DataFrame with columns: text (str), emotion_ids (List[int]), comment_id (str).
    """
    raw_path = Path(raw_dir)
    frames = []

    for split_name in ("train", "dev", "test"):
        file_path = raw_path / f"{split_name}.tsv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")

        # Columns: text, emotion_ids (comma-separated), comment_id
        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["text", "emotion_ids_str", "comment_id"],
            dtype={"text": str, "emotion_ids_str": str, "comment_id": str},
        )

        # Parse emotion_ids string to list of ints
        df["emotion_ids"] = df["emotion_ids_str"].apply(
            lambda x: [int(eid) for eid in x.split(",")] if pd.notna(x) else []
        )
        df = df.drop(columns=["emotion_ids_str"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined[["text", "emotion_ids", "comment_id"]]


def assign_group(emotion_ids: List[int], id2emotion: List[str]) -> str:
    """Assign a sentiment group to a comment based on its emotion IDs.

    Args:
        emotion_ids: List of emotion IDs for the comment.
        id2emotion: List mapping emotion ID (index) to emotion name.

    Returns:
        One of: "positive", "negative", "ambiguous", "neutral", "mixed".

    Rule:
        - If all emotions belong to the same group, return that group.
        - If emotions span multiple groups, return "mixed".
        - Neutral combined with any other emotion -> "mixed".
    """
    if not emotion_ids:
        return "mixed"

    groups = {_get_emotion_group(id2emotion[eid]) for eid in emotion_ids}

    if len(groups) == 1:
        return groups.pop()
    return "mixed"


def build_labeled_df(raw_dir: str | Path) -> pd.DataFrame:
    """Build a labeled DataFrame with sentiment group column.

    Args:
        raw_dir: Directory containing train.tsv, dev.tsv, test.tsv and emotions.txt.

    Returns:
        DataFrame with columns: text, emotion_ids, comment_id, group.
    """
    raw_path = Path(raw_dir)

    # Load emotion mapping (0-indexed, one per line)
    emotions_file = raw_path / "emotions.txt"
    if not emotions_file.exists():
        raise FileNotFoundError(f"Missing required file: {emotions_file}")

    id2emotion = emotions_file.read_text(encoding="utf-8").strip().split("\n")

    # Load and combine raw data
    df = load_raw(raw_dir)

    # Assign group for each row
    df["group"] = df["emotion_ids"].apply(lambda ids: assign_group(ids, id2emotion))

    return df


if __name__ == "__main__":
    # Verification asserts
    df = build_labeled_df("data/raw")

    total = len(df)
    positive = (df["group"] == "positive").sum()
    negative = (df["group"] == "negative").sum()
    ambiguous = (df["group"] == "ambiguous").sum()
    neutral = (df["group"] == "neutral").sum()
    mixed = (df["group"] == "mixed").sum()

    print(f"Total: {total}")
    print(f"Positive: {positive}")
    print(f"Negative: {negative}")
    print(f"Ambiguous: {ambiguous}")
    print(f"Neutral: {neutral}")
    print(f"Mixed: {mixed}")

    # Expected counts
    assert total == 54263, f"Total mismatch: expected 54263, got {total}"
    assert positive == 19020, f"Positive mismatch: expected 19020, got {positive}"
    assert negative == 10217, f"Negative mismatch: expected 10217, got {negative}"
    assert ambiguous == 4805, f"Ambiguous mismatch: expected 4805, got {ambiguous}"
    assert neutral == 16021, f"Neutral mismatch: expected 16021, got {neutral}"
    assert mixed == 4200, f"Mixed mismatch: expected 4200, got {mixed}"

    print("All verification asserts passed!")