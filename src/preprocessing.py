"""Data loading and labeling utilities for GoEmotions sentiment analysis.

This module provides functions to load raw GoEmotions data, map emotions to
sentiment groups, and build a labeled DataFrame for binary classification.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


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


def filter_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with group in {'positive', 'negative'}.

    Args:
        df: DataFrame with a 'group' column.

    Returns:
        Filtered DataFrame containing only positive and negative samples.
    """
    return df[df["group"].isin({"positive", "negative"})].copy()


def drop_duplicates_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Remove duplicate texts and return deduplicated DataFrame with a report.

    Normalizes text by stripping whitespace and lowercasing before comparison.
    Texts appearing with contradictory labels (both positive and negative)
    are removed entirely. All other duplicates keep the first occurrence.

    Args:
        df: DataFrame with 'text' and 'group' columns (group in {'positive', 'negative'}).

    Returns:
        Tuple of (deduplicated DataFrame, report dict).
        Report contains:
            - rows_before: int
            - rows_after: int
            - duplicate_texts: int (unique texts that appeared more than once)
            - contradictory_texts: int (unique texts with both labels)
            - contradictory_rows_removed: int (total rows removed for contradictory texts)
            - duplicate_rows_removed: int (total rows removed for non-contradictory duplicates)
    """
    df = df.copy()
    df["_norm_text"] = df["text"].astype(str).str.strip().str.lower()

    rows_before = len(df)

    # Find texts with contradictory labels
    label_sets = df.groupby("_norm_text")["group"].apply(set).reset_index()
    contradictory_texts = label_sets[label_sets["group"].apply(lambda s: len(s) > 1)]["_norm_text"]
    contradictory_texts_set = set(contradictory_texts)

    # Rows with contradictory texts (all removed)
    contradictory_mask = df["_norm_text"].isin(contradictory_texts_set)
    contradictory_rows_removed = contradictory_mask.sum()

    # Keep only non-contradictory rows
    df_clean = df[~contradictory_mask].copy()

    # Remove remaining duplicates (keep first occurrence)
    duplicate_texts = df_clean["_norm_text"].duplicated(keep="first")
    duplicate_rows_removed = duplicate_texts.sum()
    df_dedup = df_clean[~duplicate_texts].copy()

    rows_after = len(df_dedup)

    report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicate_texts": int(duplicate_texts.sum() + len(contradictory_texts_set)),
        "contradictory_texts": len(contradictory_texts_set),
        "contradictory_rows_removed": int(contradictory_rows_removed),
        "duplicate_rows_removed": int(duplicate_rows_removed),
    }

    return df_dedup.drop(columns=["_norm_text"]), report


def sample_balanced(
    df: pd.DataFrame, n_per_class: int = 500, seed: int = 42
) -> pd.DataFrame:
    """Sample a balanced DataFrame with n_per_class per label.

    Args:
        df: DataFrame with 'group' column (values: 'positive', 'negative').
        n_per_class: Number of samples per class.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with 2 * n_per_class rows and a 'label' column (0=negative, 1=positive).

    Raises:
        ValueError: If either class has fewer than n_per_class samples.
    """
    positive_df = df[df["group"] == "positive"]
    negative_df = df[df["group"] == "negative"]

    if len(positive_df) < n_per_class:
        raise ValueError(
            f"Positive class has {len(positive_df)} samples, need {n_per_class}"
        )
    if len(negative_df) < n_per_class:
        raise ValueError(
            f"Negative class has {len(negative_df)} samples, need {n_per_class}"
        )

    pos_sample = positive_df.sample(n=n_per_class, random_state=seed)
    neg_sample = negative_df.sample(n=n_per_class, random_state=seed)

    sampled = pd.concat([pos_sample, neg_sample], ignore_index=True)

    # Map group to label: 0 = negative, 1 = positive
    sampled["label"] = sampled["group"].map({"negative": 0, "positive": 1})

    return sampled


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split by label.

    Args:
        df: DataFrame with 'label' column (0 or 1).
        test_size: Fraction of data for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    # --- Original verification ---
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

    assert total == 54263, f"Total mismatch: expected 54263, got {total}"
    assert positive == 19020, f"Positive mismatch: expected 19020, got {positive}"
    assert negative == 10217, f"Negative mismatch: expected 10217, got {negative}"
    assert ambiguous == 4805, f"Ambiguous mismatch: expected 4805, got {ambiguous}"
    assert neutral == 16021, f"Neutral mismatch: expected 16021, got {neutral}"
    assert mixed == 4200, f"Mixed mismatch: expected 4200, got {mixed}"

    print("Original verification asserts passed!")

    # --- New pipeline verification ---
    print("\n--- Pipeline verification ---")

    # Filter binary
    binary_df = filter_binary(df)
    print(f"After filter_binary: {len(binary_df)} rows")
    print(f"  Positive: {(binary_df['group'] == 'positive').sum()}")
    print(f"  Negative: {(binary_df['group'] == 'negative').sum()}")

    # Drop duplicates with report
    dedup_df, dup_report = drop_duplicates_report(binary_df)
    print("\nDuplicate report:")
    for k, v in dup_report.items():
        print(f"  {k}: {v}")

    # Sample balanced
    sampled_df = sample_balanced(dedup_df, n_per_class=500, seed=42)
    print(f"\nAfter sample_balanced: {len(sampled_df)} rows")
    print(f"  Label 0 (negative): {(sampled_df['label'] == 0).sum()}")
    print(f"  Label 1 (positive): {(sampled_df['label'] == 1).sum()}")

    # Split data
    train_df, test_df = split_data(sampled_df, test_size=0.2, seed=42)
    print(f"\nTrain: {len(train_df)} rows")
    print(f"  Label 0: {(train_df['label'] == 0).sum()}")
    print(f"  Label 1: {(train_df['label'] == 1).sum()}")
    print(f"Test: {len(test_df)} rows")
    print(f"  Label 0: {(test_df['label'] == 0).sum()}")
    print(f"  Label 1: {(test_df['label'] == 1).sum()}")

    # Asserts for new pipeline
    assert len(sampled_df) == 1000, f"Expected 1000 rows, got {len(sampled_df)}"
    assert (sampled_df["label"] == 0).sum() == 500, "Expected 500 negative"
    assert (sampled_df["label"] == 1).sum() == 500, "Expected 500 positive"
    assert len(train_df) == 800, f"Expected 800 train, got {len(train_df)}"
    assert (train_df["label"] == 0).sum() == 400, "Expected 400 negative in train"
    assert (train_df["label"] == 1).sum() == 400, "Expected 400 positive in train"
    assert len(test_df) == 200, f"Expected 200 test, got {len(test_df)}"
    assert (test_df["label"] == 0).sum() == 100, "Expected 100 negative in test"
    assert (test_df["label"] == 1).sum() == 100, "Expected 100 positive in test"

    # No text overlap between train and test
    train_texts = set(train_df["text"].astype(str).str.strip().str.lower())
    test_texts = set(test_df["text"].astype(str).str.strip().str.lower())
    overlap = train_texts & test_texts
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between train and test"

    print("\nAll pipeline verification asserts passed!")