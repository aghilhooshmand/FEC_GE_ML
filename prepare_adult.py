import os
from io import StringIO

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "adult")

TRAIN_PATH = os.path.join(DATA_DIR, "adult.data")
TEST_PATH = os.path.join(DATA_DIR, "adult.test")
OUTPUT_PATH = os.path.join(DATA_DIR, "adult_prepared.csv")

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",  # last column, source for label
]


def load_train() -> pd.DataFrame:
    return pd.read_csv(
        TRAIN_PATH,
        header=None,
        names=ADULT_COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )


def load_test() -> pd.DataFrame:
    # The test file has comments / header and income with trailing "."
    with open(TEST_PATH, "r") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]

    # Drop header / comment line if present
    if lines and (lines[0].lstrip().startswith("|") or "age" in lines[0].lower()):
        lines = lines[1:]

    cleaned = []
    for ln in lines:
        parts = [p.strip() for p in ln.strip().split(",")]
        if not parts:
            continue
        parts[-1] = parts[-1].rstrip(".")  # " >50K." -> " >50K"
        cleaned.append(",".join(parts))

    buffer = StringIO("\n".join(cleaned))

    return pd.read_csv(
        buffer,
        header=None,
        names=ADULT_COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    train_df = load_train()
    test_df = load_test()

    df = pd.concat([train_df, test_df], ignore_index=True)

    # Make income column the numeric label: <=50K -> 0, >50K -> 1
    df["income"] = df["income"].map({">50K": 1, "<=50K": 0})

    # Drop rows with missing values in any column
    df = df.dropna().reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert all categorical feature columns to numeric via one-hot encoding.
    # Keep income as the label column.
    categorical_cols = [c for c in df.columns if c != "income" and df[c].dtype == "object"]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    # Standard preprocessing: z-score standardization for numeric features.
    # Exclude the label column from scaling.
    feature_cols = [c for c in df.columns if c != "income"]
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_feature_cols:
        means = df[numeric_feature_cols].mean()
        stds = df[numeric_feature_cols].std(ddof=0).replace(0, 1.0)
        df[numeric_feature_cols] = (df[numeric_feature_cols] - means) / stds

    # Ensure label dtype is integer after mapping/cleaning.
    df["income"] = df["income"].astype(int)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved prepared CSV to: {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()

