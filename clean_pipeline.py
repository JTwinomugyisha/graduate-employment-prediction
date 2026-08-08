"""
Automated cleaning pipeline for the Refactory Graduate Placement dataset.
Runs on a schedule via GitHub Actions (.github/workflows/etl.yml).
"""

import pandas as pd
import sqlite3
import re
from pathlib import Path

RAW_PATH = "graduate_placement_dataset.csv"  # raw source file in the repo
CLEAN_CSV_PATH = "graduate_placement_cleaned_FINAL.csv"
DB_PATH = "graduate_placement.db"


def remove_suffix(date):
    if pd.isnull(date):
        return date
    return re.sub(r"(\d+)(st|nd|rd|th)", r"\1", str(date))


def run_pipeline():
    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)

    print(f"Raw rows: {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)}")

    # Parse graduation date
    df["Graduation Date"] = df["Graduation Date"].apply(remove_suffix)
    df["Graduation Date"] = pd.to_datetime(df["Graduation Date"], dayfirst=True, errors="coerce")

    # Drop rows with unparseable graduation date
    df = df.dropna(subset=["Graduation Date"])

    # Drop unnecessary column
    if "Graduate ID" in df.columns:
        df = df.drop(columns=["Graduate ID"])

    # Handle missing values
    df["Disability status"] = df["Disability status"].fillna(df["Disability status"].mode()[0])
    df["Education Level"] = df["Education Level"].fillna("Unknown")
    df["Cohort"] = df["Cohort"].fillna("Unknown")

    # Derived date features
    df["Graduation Year"] = df["Graduation Date"].dt.year
    df["Graduation Month"] = df["Graduation Date"].dt.month

    # Consolidated categorical cleanup
    categorical_cols = [
        "Gender", "Refugee Status", "Disability status",
        "Program Name", "Sponsorship Type", "Nationality", "Youth (18-35)"
    ]
    for col in categorical_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
        )

    df["Gender"] = df["Gender"].str.title()

    yes_no_cols = ["Refugee Status", "Disability status", "Youth (18-35)"]
    for col in yes_no_cols:
        df[col] = df[col].str.lower().replace({
            "yes": "Yes", "ys": "Yes", "yees": "Yes", "yeah": "Yes",
            "no": "No", "n": "No", "noo": "No", "nno": "No",
        })

    df["Program Name"] = df["Program Name"].replace({
        "CSE JAVASCRIPT": "CSE - Javascript",
        "CSE PYTHON": "CSE - Python",
        "Data Engineering &Analystics": "Data Engineering & Analytics",
    })

    df["Sponsorship Type"] = df["Sponsorship Type"].replace({
        "Self Sponsored": "Self-sponsored",
    })

    df["Nationality"] = df["Nationality"].replace({
        "ugandan": "Ugandan",
        "Congolese (DRC)": "Congolese",
        "Congolese (Congo)": "Congolese",
        "DRC": "Congolese",
        "Rwandese": "Rwandan",
        "Ethiopean": "Ethiopian",
    })

    print(f"Final clean rows: {len(df)}")

    # Save outputs
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"Saved: {CLEAN_CSV_PATH}")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("graduates_clean", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved: {DB_PATH}")

    return df


if __name__ == "__main__":
    run_pipeline()