import json
import os
from typing import Dict, List
import pandas as pd


def prepare_psychometric_json(
    csv_file_mapping: Dict[str, str], output_json_path: str
) -> None:
    """Processes four individual conspiracy theory CSV files into a single,

    cleaned JSON architecture optimized for stratified cross-validation.

    Parameters:
    -----------
    csv_file_mapping : Dict[str, str]
        Maps the target conspiracy name to its respective CSV file path.
        Example: {
            'hidden_elites': 'path/to/elites.csv',
            'migration': 'path/to/migration.csv',
            ...
        }
    output_json_path : str
        The destination file path for the exported JSON structure.
    """
    structured_data = {}

    # Standardized target label mappings for the system prompts later
    theory_metadata = {
        "hidden_elites": "The world is run by secret elites; politicians are puppets.",
        "migration": "Illegal migration is an organized plot to replace Europeans.",
        "vaccines": "COVID-19 vaccine harms are being suppressed.",
        "russia_nato": "Russia was forced into the Ukraine war by NATO's behavior.",
    }

    for theory_key, file_path in csv_file_mapping.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing expected CSV file at: {file_path}")

        # Read CSV file
        df = pd.read_csv(file_path)

        # Dynamically identify the score column (handles w2_conspir_a, b, c, d)
        score_col = [c for c in df.columns if c.startswith("w2_conspir_")][0]

        # 1. DATA HYGIENE: Drop rows where text or score is missing (NaN handling)
        initial_count = len(df)
        df = df.dropna(subset=["w4_llm_text", score_col])
        dropped_count = initial_count - len(df)

        if dropped_count > 0:
            print(
                f"[{theory_key}] Cleaned data: dropped {dropped_count} rows due to missing values."
            )

        # 2. TYPE CASTING: Ensure Likert scores are strict integers (1-5)
        # Bypasses float parsing bugs (e.g., 3.0 -> 3) which breaks token generation mappings
        df[score_col] = df[score_col].astype(int)

        # 3. STRUCTURE STRUCTURING: Convert to records format
        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    "id": str(row["ID"]),
                    "text": str(row["w4_llm_text"]).strip(),
                    "score": int(row[score_col]),
                }
            )

        # Store partition with meta-information for the downstream system prompts
        structured_data[theory_key] = {
            "theory_description": theory_metadata.get(theory_key, ""),
            "sample_size": len(records),
            "data": records,
        }

    # Save to JSON with UTF-8 encoding preserved for Czech diacritics
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=4)

    print(f"\nSuccessfully generated unified JSON file at: {output_json_path}")


# ==========================================
# Execution Wrapper
# ==========================================
if __name__ == "__main__":
    # Define your local file paths here
    input_files = {
        "hidden_elites": "D:/Programovani/VYZKUMAK/data/LLM_conspir_a_300.csv",
        "migration": "D:/Programovani/VYZKUMAK/data/LLM_conspir_b_300.csv",
        "vaccines": "D:/Programovani/VYZKUMAK/data/LLM_conspir_c_300.csv",
        "russia_nato": "D:/Programovani/VYZKUMAK/data/LLM_conspir_d_300.csv",
    }

    output_destination = "latent_psychometric_data.json"

    # Execute transformation
    prepare_psychometric_json(input_files, output_destination)