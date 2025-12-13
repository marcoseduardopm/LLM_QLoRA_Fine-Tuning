import pandas as pd
import numpy as np
import os
import json
import re
import torch

import random
import numpy as np
import argparse 

# --- 1. Argument Parsing ---
DEFAULT_SEED = 4315468 
DEFAULT_TARGET_FOLDERS = ('seed_1', 'seed_2', 'seed_3')

parser = argparse.ArgumentParser(description="Process raw evaluation data for multiple folders.")
parser.add_argument('--target_folders', nargs='+', default=list(DEFAULT_TARGET_FOLDERS),
                    help="List of target folder directories to pull results from (e.g., '0_12' '0_23').")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for reproducibility. Defaults to 4315468.")
args = parser.parse_args()

TARGET_FOLDERS = args.target_folders
SEED = args.seed

def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

# Set the seed at the start of the script
set_seed(SEED)

# Global constant for Ground Truth (same for all folders)
GROUND_TRUTH_FILE = 'datasets/skyrim_characters_all.csv'

# --- Main Processing Function ---
def process_evaluation_data(target_folder):
    """
    Reads the final evaluation CSV for a specific folder, processes the raw judge rankings,
    and compares factual fields against ground truth.
    """
    # Define paths dynamically based on the current target_folder
    input_file = f'results_{target_folder}/final_evaluation_all_configs.csv'
    output_file = f'results_{target_folder}/final_evaluation_all_configs_proc.csv'

    print(f"\n--- Processing folder: {target_folder} ---")

    # 1. Load the data
    try:
        df = pd.read_csv(input_file)
        gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure input files exist for folder '{target_folder}'. Skipping.")
        return

    # --- Prepare Ground Truth Data ---
    # We create a lookup dictionary for efficient access
    gt_df.drop_duplicates(subset=['name'], keep='first', inplace=True)
    gt_lookup = gt_df.set_index('name').to_dict('index')

    
    # Initialize new columns with default values
    df['rank_first_place'] = np.nan
    df['rank_second_place'] = np.nan
    df['rank_third_place'] = np.nan

    # Function to safely parse the JSON ranking and assign scores
    def parse_ranking(ranking_str):
        try:
            ranking = json.loads(ranking_str)
            if ranking and isinstance(ranking, list):
                first_place = ranking[0] if len(ranking) > 0 else np.nan
                second_place = ranking[1] if len(ranking) > 1 else np.nan
                third_place = ranking[2] if len(ranking) > 2 else np.nan
                return pd.Series([first_place, second_place, third_place])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass # Return default NaNs
        return pd.Series([np.nan, np.nan, np.nan])
    
    # Apply to the standard Gemini column
    if 'overall_ranking_for_char' in df.columns:
        df[['rank_first_place', 'rank_second_place', 'rank_third_place']] = df['overall_ranking_for_char'].apply(parse_ranking)
        print(f"[{target_folder}] Processed Gemini judge rankings (standard columns).")

    # --- Process GGUF Rankings (_2) ---
    df['rank_first_place_2'] = np.nan
    df['rank_second_place_2'] = np.nan
    df['rank_third_place_2'] = np.nan
    
    # Apply the *same* function to the _2 column
    if 'overall_ranking_for_char_2' in df.columns:
        df[['rank_first_place_2', 'rank_second_place_2', 'rank_third_place_2']] = df['overall_ranking_for_char_2'].apply(parse_ranking)
        print(f"[{target_folder}] Processed GGUF judge rankings (_2 columns).")

    # --- HYBRID LOGIC: Trust Judge, but Correction for Definite Matches & Unknowns ---
    for index, row in df.iterrows():
        character_name = row['character_name']
        gt_data = gt_lookup.get(character_name, {})

        factual_fields = ['home_city', 'race', 'gender', 'location']
        
        for field in factual_fields:
            gt_value = gt_data.get(field)
            model_value = row.get(field) 
            
            correct_col = f"{field}_correct"
            correct_col_2 = f"{field}_correct_2"

            calculated_result = None # Start with None (neutral - trust the judge)

            # 1. Check validity of Ground Truth
            #    If GT is missing/unknown, we cannot verify, so we leave the Judge's decision alone.
            is_gt_valid = pd.notna(gt_value) and str(gt_value).strip() != "" and str(gt_value).lower() != "unknown"

            if is_gt_valid:
                # 2. Check if Model Value is explicitly "Unknown"
                #    If GT exists but Model says "Unknown", it is definitely Incorrect.
                if pd.isna(model_value) or str(model_value).strip().lower() == "unknown":
                    calculated_result = "Incorrect"
                
                # 3. If Model Value exists, check for strict match
                elif str(model_value).strip() != "":
                    cleaned_gt = re.sub(r'[\'"]', '', str(gt_value)).lower()
                    cleaned_model = re.sub(r'[\'"]', '', str(model_value)).lower()
                    gt_words = set(cleaned_gt.split())

                    if any(word in cleaned_model for word in gt_words):
                        calculated_result = "Correct"

            # 4. Update Columns
            #    Only overwrite if we calculated a specific result (Correct Match or Definite Incorrect).
            #    Otherwise, the Judge's original "Incorrect" or "Correct" remains.
            if calculated_result:
                df.loc[index, correct_col] = calculated_result
                if correct_col_2 in df.columns:
                    df.loc[index, correct_col_2] = calculated_result
            
            # 5. Fallback: If column is still empty (Judge failed to output anything), default to Incorrect
            if pd.isna(df.loc[index, correct_col]):
                df.loc[index, correct_col] = "Incorrect"
            if correct_col_2 in df.columns and pd.isna(df.loc[index, correct_col_2]):
                df.loc[index, correct_col_2] = "Incorrect"

    print(f"[{target_folder}] Compared factual fields against ground truth.")

    # 4. Save the processed DataFrame
    df.to_csv(output_file, index=False)
    print(f"Processing complete for {target_folder}. Results saved to '{output_file}'.")

# --- Run the Script ---
if __name__ == "__main__":
    for folder in TARGET_FOLDERS:
        process_evaluation_data(folder)