import pandas as pd
import numpy as np
import os
import argparse
import re
import torch
import random
import glob

# --- 1. Argument Parsing ---
DEFAULT_SEED = 1003722 
DEFAULT_TARGET_FOLDERS = ('seed_1', 'seed_2', 'seed_3')
#DEFAULT_TARGET_FOLDER = ('0_12','0_12','0_12')
DEFAULT_EVALUATE_CF = False 
#DEFAULT_JUDGE_TYPE = 'gemini'  # 'gemini', 'local', or 'both'
DEFAULT_JUDGE_TYPE = 'both'  # 'gemini', 'local', or 'both'

parser = argparse.ArgumentParser(description="Generate LaTeX tables from processed evaluation data (Mean +- Std across runs).")
parser.add_argument('--target_folders', nargs='+', default=list(DEFAULT_TARGET_FOLDERS),
                    help="List of target folder directories to pull results from (e.g., '0_12' '0_06').")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for reproducibility.")
parser.add_argument('--judge_type', type=str, default=DEFAULT_JUDGE_TYPE, choices=['gemini', 'local', 'both'],
                    help="Which judge's results to process ('gemini', 'local', or 'both').")
parser.add_argument('--evaluate_cf', action='store_true',
                    help="If set, generates the Catastrophic Forgetting table instead of the main evaluation tables.")

args = parser.parse_args()

TARGET_FOLDERS = args.target_folders
if len(TARGET_FOLDERS) > 1:
    OUTPUT_BASE = "."
else:
    OUTPUT_BASE = f"results_{TARGET_FOLDERS[0]}"

JUDGE_TYPE = args.judge_type
#JUDGE_SUFFIX = f"_{JUDGE_TYPE}" 
SEED = args.seed
EVALUATE_CF = args.evaluate_cf if args.evaluate_cf else DEFAULT_EVALUATE_CF

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

set_seed(SEED)

def format_mean_std(mean_val, std_val, fmt="%.1f"):
    # Safety check for NaNs (e.g. if all values were 0 and got filtered out)
    if pd.isna(mean_val):
        return "-"
    if pd.isna(std_val) or std_val == 0:
        return f"{mean_val:{fmt[1:]}}"
    return f"{mean_val:{fmt[1:]}} \\pm {std_val:{fmt[1:]}}"

# --- Helper: Factual Score Converter ---
def get_factual_score(value):
    """Converts 'correct'/'incorrect' strings to 1.0/0.0 floats."""
    if pd.isna(value): return 0.5
    v = str(value).lower()
    if v == 'unknown': return np.nan
    return 1.0 if v in ['correct', 'true', '1', '1.0'] else 0.0 if v in ['incorrect', 'false', '0', '0.0'] else 0.5

def generate_cf_table():
    print(f"--- Generating Catastrophic Forgetting Table for folders: {TARGET_FOLDERS} ---")
    all_data = []
    pattern_ft = re.compile(r"TriviaQA_results_R(\d+)_([^_]+)_(.+)\.csv")
    pattern_base = re.compile(r"TriviaQA_results_R0_([^_]+)_base\.csv")

    for run_id, folder in enumerate(TARGET_FOLDERS):
        search_path = f'results_{folder}/TriviaQA_results_*.csv'
        files = glob.glob(search_path)
        if not files:
            continue
        for f in files:
            basename = os.path.basename(f)
            rank = None; tier = None; dataset_type = None
            match_base = pattern_base.match(basename)
            if match_base:
                rank = 0; tier = match_base.group(1); dataset_type = 'base' 
            else:
                match_ft = pattern_ft.match(basename)
                if match_ft:
                    rank = int(match_ft.group(1)); tier = match_ft.group(2); dataset_type = match_ft.group(3)
            if rank is not None:
                try:
                    df = pd.read_csv(f)
                    df['lora_rank'] = rank; df['dataset_type'] = dataset_type; df['run_id'] = run_id
                    if 'model_key' in df.columns: df.rename(columns={'model_key': 'model_name'}, inplace=True)
                    all_data.append(df)
                except Exception as e: print(f"Error reading {f}: {e}")
    
    if not all_data: return

    df = pd.concat(all_data, ignore_index=True)
    accuracy_df = df.groupby(['model_name', 'lora_rank', 'dataset_type', 'run_id'])['is_correct'].mean().reset_index()
    accuracy_df['accuracy_pct'] = accuracy_df['is_correct'] * 100

    base_rows = accuracy_df[accuracy_df['lora_rank'] == 0].copy()
    base_specific = base_rows[base_rows['dataset_type'] != 'base'][['model_name', 'run_id', 'dataset_type', 'accuracy_pct']]
    base_specific.rename(columns={'accuracy_pct': 'base_acc_specific'}, inplace=True)
    base_generic = base_rows[base_rows['dataset_type'] == 'base'][['model_name', 'run_id', 'accuracy_pct']]
    base_generic.rename(columns={'accuracy_pct': 'base_acc_generic'}, inplace=True)
    
    tuned_models = accuracy_df[accuracy_df['lora_rank'] > 0].copy()
    tuned_models = pd.merge(tuned_models, base_specific, on=['model_name', 'run_id', 'dataset_type'], how='left')
    tuned_models = pd.merge(tuned_models, base_generic, on=['model_name', 'run_id'], how='left')
    tuned_models['base_acc'] = tuned_models['base_acc_specific'].combine_first(tuned_models['base_acc_generic'])
    tuned_models.drop(columns=['base_acc_specific', 'base_acc_generic'], inplace=True)

    stats = tuned_models.groupby(['model_name', 'lora_rank', 'dataset_type'])[['base_acc', 'accuracy_pct']].agg(['mean', 'std'])
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    stats['ft_acc_str'] = stats['accuracy_pct_mean'].combine(stats['accuracy_pct_std'], format_mean_std)
    stats['base_acc_str'] = stats['base_acc_mean'].combine(stats['base_acc_std'], format_mean_std)

    base_df = stats[['model_name', 'dataset_type', 'base_acc_str']].drop_duplicates(subset=['model_name', 'dataset_type'])
    pivot_df = stats.pivot(index=['model_name', 'dataset_type'], columns='lora_rank', values='ft_acc_str').reset_index()
    final_df = pd.merge(base_df, pivot_df, on=['model_name', 'dataset_type'])

    # Mapping
    model_name_map = {
        'deepseek_7b': 'DeepSeek-7B', 'llama_8b': 'Llama-3.1-8B', 'qwen_8b': 'Qwen3-8B',
        'deepseek_16b': 'D-Coder-V2-16B', 'llama2_13b': 'Llama-2-13B', 'qwen_14b': 'Qwen3-14B',
        'deepseek_33b': 'D-Coder-33B', 'codellama_34b': 'CodeLlama-34B', 'qwen_32b': 'Qwen3-32B',
    }
    final_df['model_name'] = final_df['model_name'].replace(model_name_map)
    
    # Sorting
    model_sort_order = [model_name_map[k] for k in ['deepseek_7b', 'llama_8b', 'qwen_8b', 'deepseek_16b', 'llama2_13b', 'qwen_14b', 'deepseek_33b', 'codellama_34b', 'qwen_32b'] if k in model_name_map]
    datatype_sort_order = ['unstructured', 'structured', 'summarized']
    final_df['model_name'] = pd.Categorical(final_df['model_name'], categories=model_sort_order, ordered=True)
    final_df['dataset_type'] = pd.Categorical(final_df['dataset_type'], categories=datatype_sort_order, ordered=True)
    final_df = final_df.sort_values(by=['model_name', 'dataset_type'])

    final_df.rename(columns={'model_name': 'Model Name', 'dataset_type': 'Dataset', 'base_acc_str': 'Base Accuracy (%)', 16: 'Rank 16 Accuracy (%)', 32: 'Rank 32 Accuracy (%)', 64: 'Rank 64 Accuracy (%)'}, inplace=True)

    os.makedirs(f'{OUTPUT_BASE}/output', exist_ok=True)
    final_df.to_csv(f'{OUTPUT_BASE}/output/catastrophic_forgetting_results.csv', index=False)
    print(f"Saved CF Table.")

# --- Main Logic ---
if EVALUATE_CF:
    generate_cf_table()
else:
    print(f"--- Generating tables for folders: {TARGET_FOLDERS} ---")
    print(f"--- Using Judge: {JUDGE_TYPE.upper()} ---")
    
    os.makedirs(f'{OUTPUT_BASE}/output', exist_ok=True)

    # 1. Load Data
    all_dfs = []
    try:
        ground_truth_df = pd.read_csv('datasets/skyrim_characters_all.csv')
        ground_truth_df.drop_duplicates(subset=['name'], keep='first', inplace=True)
        gt_lookup = ground_truth_df.set_index('name').to_dict('index')
        print("Successfully loaded ground truth data.")

        for run_id, folder in enumerate(TARGET_FOLDERS):
            fpath = f'results_{folder}/final_evaluation_all_configs_proc.csv'
            try:
                tmp_df = pd.read_csv(fpath, low_memory=False)
                tmp_df['run_id'] = run_id
                all_dfs.append(tmp_df)
            except FileNotFoundError:
                print(f"Warning: File not found {fpath}")
        
        if not all_dfs:
            print("Error: No data loaded.")
            exit()
            
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined data: {len(df)} rows.")

        if 'generate_type' in df.columns:
            # Normalize strings to ensure consistent counting
            gen_counts = df['generate_type'].astype(str).str.lower().value_counts()
            
            det_count = gen_counts.get('deterministic', 0)
            non_det_count = gen_counts.get('non-deterministic', 0)
            total_gen = det_count + non_det_count

            print("\n--- Generation Type Distribution ---")
            print(f"Deterministic:     {det_count}")
            print(f"Non-Deterministic: {non_det_count}")
            if total_gen > 0:
                print(f"Ratio (Non-Det / Total): {non_det_count / total_gen:.2%}")
                if det_count > 0:
                    print(f"Ratio (Non-Det / Det):   {non_det_count / det_count:.2f}")
            print("------------------------------------\n")
        else:
            print("\n[Warning] 'generate_type' column not found in dataframe.\n")

        print("Applying 'Unknown' -> 'Incorrect' override based on Ground Truth...")

        # 1. Map Ground Truth fields to the main dataframe
        #    (Assumes GT keys in gt_lookup match these specific names)
        gt_field_map = {
            'home_city_correct': 'home_city',
            'race_correct': 'race',
            'gender_correct': 'gender',
            'location_correct': 'location'
        }

        for metric_base, gt_key in gt_field_map.items():
            # Create a temporary GT column for this metric
            gt_col_name = f'gt_temp_{gt_key}'
            df[gt_col_name] = df['character_name'].map(lambda x: gt_lookup.get(x, {}).get(gt_key))

            # Identify all judge columns related to this metric (e.g., 'home_city_correct', 'home_city_correct_2')
            relevant_cols = [c for c in df.columns if c.startswith(metric_base)]

            for col in relevant_cols:
                # Condition A: The judge result is 'Unknown' (case-insensitive)
                is_unknown = df[col].astype(str).str.lower().str.contains('unknown')
                
                # Condition B: The Ground Truth actually HAS an answer (not NaN and not empty)
                has_gt = df[gt_col_name].notna() & (df[gt_col_name] != '') & (~df[gt_col_name].astype(str).str.lower().str.contains('unknown'))

                # Apply Override: If Unknown AND GT Exists -> Incorrect
                mask = is_unknown & has_gt
                if mask.any():
                    count = mask.sum()
                    print(f"  > Overriding {count} 'Unknown' values in '{col}' to 'Incorrect' (GT found).")
                    df.loc[mask, col] = 'Incorrect'

            # Clean up temp column
            df.drop(columns=[gt_col_name], inplace=True)
        # =========================================================================

    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- 2. JUDGE PRE-PROCESSING (Score Calculation & Averaging) ---
    cols_map = {
        'home_city_correct': 'home_city_correct_2',
        'race_correct': 'race_correct_2',
        'gender_correct': 'gender_correct_2',
        'location_correct': 'location_correct_2',
        'overall_model_score': 'overall_model_score_2',
        'biography_quality_score': 'biography_quality_score_2',
    }

    print(f"Applying Judge Logic: {JUDGE_TYPE}...")
    if JUDGE_TYPE == 'local':
        # --- CASE: LOCAL ONLY ---
        print("Applying Judge Logic: local...")
        for std, gguf in cols_map.items():
            if gguf in df.columns:
                df[std] = df[gguf]
            else:
                print(f"Warning: {gguf} missing.")
        
        numeric_cols = ['overall_model_score', 'biography_quality_score','overall_model_score']
        for col in numeric_cols:
            if col in df.columns:
                # Force convert to numeric (Unknown -> NaN)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Discard rows that became NaN
                before_drop = len(df)
                df = df.dropna(subset=[col])
                dropped = before_drop - len(df)
                if dropped > 0:
                    print(f"  > Dropped {dropped} rows with invalid '{col}'.")

        df['rank_first_place_score'] = np.where(df['rank_first_place_2'] == df['model_key'], 1, 0)

    elif JUDGE_TYPE == 'gemini':
        # --- CASE: GEMINI ONLY ---
        print("Applying Judge Logic: gemini...")
        
        numeric_cols = ['overall_model_score', 'biography_quality_score','overall_model_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Discard rows that became NaN
                before_drop = len(df)
                df = df.dropna(subset=[col])
                dropped = before_drop - len(df)
                if dropped > 0:
                    print(f"  > Dropped {dropped} rows with invalid '{col}'.")

        df['rank_first_place_score'] = np.where(df['rank_first_place'] == df['model_key'], 1, 0)
    elif JUDGE_TYPE == 'both':
        # --- CASE: BOTH (AVERAGE WITH NORMALIZATION) ---
        print("Applying Judge Logic: both...")
        print("Normalizing and averaging scores between Gemini and Local judges...")
        
        # A. Factual & Boolean Columns (Strings to Scores 0.0-1.0)
        # These are already normalized (0 or 1), so we just average them.
        factual_cols = ['home_city_correct', 'race_correct', 'gender_correct', 'location_correct']
        
        cols_gemini = factual_cols
        cols_gguf = [f"{col}_2" for col in factual_cols]

        for col_1, col_2 in zip(cols_gemini, cols_gguf):
            if col_2 in df.columns:
                # 1. Convert strings to 1.0/0.0/NaN
                df[col_2] = df[col_2].apply(get_factual_score)
                
                # 2. Discard rows where the value is NaN
                before_drop = len(df)
                df = df.dropna(subset=[col_2])
                dropped_count = before_drop - len(df)
                if dropped_count > 0:
                    print(f"  > Dropped {dropped_count} rows with 'Unknown' values in '{col_2}'.")
                
                # 3. Handle col_1 (Gemini)
                df[col_1] = df[col_1].apply(get_factual_score)
                
                # 4. Average
                #avg_score = (df[col_1] + df[col_2]) / 2
                df[f'{col_1}_score'] = df[[col_1, col_2]].mean(axis=1)
                
                # 5. Assign to _score
                #df[f'{col_1}_score'] = avg_score

        # B. Numeric Quality Scores (Normalize Max then Average)
        numeric_cols = ['overall_model_score', 'biography_quality_score', 'overall_model_score']
        
        for col in numeric_cols:
            col_2 = f"{col}_2"
            
            if col in df.columns and col_2 in df.columns:
                # 1. Convert to numeric (coerce errors to NaN)
                #    We do NOT fillna(0) yet, because we don't want 0s to drag down the detected MAX
                #    if the scale is naturally 1-5.
                s1 = pd.to_numeric(df[col], errors='coerce').fillna(0)
                s2 = pd.to_numeric(df[col_2], errors='coerce').fillna(0)
                
                # 1. Determine the "Ceiling" for each judge
                #    We use the observed maximum, but force a minimum floor of 3.0
                #    to prevent scaling extreme garbage (e.g. if max score was 0.5) up to 5.0.
                obs_max_1 = s1.max()
                obs_max_2 = s2.max()

                # Safety: If a judge never gave a score above 1.0, treating 1.0 as "Perfect" (5/5) 
                # might be too aggressive. We set a floor (e.g., 3.0) to represent a "Minimum Acceptable Max".
                # If you want pure equal weighting regardless of absolute quality, set this to 1.0.
                calib_max_1 = max(obs_max_1, 3.0) 
                calib_max_2 = max(obs_max_2, 3.0)

                print(f"  > Calibrating '{col}': Judge 1 Max={obs_max_1:.2f} (Scale Factor={5/calib_max_1:.2f}), Judge 2 Max={obs_max_2:.2f} (Scale Factor={5/calib_max_2:.2f})")
                
                # 2. Normalize both to 0-1 scale relative to their own calibrated max
                s1_norm = s1 / calib_max_1
                s2_norm = s2 / calib_max_2
                
                # 3. Average the normalized scores (Equal Weight)
                avg_norm = (s1_norm + s2_norm) / 2
                
                # 4. Scale Final Result to 0-5
                df[col] = avg_norm * 5

        # C. Win Rate Score (Calculate Win per judge, then average)
        # This is binary (0 or 1), so no normalization needed.
        win_1 = np.where(df['rank_first_place'] == df['model_key'], 1, 0)
        win_2 = np.where(df['rank_first_place_2'] == df['model_key'], 1, 0)
        df['rank_first_place_score'] = (win_1 + win_2) / 2

    # --- 3. Additional Post-Processing ---
    # Convert 'unknown' to nan, clean up types
    # Note: If Judge Type was 'both', factual columns are already processed into '_score'. 
    # We only need to run this for 'gemini' or 'local' to generate the '_score' columns.
    if JUDGE_TYPE != 'both':
        for col in ['home_city_correct', 'race_correct', 'gender_correct', 'location_correct']:
            df[f'{col}_score'] = df[col].apply(get_factual_score)

    # Gender Ground Truth Logic
    # For 'both', we assume the generic correctness check (averaged above) is sufficient, 
    # but to support the GT logic strictly:
    df['gt_gender'] = df['character_name'].map(lambda x: gt_lookup.get(x, {}).get('gender'))
    
    # Composite Scores
    df['home_city_correct_score'] = df['home_city_correct_score'].astype(float)
    df['location_correct_score'] = df['location_correct_score'].astype(float)
    df['race_correct_score'] = df['race_correct_score'].astype(float)
    df['gender_correct_score'] = df['gender_correct_score'].astype(float)

    df['city_location_correct_score'] = df[['home_city_correct_score', 'location_correct_score']].max(axis=1)
    #df['city_location_correct_score'] = df[['home_city_correct_score', 'location_correct_score']].mean(axis=1)
    # =========================================================================
    # NEW LOGIC: Calculate Mean Factual Score and Scale to 0-5
    # =========================================================================
    df['avg_factual_score'] = df[['city_location_correct_score', 'race_correct_score', 'gender_correct_score']].mean(axis=1) * 5
    # =========================================================================

# =========================================================================
    # NEW LOGIC: Overwrite Overall Score with Mean(Factual, Bio)
    # =========================================================================
    # Ensure Bio score is on 0-5 scale to match avg_factual_score before averaging.
    # (If JUDGE_TYPE is 'both', bio score was normalized to 0-1, so we scale it up)
    temp_bio_score = df['biography_quality_score']
    #if JUDGE_TYPE == 'both':
    #    temp_bio_score = temp_bio_score * 5
        
    df['overall_model_score'] = (df['avg_factual_score'] + temp_bio_score) / 2
    # =========================================================================

    # =========================================================================
    # NEW LOGIC: Create Numeric Column for Generation Type Ratio
    # =========================================================================
    # We map 'non-deterministic' to 1.0 and everything else to 0.0.
    # The mean of this column becomes the "Ratio of Non-Deterministic" responses.
    if 'generate_type' in df.columns:
        df['is_non_deterministic'] = np.where(
            df['generate_type'].astype(str).str.lower() == 'non-deterministic', 1.0, 0.0
        )
    else:
        df['is_non_deterministic'] = np.nan
    # =========================================================================
    
    # =========================================================================
    # FIX: Robust Tier Conversion
    # =========================================================================
    # 1. Normalize the column to string, lowercase, and strip whitespace
    df['tier'] = df['tier'].astype(str).str.lower().str.strip()

    # 2. Use .map() instead of .replace(). 
    #    .map() converts anything NOT in the dictionary to NaN (which is what we want for safety).
    tier_map = {'8b': 8, '13b': 13, '33b': 33}
    df['tier_val'] = df['tier'].map(tier_map)

    # 3. Handle Invalid/Missing Tiers
    #    If we have NaNs (invalid tiers), we drop those rows so they don't break the int cast.
    missing_tiers = df['tier_val'].isna()
    if missing_tiers.any():
        print(f"\n[Warning] Found {missing_tiers.sum()} rows with invalid or missing 'tier' values. Dropping them to proceed.")
        df = df.dropna(subset=['tier_val'])

    # 4. Safely convert to integer now that we know all values are valid numbers
    df['tier_val'] = df['tier_val'].astype(int)
    # =========================================================================

    # --- 4. Mapping and Sorting ---
    model_name_map = {
        'deepseek_7b': 'DeepSeek-7B', 'llama_8b': 'Llama-3.1-8B', 'qwen_8b': 'Qwen3-8B',
        'deepseek_16b': 'D-Coder-V2-16B', 'llama2_13b': 'Llama-2-13B', 'qwen_14b': 'Qwen3-14B',
        'deepseek_33b': 'D-Coder-33B', 'codellama_34b': 'CodeLlama-34B', 'qwen_32b': 'Qwen3-32B',
    }
    df['model_key'] = df['model_key'].replace(model_name_map)

    # =========================================================================
    # NEW LOGIC: Fix Rank 0 labels and Sort Order
    # =========================================================================
    # 1. Replace NaN dataset_type for Rank 0 with 'Base'
    df.loc[df['lora_rank'] == 0, 'dataset_type'] = 'base'

    # 2. Define sort order with 'Base' first to ensure Rank 0 is at the top
    datatype_sort_order = ['base', 'unstructured', 'structured', 'summarized']
    # =========================================================================


    model_sort_order = [model_name_map[k] for k in ['deepseek_7b', 'llama_8b', 'qwen_8b', 'deepseek_16b', 'llama2_13b', 'qwen_14b', 'deepseek_33b', 'codellama_34b', 'qwen_32b'] if k in model_name_map]
    
    #datatype_sort_order = ['unstructured', 'summarized', 'structured']

    # --- 5. Aggregation Function ---
    def generate_aggregated_table(groupby_cols, filename_suffix, caption, pivot_win_rate=False, input_df=None):
        target_df = df if input_df is None else input_df
        target_df = target_df.copy() # Avoid SettingWithCopy warnings

        # --- FIX: Filter out 0 values for specific metrics to avoid skewing stats ---
        # Removing 0.0 for BLEU and ROUGE ensures we don't count failed/garbage generations
        # as a statistical zero, which creates high standard deviation.
        metrics_to_clean = ['bleu_score', 'rouge_score']
        for m in metrics_to_clean:
            if m in target_df.columns:
                target_df.loc[target_df[m] == 0, m] = np.nan
        # --------------------------------------------------------------------------

        metrics = {
            'perplexity': 'PPL',
            'rouge_score': 'ROUGE',
            'bleu_score': 'BLEU',
            'avg_factual_score': 'Factual Score',
            'biography_quality_score': 'Bio Quality',
            'overall_model_score': 'Overall Score',
            #'is_non_deterministic': 'Gen. Ratio'
        }
        per_run_agg = target_df.groupby(groupby_cols + ['run_id'])[list(metrics.keys())].mean().reset_index()
        final_agg = per_run_agg.groupby(groupby_cols)[list(metrics.keys())].agg(['mean', 'std']).reset_index()
        
        csv_df = pd.DataFrame()
        latex_df = pd.DataFrame()

        for col in groupby_cols:
            csv_df[col] = final_agg[col]
            latex_df[col] = final_agg[col]
            
        for metric_src, metric_dest in metrics.items():
            csv_df[f'{metric_dest}_Mean'] = final_agg[(metric_src, 'mean')]
            csv_df[f'{metric_dest}_Std'] = final_agg[(metric_src, 'std')]
            latex_df[metric_dest] = final_agg[(metric_src, 'mean')].combine(
                final_agg[(metric_src, 'std')], 
                lambda m, s: format_mean_std(m, s, "%.3f")
            )
            
        if 'model_key' in groupby_cols:
            for d in [csv_df, latex_df]:
                d['model_key'] = pd.Categorical(d['model_key'], categories=model_sort_order, ordered=True)
                d.rename(columns={'model_key': 'Model'}, inplace=True)
        if 'dataset_type' in groupby_cols:
            for d in [csv_df, latex_df]:
                d['dataset_type'] = pd.Categorical(d['dataset_type'], categories=datatype_sort_order, ordered=True)
                d.rename(columns={'dataset_type': 'Dataset'}, inplace=True)
        if 'lora_rank' in groupby_cols:
            for d in [csv_df, latex_df]: d.rename(columns={'lora_rank': 'Rank'}, inplace=True)
            
        sort_cols = [c for c in ['Model', 'Dataset', 'Rank'] if c in csv_df.columns]
        if sort_cols:
            csv_df.sort_values(by=sort_cols, inplace=True)
            latex_df.sort_values(by=sort_cols, inplace=True)

        # =========================================================================
        # FIX: Format Rank as Integer for LaTeX (removes decimal points like 16.0)
        # =========================================================================
        if 'Rank' in latex_df.columns:
            latex_df['Rank'] = latex_df['Rank'].apply(lambda x: str(int(x)) if pd.notnull(x) else "")
        # =========================================================================
        csv_df.to_csv(f'{OUTPUT_BASE}/output/{filename_suffix}.csv', index=False)
        print(f"Saved {filename_suffix}")

    # --- A. Overall Results (Split by Tier) ---
    print("Generating split tables for Overall Results (8B, 13B, 33B)...")
    for tier_val in [8, 13, 33]:
        tier_subset = df[df['tier_val'] == tier_val].copy()
        if tier_subset.empty: continue
        generate_aggregated_table(
            ['model_key', 'lora_rank', 'dataset_type'], 
            f'overall_results_{tier_val}B', 
            f"Overall Results ({tier_val}B Tier) (Mean \pm Std)",# - {JUDGE_TYPE.upper()}",
            input_df=tier_subset
        )
        
    # Define a filtered DataFrame excluding Rank 0 (Base models)
    df_no_rank0 = df[df['lora_rank'] != 0].copy()

    # --- B. Exp 1: Model Scale (Exclude Rank 0) ---
    # We use df_no_rank0 here
    generate_aggregated_table(
        ['model_key'], 
        f'exp1_model_scale', 
        f"Experiment 1: Impact of Model Scale (Averaged over Ranks > 0 and Datasets)",
        input_df=df_no_rank0 
    )

    # --- C. Exp 2: Data Formatting (Exclude Rank 0) ---
    # We use df_no_rank0 here
    generate_aggregated_table(
        ['model_key', 'dataset_type'], 
        f'exp2_data_formatting', 
        f"Experiment 2: Efficacy of Data Formatting (Averaged over Ranks > 0)",
        input_df=df_no_rank0
    )

    # --- D. Exp 3: LoRA Rank Analysis (INCLUDE Rank 0) ---
    # We use the original 'df' here because we want to see the base model performance
    generate_aggregated_table(
        ['model_key', 'lora_rank'], 
        f'exp3_lora_rank', 
        f"Experiment 3: LoRA Rank Analysis (Averaged over Datasets)",
        input_df=df # Explicitly using full df
    )

    # --- E. Exp 4: Win Rate (Exclude Rank 0) ---
    print(f"\n--- Generating Exp 4: Win Rate ({JUDGE_TYPE}) ---")
    def get_family(model_name):
        if 'DeepSeek' in model_name or 'D-Coder' in model_name: return 'DeepSeek'
        if 'Llama' in model_name: return 'Llama'
        if 'Qwen' in model_name: return 'Qwen'
        return 'Other'

    # Filter for Exp 4 calculation
    df_exp4 = df[df['lora_rank'] != 0].copy() # Exclude Rank 0
    df_exp4['Family'] = df_exp4['model_key'].apply(get_family)
    
    # Note: rank_first_place_score is already handled in pre-processing for 'both'
    # Use df_exp4 instead of df
    win_per_run = df_exp4.groupby(['tier', 'Family', 'run_id'])['rank_first_place_score'].mean().reset_index()
    win_stats = win_per_run.groupby(['tier', 'Family'])['rank_first_place_score'].agg(['mean', 'std']).reset_index()
    
    pivot_numeric = win_stats.pivot(index='tier', columns='Family', values='mean')
    pivot_numeric = pivot_numeric[['DeepSeek', 'Llama', 'Qwen']] if all(x in pivot_numeric.columns for x in ['DeepSeek', 'Llama', 'Qwen']) else pivot_numeric
    
    win_stats['fmt'] = win_stats['mean'].combine(win_stats['std'], lambda m, s: format_mean_std(m, s, "%.3f"))
    pivot_latex = win_stats.pivot(index='tier', columns='Family', values='fmt')
    pivot_latex = pivot_latex[['DeepSeek', 'Llama', 'Qwen']] if all(x in pivot_latex.columns for x in ['DeepSeek', 'Llama', 'Qwen']) else pivot_latex
    
    pivot_numeric.to_csv(f'{OUTPUT_BASE}/output/exp4_win_rate_pivot.csv')
    # =========================================================================
    # NEW LOGIC: Save Standard Deviation for Error Bars
    # =========================================================================
    pivot_std = win_stats.pivot(index='tier', columns='Family', values='std')
    # Ensure column order matches the mean table
    pivot_std = pivot_std[['DeepSeek', 'Llama', 'Qwen']] if all(x in pivot_std.columns for x in ['DeepSeek', 'Llama', 'Qwen']) else pivot_std
    # Save to a separate CSV
    pivot_std.to_csv(f'{OUTPUT_BASE}/output/exp4_win_rate_pivot_std.csv')
    print("Saved Exp 4 Win Rate STD table.")
    # =========================================================================