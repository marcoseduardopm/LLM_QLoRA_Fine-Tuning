# ===================================================================================
# Catastrophic Forgetting (CF) Evaluation Script
#
# This script evaluates base models and fine-tuned LoRA adapters on a general
# knowledge QA dataset (datasets/QA_1.csv) to measure knowledge retention.
# ===================================================================================

# --- 1. Imports ---
import os
import glob
import warnings
import argparse
import torch
import pandas as pd
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # <<< ADDED
from peft import PeftModel # <<< ADDED
from tqdm import tqdm
import re
import json
import dirtyjson
import random
import numpy as np
from num2words import num2words # pip install num2words --- NEW: --- For numerical answer comparison
from dateutil.parser import parse, ParserError #pip install python-dateutil

# --- 2. Default Configuration ---
# (Reusing relevant parameters from the original script)
DEFAULT_CUDA_VISIBLE_DEVICES = None
DEFAULT_CATALOG_NUMBER = 3
DEFAULT_TIER = "8b"
DEFAULT_IGNORE_MODEL = ""
DEFAULT_IGNORE_DATASET_TYPE = ""
DEFAULT_IGNORE_LORA_RANK = ""
DEFAULT_TARGET_FOLDER = "seed_1"
DEFAULT_SEED = 4315468
DEFAULT_LORA_PATH_NAME = ""
DEFAULT_ATTEMPT_FULL_BATCH = False
DEFAULT_USE_UNSLOTH = True # <<< ADDED

# --- 3. Argument Parsing ---
parser = argparse.ArgumentParser(description="Catastrophic Forgetting (CF) Evaluation Script for LoRA Models")
parser.add_argument("--cuda_devices", type=str, default=DEFAULT_CUDA_VISIBLE_DEVICES)
parser.add_argument("--lora_path_name", type=str, default=DEFAULT_LORA_PATH_NAME)
parser.add_argument("--catalog_number", type=int, default=DEFAULT_CATALOG_NUMBER)
parser.add_argument("--tier", type=str, default=DEFAULT_TIER)
parser.add_argument("--disable_unsloth", action="store_true", # <<< ADDED
                    help="If set, disables Unsloth and uses standard Transformers/Peft loading.")
parser.add_argument("--disable_quantization", action="store_true", 
                    help="If set, disables 4-bit quantization and loads the model in 16-bit precision. Defaults to False.")
parser.add_argument("--ignore_model", type=str, default=DEFAULT_IGNORE_MODEL,
                    help="Model(s) to ignore during evaluation. Separate multiple models with commas (,). Default is ''. If not provided, the script will not set a specific ignore model.")
parser.add_argument("--ignore_dataset_type", type=str, default=DEFAULT_IGNORE_DATASET_TYPE,
                    help="Dataset type(s) to ignore during evaluation. Separate multiple types with commas (,). Default is ''. If not provided, the script will not set a specific ignore dataset type.")
parser.add_argument("--ignore_lora_rank", type=str, default=DEFAULT_IGNORE_LORA_RANK,
                    help="LoRA rank(s) to ignore during evaluation. Separate multiple ranks with commas (,). Default is ''. If not provided, the script will not set a specific ignore LoRA rank.")
parser.add_argument("--force_regenerate_model", type=str, default="", 
                    help="Specify a model key (e.g., 'llama_8b') to force regeneration of its results, deleting existing entries for it.")
parser.add_argument("--recalculate_correctness", action='store_true', 
                    help="Set this flag to re-evaluate the 'is_correct' column for all existing answers based on the current logic.")
parser.add_argument("--target_folder", type=str, default=DEFAULT_TARGET_FOLDER)
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
args = parser.parse_args()

# --- 4. Apply Arguments and Set Up Environment ---
# Apply parsed arguments
if args.cuda_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    # If not provided, ensure it's not set by the script if it was previously
    if DEFAULT_CUDA_VISIBLE_DEVICES is None and "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

TIER = args.tier.lower().replace(" ", "_")
TARGET_FOLDER = args.target_folder
SEED = args.seed
ignore_model = str(args.ignore_model).split(",") if args.ignore_model else []
ignore_dataset_type = str(args.ignore_dataset_type).split(",") if args.ignore_dataset_type else []
ignore_lora_rank = str(args.ignore_lora_rank).split(",") if args.ignore_lora_rank else []
catalog_number = args.catalog_number
lora_path_name = args.lora_path_name
USE_UNSLOTH = not args.disable_unsloth if DEFAULT_USE_UNSLOTH else False # <<< ADDED
DISABLE_QUANTIZATION = args.disable_quantization

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

set_seed(SEED)

# --- 5. Constants and Catalogs ---
INPUT_CF_QUESTIONS_CSV = f"datasets/QA_{str(catalog_number)}.csv"
LORA_RANKS_TO_EVALUATE = [0, 16, 32, 64]
DATASET_TYPES_TO_EVALUATE = ["structured", "unstructured", "summarized"]
MODEL_CATALOG = {}
if TIER == "8b" or TIER == "all":
    MODEL_CATALOG.update({
        # ~8B Tier
        "llama_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "qwen_8b": "Qwen/Qwen3-8B",
        "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    })
if TIER == "13b" or TIER == "all":
    MODEL_CATALOG.update({
        # ~13B Tier
        "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
        "qwen_14b": "Qwen/Qwen3-14B",
        "deepseek_16b": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    })
if TIER == "33b" or TIER == "all":
    MODEL_CATALOG.update({
        # ~32B Tier
        "codellama_34b": "codellama/CodeLlama-34b-Instruct-hf",
        "qwen_32b": "Qwen/Qwen3-32B",
        "deepseek_33b": "deepseek-ai/deepseek-coder-33b-instruct",
    })

# --- 6. Helper Functions ---
def clean_chain_of_thought(text: str) -> str:
    """
    Removes <think>...</think> blocks from the text.
    Returns the cleaned text stripped of whitespace.
    """
    # 1. Handle non-string inputs (NaN, None, float)
    if text is None:
        return ""
    if not isinstance(text, str):
        # Check if it's a NaN (float)
        try:
            if np.isnan(text):
                return ""
        except:
            pass

    # Convert to string for processing
    text_str = str(text)

    # FIX: Only check for "nan" (pandas artifact). 
    # REMOVED 'none' from this check to allow "None" as a valid answer.
    if text_str.strip().lower() == 'nan':
        return ""
    
    

    # Snapshot for debug comparison
    original_text = text_str

    # 2. Remove content between <think> tags (handling unclosed tags at end of text)
    # This regex removes everything from <think> to </think>, including newlines.

    #cleaned = re.sub(r'<think>.*?(?:</think>|$)', '', text_str, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    #Executed temporary, should not be needed anymore
    cleaned = cleaned.replace('(empty answer)', '')

    # 3. Cleanup residual tags
    cleaned = cleaned.replace('<think>', '').replace('</think>', '')

    # 4. Remove "Sure! ... :" prefixes
    cleaned = re.sub(r'^Sure!.*?:', '', cleaned.strip(), flags=re.IGNORECASE | re.DOTALL)
    
    # 5. Remove "*Answer*:" prefixes
    # Matches starting with one or more asterisks, "answer", one or more asterisks, then a colon
    # Example: "**Answer**: 5" -> "5" | "*answer*: 5" -> "5"
    cleaned = re.sub(r'^\*+answer\*+:\s*', '', cleaned.strip(), flags=re.IGNORECASE)

    # 6. Remove "Okay/Alright" fillers if the answer is short
    # This catches "Alright, let's tackle this..." or "Okay, the answer is..."
    # We define "short" as less than ~25 words to avoid deleting long valid explanations that happen to start with "Okay".
    temp_clean = cleaned.strip()
    if len(temp_clean.split()) < 25:
        # Regex to match starting with Okay or Alright, followed by punctuation or space
        if re.match(r'^(Okay|Alright)(?:,|!|\.|\s)', temp_clean, flags=re.IGNORECASE):
            cleaned = ""

    # --- DEBUG VERBOSE: Print only if changes occurred ---
    if original_text.strip() != cleaned:
        print(f"[Cleaning Debug] Original: {repr(original_text)}  ->  Cleaned: {repr(cleaned)}")

    return cleaned.strip()

def _parse_json_fallback(response_text: str) -> dict | None:
    """
    Robustly parses a JSON object from a model's response string.
    Handles markdown code blocks and uses dirtyjson for flexibility.
    """
    try:
        # Attempt 1: Look for a JSON code block
        match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Attempt 2: Find the first '{' and last '}'
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return None # No JSON object found
            json_str = response_text[json_start:json_end]
        
        # Use dirtyjson to handle trailing commas, etc.
        return dirtyjson.loads(json_str)
    except (json.JSONDecodeError, dirtyjson.Error, Exception):
        return None

def _parse_key_value_fallback(response_text: str) -> dict | None:
    """
    (Adapted from original script)
    Parses plain text "Key: Value" responses when JSON parsing fails.
    """
    pattern = re.compile(r"^\s*([\w\s'’]+?)\s*:\s*(.*?)(\n|$)", re.MULTILINE)
    matches = pattern.findall(response_text)
    if not matches:
        return None
    parsed_data = {}
    for key, value, _ in matches:
        normalized_key = key.strip().lower()
        if 'answers' in normalized_key:
            try:
                answers_list = dirtyjson.loads(value.strip())
                if isinstance(answers_list, list):
                    parsed_data['answers'] = answers_list
                    return parsed_data
            except:
                parsed_data['answers'] = [item.strip() for item in value.strip().split(',')]
                return parsed_data
    return None

def _parse_greedy_key_value_fallback(response_text: str) -> dict | None:
    """
    (Adapted from original script)
    Parses a string where keys and values are separated by a colon.
    """
    pattern = re.compile(r"(\w+)\s*:\s*(.*?)(?=\s*\w+\s*:|\Z)", re.DOTALL)
    matches = pattern.finditer(response_text)
    parsed_data = {}
    for match in matches:
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if 'answers' in key:
            try:
                answers_list = dirtyjson.loads(value)
                if isinstance(answers_list, list):
                    parsed_data['answers'] = answers_list
                    return parsed_data
            except:
                if '\n' in value:
                    parsed_data['answers'] = [item.strip() for item in value.split('\n')]
                else:
                    parsed_data['answers'] = [item.strip() for item in value.split(',')]
                return parsed_data
    return None

def normalize_text(text: str) -> str:
    """
    Standard text normalization: lowercasing, stripping, and handling numbers.
    Assumes your existing number conversion logic is here (e.g., "four" -> "4").
    """
    if not isinstance(text, str):
        return str(text).strip().lower()
        
    text = text.strip().lower()
    
    # 1. Clean punctuation/articles
    text = re.sub(r"[\.,!?'\"()]", '', text)
    text = re.sub(r"\b(a|an|the|of|in|and|is|was|were|are)\b", '', text).strip()

    # 2. Standardize spacing
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = re.sub(r'\bspouse\b', 'wife', text)
    text = re.sub(r'\bpartner\b', 'wife', text)
    
    return text

def normalize_answer_date(answer: str) -> str | None:
    """
    Attempts to parse a string as a date and returns it in a standardized
    format (YYYY-MM-DD). Returns None if parsing fails.
    """
    if not answer:
        return None
    
    try:
        # Use dateutil.parser for flexible parsing of formats like "14-Jul", "July 15th"
        # fuzzy=True allows it to ignore extraneous characters like 'th', 'rd', etc.
        date_obj = parse(answer, fuzzy=True, dayfirst=False)
        
        # Return a standard, unambiguous format for comparison
        return date_obj.strftime('%Y-%m-%d')
    except (ParserError, ValueError, OverflowError):
        # If date parsing fails completely (e.g., "Jarl Balgruuf"), return None
        return None
    
def is_answer_correct(generated_answer: str, ground_truth: str) -> int:
    """
    Checks if the generated answer is correct using a tiered normalization strategy.
    """
    # Ensure inputs are strings
    gt = str(ground_truth)
    ga = str(generated_answer)

    if not ga.strip():
        return 0
    # ------------------- TIER 1: Strict Match -------------------
    if gt.lower() == ga.lower():
        return 1
    
    # ------------------- TIER 2: Basic Text Normalization -------------------
    norm_gt = normalize_text(gt)
    norm_ga = normalize_text(ga)
    
    if norm_gt == norm_ga:
        return 1

    # ------------------- TIER 3: Date Normalization (Fallback) -------------------
    date_gt = normalize_answer_date(gt)
    date_ga = normalize_answer_date(ga)
    
    if date_gt is not None and date_ga is not None and date_gt == date_ga:
        return 1
    # ------------------- TIER 4: Numeric & Mixed Logic -------------------
    is_both_numeric = False
    try:
        gen_num = float(norm_ga)
        truth_num = float(norm_gt)
        is_both_numeric = True
    except (ValueError, TypeError):
        is_both_numeric = False

    if is_both_numeric:
        # A. Strict Numeric Comparison
        if gen_num == truth_num:
            return 1
    else:
        try:
            # B. Mixed Type Comparison
            # Check if one is a number and the other is text representation
            if num2words:
                # Helper to safely convert number to words
                def safe_num2words(val):
                    # Convert 87.0 -> 87 to avoid "eighty-seven point zero"
                    if val.is_integer():
                        val = int(val)
                    return num2words(val).replace('-', ' ').replace(',', '').lower()

                try:
                    # Case 1: GA is number (87), GT is text ("Eighty-seven")
                    gen_val = float(norm_ga)
                    gen_word = safe_num2words(gen_val)
                    # FIX: Replace hyphens in the text target too
                    if gen_word in norm_gt.replace('-', ' '):
                        return 1
                except (ValueError, TypeError):
                    pass

                try:
                    # Case 2: GT is number (87), GA is text ("Eighty-seven")
                    truth_val = float(norm_gt)
                    truth_word = safe_num2words(truth_val)
                    # FIX: Replace hyphens in the text target too
                    if truth_word in norm_ga.replace('-', ' '):
                        return 1
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass
        # C. Substring Comparison
        if norm_ga and (norm_ga in norm_gt or norm_gt in norm_ga):
            return 1
        
        # D. Suffix standardization
        tmp_ga = norm_ga.replace('ism','ist')
        tmp_gt = norm_gt.replace('ism','ist')
        if tmp_ga in tmp_gt or tmp_gt in tmp_ga:
            return 1
        
        # ==============================================================================
        # E. Bag of Words / Token Matching (NEW)
        # ==============================================================================
        # This handles "green and white" vs "White and green".
        # Since normalize_text removes "and", we just check if {white, green} matches.
        gt_tokens = set(norm_gt.split())
        ga_tokens = set(norm_ga.split())
        
        # We ensure gt_tokens is not empty to avoid matching empty strings as correct
        if len(gt_tokens) > 0 and gt_tokens.issubset(ga_tokens):
            return 1

    return 0

# --- 7. Main Evaluation Function ---
def run_cf_evaluation():
    """
    Main function to loop through models and adapters and run CF evaluation.
    """
    if not os.path.exists(INPUT_CF_QUESTIONS_CSV):
        print(f"Error: QA dataset not found at '{INPUT_CF_QUESTIONS_CSV}'")
        return

    qa_df = pd.read_csv(INPUT_CF_QUESTIONS_CSV)
    questions_list = qa_df['question'].tolist()
    ground_truth_answers = qa_df['answer'].tolist()
    
    print(f"Loaded {len(questions_list)} questions for CF evaluation.")

    for lora_rank in LORA_RANKS_TO_EVALUATE:
        if str(lora_rank) in ignore_lora_rank:
            print(f"Skipping LoRA rank '{lora_rank}' as it is set to be ignored.")
            continue
            
        current_dataset_types = ["base"] if lora_rank == 0 else DATASET_TYPES_TO_EVALUATE

        for dataset_type in current_dataset_types:

            print(f"\n--- Starting inference for LoRA Rank: {lora_rank}, Dataset Type: {dataset_type} ---")

            if dataset_type in ignore_dataset_type:
                print(f"Skipping dataset type '{dataset_type}' as it is set to be ignored.")
                continue

            # This is the new dynamic output file name
            OUTPUT_RESULTS_CSV = f"results_{TARGET_FOLDER}/TriviaQA_results_R{lora_rank}_{TIER}_{dataset_type}.csv"
            
            # Check for and load existing results to resume if possible
            if os.path.exists(OUTPUT_RESULTS_CSV):
                print(f"Found existing results at '{OUTPUT_RESULTS_CSV}'. Resuming inference.")
                results_df = pd.read_csv(OUTPUT_RESULTS_CSV, keep_default_na=False)

                # --- NEW: Clean and Save Immediately ---
                # Apply cleaning to the entire column. This updates existing rows that don't need regeneration.
                # We fill NaNs with "" first to ensure the function receives strings.
                results_df['generated_answer'] = results_df['generated_answer'].fillna("").apply(clean_chain_of_thought)
                
                # ==============================================================================
                # <<< --- NEW LOGIC: Force Regenerate Specific Model --- >>>
                # ==============================================================================
                if args.force_regenerate_model:
                    model_to_purge = args.force_regenerate_model
                    initial_len = len(results_df)
                    
                    # Remove rows where model_key matches the flag
                    results_df = results_df[results_df['model_key'] != model_to_purge]
                    
                    removed_count = initial_len - len(results_df)
                    if removed_count > 0:
                        print(f"Force regeneration active: Removed {removed_count} existing rows for model '{model_to_purge}'.")
                        # Save immediately so the purge persists even if script crashes
                        results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
                    else:
                        print(f"Force regeneration active: No existing rows found for model '{model_to_purge}'.")
                # ==============================================================================

                # ==============================================================================
                # <<< --- NEW LOGIC: Recalculate Correctness if Flag is Set --- >>>
                # ==============================================================================
                if args.recalculate_correctness:
                    print(f"Flag detected: Recalculating correctness scores for {len(results_df)} rows in '{OUTPUT_RESULTS_CSV}'...")
                    
                    # Apply the updated is_answer_correct function to every row
                    results_df['is_correct'] = results_df.apply(
                        lambda row: is_answer_correct(row['generated_answer'], row['ground_truth_answer']), 
                        axis=1
                    )
                    
                    # Save immediately so the file is updated even if the script stops later
                    results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
                    print("Correctness recalculation complete and saved.")
                # ==============================================================================

                # Save the cleaned state immediately. 
                # This ensures valid answers are stripped of tags in the file, even if we skip this model.
                results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
                print(f"Cleaned existing data and saved to '{OUTPUT_RESULTS_CSV}'.")
                # ---------------------------------------
                
                # Create a set of (character_name, model_key) tuples for fast lookup of completed tasks
                completed_tasks = set(tuple(row) for row in results_df[['question_text', 'model_key']].values)
            else:
                print(f"No existing results found at '{OUTPUT_RESULTS_CSV}'. Starting from scratch.")
                results_df = pd.DataFrame(columns=['model_key', 'question_id', 'question_text', 'ground_truth_answer', 'generated_answer', 'is_correct'])         
                completed_tasks = set()

            for model_key, model_id in MODEL_CATALOG.items():
                if model_key in ignore_model:
                    print(f"Skipping model '{model_key}' as it is set to be ignored.")
                    continue
# A temporary list to hold results for the CURRENT model before saving
                new_results_for_model = []

                # --- NEW LOGIC: Check for existing VALID answers ---
                # 1. Identify which questions have a VALID (non-empty) answer already saved
                valid_processed_questions = set()
                if not results_df.empty:
                    current_model_results = results_df[results_df['model_key'] == model_key]
                    for _, row in current_model_results.iterrows():
                        # Clean the answer using the NEW logic (removes <think> AND "Sure!...")
                        # If the result is empty, it is NOT valid and will be re-processed.
                        if clean_chain_of_thought(str(row['generated_answer'])): 
                            valid_processed_questions.add(row['question_text'])

                # 2. Determine which questions from the original dataframe need processing
                questions_to_process_indices = []
                all_questions_ref = qa_df['question'].tolist()
                
                for idx, q in enumerate(all_questions_ref):
                    if q not in valid_processed_questions:
                        questions_to_process_indices.append(idx)

                # 3. Decision: Skip or Process Subset
                if not questions_to_process_indices:
                    print(f"All questions for model '{model_key}' already processed and validated. Skipping.")
                    continue
                
                # 4. Filter the lists to only run inference on the missing/empty ones
                questions_list = qa_df.iloc[questions_to_process_indices]['question'].tolist()
                ground_truth_answers = qa_df.iloc[questions_to_process_indices]['answer'].tolist()
                
                print(f"Model '{model_key}': Processing {len(questions_list)} questions ({len(all_questions_ref) - len(questions_list)} skipped).")
                # --- END NEW LOGIC ---

                use_cache = True
                print(f"\n--- Starting CF eval for Model: {model_key}, Rank: {lora_rank}, Type: {dataset_type} ---")


                lora_path = None
                if lora_rank > 0:
                    # if no lora_path_name is provided, search for the lastest checkpoint
                    if lora_path_name == "":
                        lora_dir = f"results_{TARGET_FOLDER}/R{lora_rank}_{model_key}_{dataset_type}"
                        # Otherwise, look for the latest checkpoint within the determined training_output_dir.
                        checkpoints = list(glob.glob(os.path.join(lora_dir, "checkpoint-*")))
                        if len(checkpoints) > 0:
                            checkpoints.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
                            lora_path = checkpoints[-1]
                        else:
                            print(f"Warning: No checkpoints found in '{lora_dir}' for '{model_key}' with rank {lora_rank} and dataset '{dataset_type}'. Skipping.")
                            continue
                    else:
                        # Dynamically construct the path to the LoRA adapter
                        lora_path = f"results_{TARGET_FOLDER}/R{lora_rank}_{model_key}_{dataset_type}/{lora_path_name}"

                MUST_AVOID_UNSLOTH = False # Example models that Unsloth may not support well
                if model_key == "deepseek_33b":
                    print(f"Model '{model_key}' is known to have issues with Unsloth. Disabling Unsloth for this model.")
                    MUST_AVOID_UNSLOTH
                # ========================= MODIFIED LOADING BLOCK =========================
                if USE_UNSLOTH and not MUST_AVOID_UNSLOTH:
                    # Load the base model and tokenizer using Unsloth
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=model_id,
                        device_map="auto",
                        max_seq_length=1024,
                        dtype=None,
                        load_in_4bit= not DISABLE_QUANTIZATION,
                        trust_remote_code=True,
                    )
                    # Conditionally load the LoRA adapter
                    if lora_rank > 0 and lora_path:
                        model.load_adapter(lora_path) 
                    
                    # Optimize for inference
                    FastLanguageModel.for_inference(model)
                else:
                    print("Unsloth disabled. Using standard Transformers/Peft loading...")
                    if not DISABLE_QUANTIZATION:
                        print("Loading with 4-bit quantization (BitsAndBytes)...")
                        # Define quantization config for inference
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                    else:
                        print("Quantization disabled. Loading in full/half precision...")
                        quantization_config = None # <<< CHANGED: No config if disabled
                    
                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    
                    # Load base model
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    
                    # Load the LoRA adapter onto the base model
                    if lora_rank > 0 and lora_path:
                        model = PeftModel.from_pretrained(model, lora_path)
                # ==========================================================================

        
                chat_template = None
                if "coder" in model_id.lower():
                    chat_template = (
                        "{% for message in messages %}"
                        "{% if message['role'] == 'user' %}"
                        "{{ '### Instruction:\\n' + message['content'] + '\\n### Response:\\n' }}"
                        "{% elif message['role'] == 'assistant' %}"
                        "{{ message['content'] + '<|EOT|>' }}"
                        "{% endif %}"
                        "{% endfor %}"
                    )
                elif "deepseek" in model_id.lower():
                    chat_template = (
                        "{% for message in messages %}"
                        "{% if message['role'] == 'user' %}"
                        "{{ 'User: ' + message['content'] + '\\n' }}"
                        "{% elif message['role'] == 'assistant' %}"
                        "{{ 'Assistant: ' + message['content'] + '<|end_of_sentence|>' }}"
                        "{% endif %}"
                        "{% endfor %}"
                    )
                # 3. Llama 
                elif "llama" in model_id.lower():
                    # Llama 3 uses strict special tokens. 
                    # If the tokenizer doesn't load it automatically, this ensures it works.
                    chat_template = (
                        "{% set loop_messages = messages %}"
                        "{% for message in loop_messages %}"
                        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                        "{% if loop.index0 == 0 %}"
                        "{% set content = '<|begin_of_text|>' + content %}"
                        "{% endif %}"
                        "{{ content }}"
                        "{% endfor %}"
                        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                    )

                # 4. Qwen
                elif "qwen" in model_id.lower():
                    chat_template = (
                        "{% for message in messages %}"
                        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                        "{% endfor %}"
                        "{% if add_generation_prompt %}"
                        "{{ '<|im_start|>assistant\n' }}"
                        "{% endif %}"
                    )
                if tokenizer.chat_template is None:
                    tokenizer.chat_template = chat_template

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                generated_answers = []

                # Initialize token limit for this model (starts small for speed)
                current_max_tokens = 768
                
                for question in tqdm(questions_list, desc="Individual Fallback"):
                    prompt = f"Question: {question}\nIf Unknown, just reply 'Unknown.'. Otherwise answer concisely, ideally a single word without explanation:"
                    messages = [{"role": "user", "content": prompt}]
                    
                    prompt_string = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    inputs = tokenizer(prompt_string, return_tensors="pt").to(model.device)
                    
                    # --- RETRY LOGIC WITH STICKY TOKEN LIMIT ---
                    final_answer = "(empty)"
                    attempts = 0
                    max_attempts = 2 # Attempt 1 (Standard) + Attempt 2 (Retry with more tokens)

                    while attempts < max_attempts:
                        try:
                            outputs = model.generate(
                                **inputs, 
                                max_new_tokens=current_max_tokens, 
                                use_cache=use_cache, 
                                do_sample=False, 
                                eos_token_id=tokenizer.eos_token_id,
                                num_beams=1
                            )
                        except RuntimeError:
                            warnings.warn(f"RuntimeError during generation. Trying with use_cache=False.")
                            torch.cuda.empty_cache()
                            use_cache = False
                            outputs = model.generate(
                                **inputs, 
                                max_new_tokens=current_max_tokens, 
                                use_cache=False, 
                                do_sample=False,
                                eos_token_id=tokenizer.eos_token_id,
                                num_beams=1
                            )
                        
                        answer_text = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
                        
                        # Clean and check
                        cleaned = clean_chain_of_thought(answer_text)
                        
                        if cleaned:
                            # Success!
                            final_answer = cleaned
                            break
                        else:
                            # Failure: Answer was empty
                            attempts += 1
                            if attempts < max_attempts and current_max_tokens < 768:
                                print(f"  > Empty answer detected. Increasing max_tokens to 768 for this model and retrying...")
                                current_max_tokens = 768 
                                # The 'current_max_tokens' variable is now updated for THIS question 
                                # AND all future questions for this model.
                            else:
                                # Max attempts reached, stick with "(empty)"
                                final_answer = "(empty)"
                                break

                    generated_answers.append(final_answer)
                for i, question in enumerate(questions_list):
                    # --- MODIFIED: --- Use the new advanced correctness checker
                    is_correct = is_answer_correct(generated_answers[i], ground_truth_answers[i])
                    new_results_for_model.append({
                        "model_key": model_key,
                        "question_id": i,
                        "question_text": question,
                        "ground_truth_answer": ground_truth_answers[i],
                        "generated_answer": generated_answers[i],
                        "is_correct": is_correct
                    })

                # --- NEW: Place saving logic here, after the character loop completes ---
                if new_results_for_model:
                    new_results_df = pd.DataFrame(new_results_for_model)
                    results_df = pd.concat([results_df, new_results_df], ignore_index=True)
                    
                    # ========================= START OF FIX =========================
                    # Explicitly remove duplicates based on the unique key combination.
                    # This ensures that even if a task was re-run, only the latest entry is kept.
                    results_df.drop_duplicates(subset=['question_text', 'model_key'], keep='last', inplace=True)
                    # ========================== END OF FIX ==========================

                    results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
                    print(f"Incremental save complete. Results for model '{model_key}' appended to '{OUTPUT_RESULTS_CSV}'.")
                else:
                    print(f"No new results generated for model '{model_key}'.")
            

                # Clean up GPU memory before loading the next model
                if 'model' in locals() and model is not None:
                    del model
                if 'tokenizer' in locals() and tokenizer is not None:
                    del tokenizer
                torch.cuda.empty_cache()

    print("\nCatastrophic forgetting evaluation complete.")

if __name__ == "__main__":
    run_cf_evaluation()

