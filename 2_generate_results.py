# ===================================================================================
# Batch Inference Script for Skyrim Model Evaluation
#
# This script systematically loads each fine-tuned LoRA adapter, runs inference
# on a list of characters, and saves all generated responses to a single CSV
# file for easy comparison and evaluation. It now supports incremental saving.
# ===================================================================================

import os
import glob
# --- 2. Import Necessary Libraries ---
import warnings
import argparse
import wandb  # Optional: For Weights & Biases integration
import re


#Default values
DEFAULT_CUDA_VISIBLE_DEVICES = None#"0"#,1,2" #Values: None, "0,1,2", "0", "1", "2"
DEFAULT_DATASET_NAME = "skyrim_characters_all.csv" #Values: None, "0,1,2", "0", "1", "2"
DEFAULT_LORA_PATH_NAME = "" #Values: None, "final_0_05", "final_0_1", "final_0_2"
DEFAULT_TIER = "8b" #"all"
MAX_RETRIES = 3 #so that we try both deterministic (json and other formats) and non-deterministic generation if needed
DEFAULT_IGNORE_MODEL = ""#"deepseek_7b,llama_8b" #
DEFAULT_IGNORE_DATASET_TYPE = "" #Values: "", "structured", "unstructured", "summarized"
DEFAULT_IGNORE_LORA_RANK = "" #Values: "0", "16", "32", "64"
DEFAULT_TARGET_FOLDER = "0_01"#"0_06, 0_01, 0_006 
DEFAULT_SEED = 4315468 # Define the default seed
DEFAULT_USE_UNSLOTH = True # <<< ADDED

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Advanced LoRa Fine-Tuning Script for Skyrim Character Dataset - Evaluation Phase")

# Optional arguments with default values
parser.add_argument("--cuda_devices", type=str, default=DEFAULT_CUDA_VISIBLE_DEVICES,
                    help="Comma-separated list of CUDA device IDs to make visible (e.g., '0,1,2'). If not provided, os.environ['CUDA_VISIBLE_DEVICES'] will not be set by the script.")
parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME,
                    help="Name of the dataset to use for evaluation. Default is 'skyrim_characters_all_example.csv'. If not provided, the script will not set a specific dataset name.")
parser.add_argument("--lora_path_name", type=str, default=DEFAULT_LORA_PATH_NAME,
                    help="Name of the LoRA path to use for evaluation. Default is ''. If not provided, the script will not set a specific LoRA path name.")
parser.add_argument("--tier", type=str, default=DEFAULT_TIER,
                    help="Model tier to use for evaluation. Default is 'all'. If not provided, the script will not set a specific tier.")
parser.add_argument("--ignore_model", type=str, default=DEFAULT_IGNORE_MODEL,
                    help="Model(s) to ignore during evaluation. Separate multiple models with commas (,). Default is ''. If not provided, the script will not set a specific ignore model.")
parser.add_argument("--ignore_dataset_type", type=str, default=DEFAULT_IGNORE_DATASET_TYPE,
                    help="Dataset type(s) to ignore during evaluation. Separate multiple types with commas (,). Default is ''. If not provided, the script will not set a specific ignore dataset type.")
parser.add_argument("--force_regenerate_model", type=str, default="", 
                    help="Specify a model key (e.g., 'llama_8b') to force regeneration of its results, deleting existing entries for it.")
parser.add_argument("--ignore_lora_rank", type=str, default=DEFAULT_IGNORE_LORA_RANK,
                    help="LoRA rank(s) to ignore during evaluation. Separate multiple ranks with commas (,). Default is ''. If not provided, the script will not set a specific ignore LoRA rank.")
parser.add_argument("--disable_unsloth", action="store_true", # <<< ADDED
                    help="If set, disables Unsloth and uses standard Transformers/Peft loading.")
parser.add_argument("--disable_quantization", action="store_true", 
                    help="If set, disables 4-bit quantization and loads the model in 16-bit precision. Defaults to False.")
parser.add_argument("--target_folder", type=str, default=DEFAULT_TARGET_FOLDER,
                    help="Target loss for training.")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for reproducibility. Defaults to 4315468.")
args = parser.parse_args()

# Apply parsed arguments
if args.cuda_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    # If not provided, ensure it's not set by the script if it was previously
    if DEFAULT_CUDA_VISIBLE_DEVICES is None and "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

DATASET_NAME = args.dataset_name
lora_path_name = args.lora_path_name.lower().replace(" ", "_")  # Ensure the LoRA path name is formatted correctly
TIER = args.tier.lower().replace(" ", "_")  # Ensure the tier name is formatted correctly
ignore_model = str(args.ignore_model).split(",") if args.ignore_model else []
ignore_dataset_type = str(args.ignore_dataset_type).split(",") if args.ignore_dataset_type else []
ignore_lora_rank = str(args.ignore_lora_rank).split(",") if args.ignore_lora_rank else []
TARGET_FOLDER = args.target_folder
SEED = args.seed
USE_UNSLOTH = not args.disable_unsloth if DEFAULT_USE_UNSLOTH else False # <<< ADDED
DISABLE_QUANTIZATION = args.disable_quantization

print(f"Using dataset: {DATASET_NAME}")
print(f"Using LoRA path name: {lora_path_name}")
print(f"Using model tier: {TIER}")

# --- 2. Import Necessary Libraries ---
import torch
import pandas as pd
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # <<< ADDED
from peft import PeftModel # <<< ADDED
from tqdm import tqdm
import re
import dirtyjson
import json

import random
import numpy as np


def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # If you are using CUDA (GPU)
    if torch.cuda.is_available():
        # Set the seed for all GPUs
        torch.cuda.manual_seed_all(seed)
        
    # Set a Python environment variable to ensure hash-based operations are deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

# Set the seed at the start of the script
set_seed(SEED)

# ===================================================================================
# --- 3. EXPERIMENT CONFIGURATION ---
# ===================================================================================

# --- Input and Output Files ---
INPUT_CHARACTERS_CSV = f"datasets/{DATASET_NAME}"
# --- Looping Parameters ---
LORA_RANKS_TO_EVALUATE = [0, 16, 32, 64]
DATASET_TYPES_TO_EVALUATE = ["structured", "unstructured", "summarized"]

NUM_GPUS = torch.cuda.device_count()

MODEL_CATALOG = {}
# --- Model and Adapter Catalogs ---
# This catalog must match the one used in your training script.
if TIER == "8b" or TIER == "all":
    #Inlude to the list the 8B models to the catalog
    MODEL_CATALOG.update({
        # ~8B Tier
        "llama_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "qwen_8b": "Qwen/Qwen3-8B",
        "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    })
if TIER == "13b" or TIER == "all":
    #Inlude to the list the 13B models to the catalog
    MODEL_CATALOG.update({
        # ~13B Tier
        "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
        "qwen_14b": "Qwen/Qwen3-14B",
        "deepseek_16b": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    })
if TIER == "33b" or TIER == "all":
    #Inlude to the list the 33B models to the catalog
    MODEL_CATALOG.update({
        # ~32B Tier
        "codellama_34b": "codellama/CodeLlama-34b-Instruct-hf",
        "qwen_32b": "Qwen/Qwen3-32B",
        "deepseek_33b": "deepseek-ai/deepseek-coder-33b-instruct",
    })

def _parse_key_value_fallback(response_text: str) -> dict | None:
    """
    Parses plain text "Key: Value" responses when JSON parsing fails.
    This version is adapted to handle multi-line biography fields.
    """
    # Regex to find lines that look like "Some Key: Some Value"
    # It handles multi-word keys and ignores leading/trailing whitespace.
    pattern = re.compile(r"^\s*([\w\s'’]+?)\s*:\s*(.*?)(\n|$)", re.MULTILINE)
    matches = pattern.findall(response_text)

    # Proceed only if we find at least 3 potential fields
    if len(matches) < 3:
        return None
    parsed_data = {}
    expected_keys = ['name','city', 'race', 'gender', 'location', 'biography']

    for key, value,_ in matches:
        normalized_key = key.strip().lower()

        if normalized_key.find("'s ") != -1:
            normalized_key = normalized_key.strip().lower().split("'s ")[1].strip()
        if normalized_key.find(" ") != -1:
            normalized_key = normalized_key.strip().lower().split(" ")[-1].strip()

        if normalized_key.startswith('description') or normalized_key.startswith('role') or normalized_key.startswith('story') or normalized_key.startswith('biography') or normalized_key.startswith('bio'):
            normalized_key = 'biography'
            
            # Find the position of 'Biography:' to start capturing from there.
            bio_start_index = response_text.lower().find(key.lower() + ':')
            if bio_start_index != -1:
                bio_content = response_text[bio_start_index + len(key) + 1:].strip()
                
                # Remove any leading '---' from the biography content.
                if bio_content.startswith('---'):
                    bio_content = bio_content.split('---', 1)[-1].strip()
                
                parsed_data[normalized_key] = bio_content
                break

        for expected_key in expected_keys:
            if normalized_key.find(expected_key) != -1:
                if expected_key in parsed_data and parsed_data[expected_key] is not None:
                    parsed_data[expected_key] += f" {value.strip()}"
                else:
                    parsed_data[expected_key] = value.strip()
                break
        else:
            # NEW: If an unexpected key is found, concatenate its value to the biography
            biography = parsed_data.get('biography', '')
            if biography:
                parsed_data['biography'] = f"{biography}\n{key}: {value}"
            else:
                parsed_data['biography'] = f"{key}: {value}"
    
    if len(parsed_data) >= 3:
        # Ensure all expected keys are present in the final dict
        for key in expected_keys:
            if key not in parsed_data:
                parsed_data[key] = "Unknown"
        return parsed_data
    else:
        return None

def _parse_greedy_key_value_fallback(response_text: str) -> dict | None:
    """
    Parses a string where keys and values are separated by a colon, and
    values are greedily taken until the next key.
    
    Args:
        response_text: The raw string output from the language model.
        
    Returns:
        A dictionary with the extracted data if successful, otherwise None.
    """
    print("Attempting to parse response using greedy key-value fallback...")

    # A more robust pattern to find keys and their values.
    # It looks for a word followed by a colon, capturing the word as the key.
    # The value is everything else up to the next key-colon or the end of the string.
    response_text = response_text.replace("Home City", "City")
    pattern = re.compile(r"(\w+)\s*:\s*(.*?)(?=\s*\w+\s*:|\Z)", re.DOTALL)
    matches = pattern.finditer(response_text)

    parsed_data = {}
    expected_keys = ['name', 'city', 'race', 'gender', 'location', 'biography', 'home']
    
    found_matches = False
    for match in matches:
        found_matches = True
        key = match.group(1).strip().lower()
        value = match.group(2).strip()

        # Normalize the key
        if key in ['home', 'city', 'home city', 'home_city']:
            normalized_key = 'city'
        elif key in ['bio', 'biography', 'description', 'routine', 'roleplay', 'role', 'story']:
            normalized_key = 'biography'
        else:
            normalized_key = key

        if normalized_key in expected_keys:
            parsed_data[normalized_key] = value
        else:
            # If an unexpected key is found, concatenate its value to the biography
            biography = parsed_data.get('biography', '')
            if biography:
                parsed_data['biography'] = f"{biography}\n{key}: {value}"
            else:
                parsed_data['biography'] = f"{key}: {value}"

    # We consider this a successful parse if we found at least 3 of our expected keys.
    if len(parsed_data) >= 3:
        print("Successfully extracted data with greedy key-value parser.")
        return parsed_data
    else:
        print(f"Fallback failed: Could not map at least 3 expected keys from found pairs.")
        return None


def lowercase_keys(obj):
    if isinstance(obj, dict):
        return {k.lower(): lowercase_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [lowercase_keys(i) for i in obj]
    else:
        return obj
# --- 4. Main Inference Function ---
def run_batch_inference():
    """
    Loops through all specified models and their LoRA adapters, generates responses
    for each character, and saves the results incrementally.
    """
    # Check if the input character file exists
    if not os.path.exists(INPUT_CHARACTERS_CSV):
        print(f"Error: Input file not found at '{INPUT_CHARACTERS_CSV}'")
        return

    # Load the character dataset
    character_df = pd.read_csv(INPUT_CHARACTERS_CSV)
    print(f"Loaded {len(character_df)} characters for evaluation.")

    # Outer loop: Iterate through each Lora rank
    for lora_rank in LORA_RANKS_TO_EVALUATE:
        # Inner loop: Iterate through each dataset type
        if str(lora_rank) in ignore_lora_rank:
            print(f"Skipping LoRA rank '{lora_rank}' as it is set to be ignored.")
            continue

        for dataset_type in DATASET_TYPES_TO_EVALUATE:

            print(f"\n--- Starting inference for LoRA Rank: {lora_rank}, Dataset Type: {dataset_type} ---")

            if dataset_type in ignore_dataset_type:
                print(f"Skipping dataset type '{dataset_type}' as it is set to be ignored.")
                continue

            # This is the new dynamic output file name
            OUTPUT_RESULTS_CSV = f"results_{TARGET_FOLDER}/evaluation_results_R{lora_rank}_{TIER}_{dataset_type}.csv"
            
            # Check for and load existing results to resume if possible
            if os.path.exists(OUTPUT_RESULTS_CSV):
                print(f"Found existing results at '{OUTPUT_RESULTS_CSV}'. Resuming inference.")
                results_df = pd.read_csv(OUTPUT_RESULTS_CSV)

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

                # Create a set of (character_name, model_key) tuples for fast lookup of completed tasks
                completed_tasks = set(tuple(row) for row in results_df[['character_name', 'model_key']].values)
            else:
                print(f"No existing results found at '{OUTPUT_RESULTS_CSV}'. Starting from scratch.")
                results_df = pd.DataFrame(columns=['character_name', 'model_key', 'question', 'home_city', 'race', 'gender', 'location', 'biography', 'generate_type', 'generated_response'])
                completed_tasks = set()

            # Innermost loop: Iterate through each model to be tested
            for model_key, model_id in MODEL_CATALOG.items():
                
                # A temporary list to hold results for the CURRENT model before saving
                new_results_for_model = []

                # --- NEW LOGIC for checking if a model is fully processed ---
                # This check is now more reliable because `completed_tasks` is loaded only once per loop
                # and reflects the state of the saved CSV.
                all_chars_for_model = [row['name'] for _, row in character_df.iterrows() if int(row['for_evaluation']) == 1]
                processed_chars_for_model = [(char_name, model_key) for char_name in all_chars_for_model if (char_name, model_key) in completed_tasks]
                
                if len(processed_chars_for_model) == len(all_chars_for_model):
                    print(f"All characters for model '{model_key}' already processed. Skipping to the next model.")
                    continue
                # --- END NEW LOGIC ---

                if model_key == "deepseek_16b":
                    print(f"WARNING: Line 1728 from modeling_deepseek.py for deepseek_16b must be changed  to 'max_cache_length = past_key_values.get_max_cache_shape()' to avoid errors.")
                    print(f"WARNING: Look for modeling_deepseek.py inside ~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11/ and at ~/.cache/huggingface/hub/d.")
                
                lora_path = None # Initialize lora_path

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

                # Check if the LoRA adapter directory exists
                if lora_rank > 0 and not os.path.isdir(lora_path):
                    print(f"Warning: LoRA adapter not found for '{model_key}' with rank {lora_rank} and dataset '{dataset_type}'. Skipping.")
                    continue

                #Check if the model is the one to ignore
                if model_key in ignore_model:
                    print(f"Skipping model '{model_key}' as it is set to be ignored.")
                    continue

                if lora_rank > 0:
                    print(f"\n--- Loading Model: {model_key} with LoRA adapter: {lora_path} ---")
                else:
                    print(f"\n--- Loading BASE Model (Rank 0): {model_key} ---")
            
                # ========================= MODIFIED LOADING BLOCK =========================

                MUST_AVOID_UNSLOTH = False # Example models that Unsloth may not support well
                if model_key == "deepseek_33b":
                    print(f"Model '{model_key}' is known to have issues with Unsloth. Disabling Unsloth for this model.")
                    MUST_AVOID_UNSLOTH
                # ========================= MODIFIED LOADING BLOCK =========================
                if USE_UNSLOTH and not MUST_AVOID_UNSLOTH:
                    # Load the base model and tokenizer using Unsloth
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=model_id,
                        device_map = "auto",
                        max_seq_length=2048,
                        dtype=None,
                        load_in_4bit=not DISABLE_QUANTIZATION,
                        trust_remote_code=True,
                    )
                    # Conditionally load the LoRA adapter
                    if lora_rank > 0 and lora_path:
                        model.load_adapter(lora_path, "default") # Added "default" for safety
                    
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

                last_use_cache = True
                # Inner loop: Iterate through each character in the dataset
                for _, row in tqdm(character_df.iterrows(), total=len(character_df), desc=f"Generating responses for {model_key}"):
                    character_name = row['name']
                    for_evaluation = row['for_evaluation']
                    if int(for_evaluation) != 1:
                        continue

                    # Check if this character has already been processed by this model
                    if (character_name, model_key) in completed_tasks:
                        continue

                    # Loop for multiple attempts
                    for i in range(MAX_RETRIES): 
                        generate_type = "deterministic" if i <= 1 else "non-deterministic"                               
                        if i == 0:
                            print(f"Attempt {i+1} for character {character_name} using JSON output.")
                            # --- Define the question to ask for JSON output ---
                            question = (
                                f"Based on the character {character_name} from the Skyrim universe, identify their name, home city, race, gender, location, and biography. "
                                f"If a specific piece of information is not available or explicitly stated in the biography, please state 'Unknown' for that attribute. "
                                f"Please provide the character's attributes and their short biography in a structured format (json or other formatted output),  filling in details or 'Unknown' as appropriate. "
                                #f"Example: Name: ... \n City: ... \n Race: ... \n Gender: ... \n Location: ... \n Biography: ..."
                                f"Example: \n { {'Name': '...', 'City': '...', 'Race': '...', 'Gender': '...', 'Location': '...', 'Biography': '...'} }"
                            )
                        else:# i == 1:
                            print(f"Attempt {i+1} for character {character_name} using alternative output.")
                            # --- Define the question to ask for JSON output ---
                            question = (
                                f"Based on the character {character_name} from the Skyrim universe, identify their name, home city, race, gender, location, and biography. "
                                f"If a specific piece of information is not available or explicitly stated in the biography, please state 'Unknown' for that attribute. "
                                f"Please provide the character's attributes and their short biography in a structured format (json or other formatted output),  filling in details or 'Unknown' as appropriate. "
                                f"Example: Name: ... \n City: ... \n Race: ... \n Gender: ... \n Location: ... \n Biography: ..."
                            )
                        messages = [{"role": "user", "content": question}]
                        
                        # STEP 1: Apply the chat template to get the formatted prompt string.
                        prompt_string = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        # STEP 2: Tokenize the formatted string.
                        inputs = tokenizer(
                            prompt_string,
                            return_tensors="pt"
                        ).to(model.device)
                        if i <= 1: 
                            try:
                                # Generate the response with deterministic settings
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=1024,
                                    use_cache=last_use_cache,#False if NUM_GPUS > 1 else True,
                                    # --- Parameters to control repetition and improve quality ---
                                    repetition_penalty=1.00,      # Add a penalty for repeating tokens.
                                    no_repeat_ngram_size=5,       # Prevent repeating sequences of 5 tokens.
                                    #For deterministic output
                                    do_sample=False,              # Force deterministic output.
                                    num_beams =1,                 # for do_sample=False, Use beam search for better quality but increase memory usage.
                                    eos_token_id=tokenizer.eos_token_id,
                                    top_p=None,
                                    top_k=None,
                                    temperature=None
                                )
                            except Exception as e:
                                print(f"Fail to use use_cache=true for model {model_key} with LoRA rank {lora_rank} and dataset type {dataset_type}. Error: {e}.\n Attempting to generate with use_cache=False.")
                                # Generate the response with deterministic settings
                                last_use_cache = False
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=1024,
                                    use_cache=last_use_cache, #Try to force use cache to False to fix issues
                                    # --- Parameters to control repetition and improve quality ---
                                    repetition_penalty=1.00,      # Add a penalty for repeating tokens.
                                    no_repeat_ngram_size=5,       # Prevent repeating sequences of 5 tokens.
                                    #For deterministic output
                                    do_sample=False,              # Force deterministic output.
                                    num_beams =1,                 # for do_sample=False, Use beam search for better quality but increase memory usage.
                                    eos_token_id=tokenizer.eos_token_id,
                                    top_p=None,
                                    top_k=None
                                )

                        else: 
                            try:
                                # Generate the response with non-deterministic settings
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=1024,
                                    use_cache=True,#False if NUM_GPUS > 1 else True,
                                    # --- Parameters to control repetition and improve quality ---
                                    repetition_penalty=1.15,      # Add a penalty for repeating tokens.
                                    no_repeat_ngram_size=5,       # Prevent repeating sequences of 3 tokens.
                                    #For non-deterministic output
                                    do_sample=True,
                                    temperature=0.5,           # Increased slightly to help the model escape generation loops.
                                    top_p=0.9,                 # Reduced to focus on more confident tokens.
                                    top_k=20,                  # Reduced to focus on a smaller, more confident set of tokens.
                                    eos_token_id=tokenizer.eos_token_id
                                )
                            except Exception as e:
                                print(f"Fail to use use_cache=true for model {model_key} with LoRA rank {lora_rank} and dataset type {dataset_type}. Error: {e}.\n Attempting to generate with use_cache=False.")
                                # Generate the response with non-deterministic settings
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=1024,
                                    use_cache=False, #force to false
                                    # --- Parameters to control repetition and improve quality ---
                                    repetition_penalty=1.15,      # Add a penalty for repeating tokens.
                                    no_repeat_ngram_size=5,       # Prevent repeating sequences of 3 tokens.
                                    #For non-deterministic output
                                    do_sample=True,
                                    temperature=0.5,           # Increased slightly to help the model escape generation loops.
                                    top_p=0.9,                 # Reduced to focus on more confident tokens.
                                    top_k=20,                  # Reduced to focus on a smaller, more confident set of tokens.
                                    eos_token_id=tokenizer.eos_token_id
                                )

                        
                        response_list = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        response_text = response_list[0].strip() if response_list else ""

                        #Remove reasoning <think> tags if present
                        if response_text.find('<think>') != -1:
                            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                            response_text = '\n'.join([line for line in response_text.split('\n') if line.strip()])
                        response_text = response_text.replace('<think>', ' ')  # Replace <think> tags with spaces for better JSON parsing
                        response_text = response_text.replace('</think>', ' ')  # Replace </think> tags with spaces for better JSON parsing

                        json_indicator = response_text.find('JSON Output:')
                        if json_indicator != -1:
                            response_text = response_text[json_indicator+12:]


                        # This is your original JSON parsing block
                        try:
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            if json_end == 0:
                                json_end = len(response_text)  # If no closing brace found, use the end of the string
                            if json_start != -1 and json_end != 0:
                                json_str = response_text[json_start:json_end]
                                parsed_json = dirtyjson.loads(json_str)
                                parsed_json = lowercase_keys(parsed_json)
                                
                                # Basic validation to ensure it's not an empty JSON
                                if not parsed_json:
                                    raise ValueError("Parsed an empty JSON object.")
                                else:
                                    print(f"Successfully parsed JSON for character {character_name} on attempt {i+1}.")
                                # NEW: Handle unexpected keys from a successful JSON parse
                                expected_keys = ['character_name', 'name', 'home_city', 'city', 'race', 'gender', 'location', 'biography']
                                if 'biography' not in parsed_json:
                                    parsed_json['biography'] = ""
                                
                                for key in list(parsed_json.keys()):
                                    if key not in expected_keys:
                                        biography_content = f"{key}: {parsed_json[key]}"
                                        # Check if biography is a pandas Series and merge it into a string if necessary
                                        if isinstance(parsed_json['biography'], pd.Series):
                                            parsed_json['biography'] = '\n'.join(parsed_json['biography'].dropna().astype(str))
                                        parsed_json['biography'] = f"{parsed_json['biography']}\n{biography_content}".strip()
                                        del parsed_json[key]
                            else:
                                # This will trigger the exception and our fallback logic
                                raise ValueError("No JSON object found in response.")

                        # --- START: MODIFIED EXCEPTION HANDLING ---
                        except (dirtyjson.Error, AttributeError, KeyError, Exception) as e:
                            # Fallback #1: Key-value parsing
                            parsed_json = _parse_key_value_fallback(response_text)
                            if not parsed_json:
                                # Fallback #2: Greedy key-value parsing
                                parsed_json = _parse_greedy_key_value_fallback(response_text)
                                if not parsed_json:
                                    # If all fallbacks fail, print warning and trigger retry
                                    if i < MAX_RETRIES - 1:
                                        print("Retrying...")
                                        continue # Go to the next attempt in the retry loop
                                    else:
                                        #response text must be safe for csv, remove " and add " at the begining and end
                                        response_text = response_text.replace('"', ' ')
                                        response_text = f'"{response_text}"'
                                        print(f"Max retries reached. Saving the text without the specific answers for character {character_name}, model {model_key}, lora rank {lora_rank}, dataset type {dataset_type}: {response_text}")
                                        new_results_for_model.append({
                                            "character_name": character_name,
                                            "model_key": model_key,
                                            "question": question,
                                            "home_city": 'Unknown',
                                            "race":'Unknown',
                                            "gender": 'Unknown',
                                            "location": 'Unknown',
                                            "biography": response_text,
                                            "generate_type": generate_type,
                                            "generated_response": response_text
                                        })
                                        break

                                    # Optionally, save an error entry
                        # --- END: MODIFIED EXCEPTION HANDLING ---

                        try:
                            biography_content = parsed_json['biography']
                        except KeyError:
                            continue

                        # --- START OF FIX ---
                        # Check if the content is a pandas Series and merge it into a string if necessary
                        if isinstance(biography_content, pd.Series):
                            biography_content = '\n'.join(biography_content.dropna().astype(str))
                        # Check if the content is a dictionary and serialize it to a string if necessary
                        elif isinstance(biography_content, dict):
                            biography_content = json.dumps(biography_content, indent=2)
                        #response text must be safe for csv, remove " and add " at the begining and end
                        response_text = response_text.replace('"', ' ')
                        response_text = f'"{response_text}"'
                        new_results_for_model.append({
                            "character_name": character_name,
                            "model_key": model_key,
                            "question": question,
                            "home_city": parsed_json.get('city', parsed_json.get('home', parsed_json.get('home_city', parsed_json.get('home city', 'Unknown')))),
                            "race": parsed_json.get('race', 'Unknown'),
                            "gender": parsed_json.get('gender', 'Unknown'),
                            "location": parsed_json.get('location', 'Unknown'),
                            "biography": biography_content,
                            "generate_type": generate_type,
                            "generated_response": response_text
                        })
                        break

                # --- NEW: Place saving logic here, after the character loop completes ---
                if new_results_for_model:
                    new_results_df = pd.DataFrame(new_results_for_model)
                    results_df = pd.concat([results_df, new_results_df], ignore_index=True)


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

    print("\nBatch inference complete.")


# --- 5. Run the Script ---
if __name__ == "__main__":
    run_batch_inference()