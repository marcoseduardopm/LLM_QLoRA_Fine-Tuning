# ===================================================================================
# Comprehensive Evaluation Script for Skyrim Fine-Tuning
# ===================================================================================


import os
import glob
import warnings
import argparse
import wandb  # Optional: For Weights & Biases integration
import numpy as np
import time
import random
import pandas as pd
import torch
import fcntl # For file locking on Unix-like systems


#Default values
DEFAULT_CUDA_VISIBLE_DEVICES = None #"0" #Values: None, "0,1,2", "0", "1", "2"
DEFAULT_DATASET_NAME = "skyrim_characters_all.csv" #Values: None, "0,1,2", "0", "1", "2"
DEFAULT_IGNORE_DATASET_TYPE = "" #Values: "", "structured", "unstructured", "summarized"
DEFAULT_IGNORE_LORA_RANK = "0,32,64" #Values: "0","16","32", "64"
DEFAULT_IGNORE_TIER = "13b,33b"#"13b,33b" #Values: "8b", "13b", "33b"
DEFAULT_TARGET_FOLDER = "seed_1"
DEFAULT_FINAL_OUTPUT_CSV = f"./results_{DEFAULT_TARGET_FOLDER}/final_evaluation_all_configs.csv"
DEFAULT_SEED = 4315468 # Define the default seed
DEFAULT_USE_GGUF_JUDGE = False
#DEFAULT_GGUF_MODEL = "EVA-Qwen2.5-72B-v0.2.Q4_K_S.gguf"
DEFAULT_GGUF_MODEL = ""
DEFAULT_PERFORM_LLM_AS_JUDGE_EVAL = True
DEFAULT_FORCE_REGENERATE_RANK_JUDGE = "" 
DEFAULT_GEMINI_MODEL_ID = "models/gemini-2.0-flash-001" #"models/gemini-1.5-pro-latest" "models/gemini-2.5-flash-lite" #"models/gemini-1.5-pro-latest"
DEFAULT_JUDGE_GGUF_N_GPU_LAYERS = 0#78 # Number of layers to offload to GPU for GGUF model
DEFAULT_TENSOR_SPLIT = None #"[30, 24, 24]"#[24, 24],
DEFAULT_USE_UNSLOTH = True 

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Advanced LoRa Fine-Tuning Script for Skyrim Character Dataset - Evaluation Phase")

# Optional arguments with default values
parser.add_argument("--cuda_devices", type=str, default=DEFAULT_CUDA_VISIBLE_DEVICES,
                    help="Comma-separated list of CUDA device IDs to make visible (e.g., '0,1,2'). If not provided, os.environ['CUDA_VISIBLE_DEVICES'] will not be set by the script.")
parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME,
                    help="Name of the dataset to use for evaluation. Default is 'skyrim_characters_all_example.csv'. If not provided, the script will not set a specific dataset name.")
parser.add_argument("--final_output_csv", type=str, default=DEFAULT_FINAL_OUTPUT_CSV,
                    help="Path to the final output CSV file where evaluation results will be saved. Default is './results/final_evaluation_all_configs.csv'. If not provided, the script will not set a specific output path.")
parser.add_argument("--perform_llm_as_judge_eval", action='store_true', default=DEFAULT_PERFORM_LLM_AS_JUDGE_EVAL,
                    help="Set this flag to perform LLM-as-a-Judge evaluation. Default is False.")
parser.add_argument("--gemini_model_id", type=str, default=DEFAULT_GEMINI_MODEL_ID,
                    help="Gemini model ID to use as the judge. If not provided, the script will not set a specific Gemini model ID.")
parser.add_argument("--ignore_dataset_type", type=str, default=DEFAULT_IGNORE_DATASET_TYPE,
                    help="Dataset type(s) to ignore during evaluation. Separate multiple types with commas (,). Default is ''. If not provided, the script will not set a specific ignore dataset type.")
parser.add_argument("--ignore_lora_rank", type=str, default=DEFAULT_IGNORE_LORA_RANK,
                    help="LoRA rank(s) to ignore during evaluation. Separate multiple ranks with commas (,). Default is ''. If not provided, the script will not set a specific ignore LoRA rank.")
parser.add_argument("--ignore_tier", type=str, default=DEFAULT_IGNORE_TIER,
                    help="Model tier(s) to ignore during evaluation. Separate multiple tiers with commas (,). Default is ''. If not provided, the script will not set a specific ignore tier.")
parser.add_argument("--force_regenerate_model", type=str, default="", 
                    help="Specify a model key (e.g., 'llama_8b') to force regeneration of its evaluation results, deleting existing entries for it.")
parser.add_argument("--force_regenerate_rank_judge", type=str, default=DEFAULT_FORCE_REGENERATE_RANK_JUDGE, 
                    help="Specify a LoRA rank (e.g., '16') to force regeneration of judge results for that rank, overwriting existing entries.")
parser.add_argument("--target_folder", type=str, default=DEFAULT_TARGET_FOLDER,
                    help="Target folder.  If not provided, the script will not set a specific target folder.")
parser.add_argument("--calculate_perplexity", action='store_true',
                    help="Set to True to calculate perplexity during evaluation. Default is False.")
parser.add_argument("--disable_unsloth", action="store_true", 
                    help="If set, disables Unsloth and uses standard Transformers/Peft loading.")
parser.add_argument("--disable_quantization", action="store_true", 
                    help="If set, disables 4-bit quantization and loads the model in 16-bit precision. Defaults to False.")
parser.add_argument("--use_gguf_judge", action='store_true', default=DEFAULT_USE_GGUF_JUDGE, 
                    help="Set this flag to use a local GGUF model as the judge instead of Gemini.")
parser.add_argument("--gguf_model", type=str, default=DEFAULT_GGUF_MODEL, 
                    help="Path to the GGUF model to use as the judge. Default is 'EVA-Qwen2.5-72B-v0.2.Q4_K_S.gguf'.")
parser.add_argument("--gguf_gpu_layers", type=int, default=DEFAULT_JUDGE_GGUF_N_GPU_LAYERS, 
                    help="Number of layers to offload to GPU for the GGUF judge model. Default is 78.")
parser.add_argument("--gguf_tensor_split", type=str, default=DEFAULT_TENSOR_SPLIT, 
                    help="Tensor split configuration for GGUF model. Default is '[30, 24, 24]'.")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for reproducibility. Defaults to 4315468.")
args = parser.parse_args()

DATASET_NAME = args.dataset_name
FINAL_OUTPUT_CSV = args.final_output_csv
PERFORM_LLM_AS_JUDGE_EVAL = args.perform_llm_as_judge_eval
TARGET_FOLDER = str(args.target_folder) if args.target_folder else ""
CALCULATE_PERPLEXITY = args.calculate_perplexity
USE_GGUF_JUDGE = args.use_gguf_judge 
USE_GEMINI_JUDGE = not args.use_gguf_judge 
JUDGE_GEMINI_MODEL_ID = args.gemini_model_id
GGUF_MODEL = args.gguf_model 
SEED = args.seed
JUDGE_GGUF_N_GPU_LAYERS = args.gguf_gpu_layers#78 # Number of layers to offload to GPU for GGUF model
USE_UNSLOTH = not args.disable_unsloth if DEFAULT_USE_UNSLOTH else False 
DISABLE_QUANTIZATION = args.disable_quantization
FORCE_REGENERATE_RANK_JUDGE = args.force_regenerate_rank_judge 

#convert the string representation of a list to an actual list
#"[30, 24, 24]" -> [30, 24, 24]
if args.gguf_tensor_split is not None:
    TENSOR_SPLIT = eval(args.gguf_tensor_split)#[30, 24, 24]#[24, 24],
else:
    TENSOR_SPLIT = None
#EVA-Qwen2.5-72B-v0.2.Q3_K_M.gguf

#if USE_GGUF_JUDGE is set, setting cuda_devices here tends to cause issues
if not USE_GGUF_JUDGE:
    # Apply parsed arguments
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        # If not provided, ensure it's not set by the script if it was previously
        if DEFAULT_CUDA_VISIBLE_DEVICES is None and "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
else:
    print("Using GGUF judge model; skipping CUDA_VISIBLE_DEVICES setting to avoid conflicts.")

if TARGET_FOLDER != DEFAULT_TARGET_FOLDER:
    FINAL_OUTPUT_CSV = f"./results_{TARGET_FOLDER}/final_evaluation_all_configs.csv"
    print(f"Note: FINAL_OUTPUT_CSV updated to {FINAL_OUTPUT_CSV} based on TARGET_FOLDER")

ignore_dataset_type = str(args.ignore_dataset_type).split(",") if args.ignore_dataset_type else []
ignore_lora_rank = str(args.ignore_lora_rank).split(",") if args.ignore_lora_rank else []
ignore_tier = str(args.ignore_tier).split(",") if args.ignore_tier else []


# --- 2. Import Necessary Libraries ---
# --- Evaluation Configuration ---
PERFORM_LLM_AS_JUDGE_EVAL_REASONING = False # If True, the judge will provide reasoning for their rankings

# NOTE: USE_GGUF_JUDGE and USE_GEMINI_JUDGE are now set via command-line arguments
# See the "Apply parsed arguments" section below.

if args.use_gguf_judge: # <<< CHANGED: Checks the parsed argument
    from llama_cpp import Llama as Llama_cpp



# --- 2. Import Necessary Libraries ---
if CALCULATE_PERPLEXITY:
    from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
from peft import PeftModel# --- ADDED ---
from tqdm import tqdm
import os
import evaluate
import json
from datasets import Dataset
from torch.utils.data import DataLoader
from huggingface_hub import login
import re # NEW: Import regular expression module

import json
import google.generativeai as genai # NEW: For Gemini API
import dirtyjson

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

# filename: load_gguf_model.py


# ===================================================================================
# --- File Locking Class ---
# ===================================================================================
class FileLocker:
    def __init__(self, file_path):
        self.lock_file = f"{file_path}.lock"
        self.file_handle = None

    def acquire_lock(self, timeout=60, poll_interval=1):
        """Acquires a file lock with a timeout."""
        start_time = time.time()
        while True:
            try:
                self.file_handle = open(self.lock_file, 'w')
                fcntl.flock(self.file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (IOError, OSError) as e:
                # File is already locked, wait and retry
                if self.file_handle:
                    self.file_handle.close()
                self.file_handle = None
                if time.time() - start_time > timeout:
                    print(f"Error: Timeout ({timeout}s) exceeded while waiting for file lock.")
                    return False
                wait_time = poll_interval + random.uniform(0, 1) # Add some randomness
                print(f"File '{self.lock_file}' is locked. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                
    def release_lock(self):
        """Releases the file lock and closes the handle."""
        if self.file_handle:
            fcntl.flock(self.file_handle, fcntl.LOCK_UN)
            self.file_handle.close()
            try:
                os.remove(self.lock_file)
            except OSError as e:
                print(f"Error removing lock file: {e}")

# ===================================================================================
# --- 3. EXPERIMENT CONFIGURATION ---
# ===================================================================================
# NEW: Gemini API Key SET YOUR GEMINI API KEY HERE or use an environment variable
#Lets use a set of keys instead, so that if one fails we can keep testing the next one

GEMINI_API_KEYS = os.environ["GEMINI_API_KEYS"].split(",") if "GEMINI_API_KEYS" in os.environ else []

print(f"Loaded {len(GEMINI_API_KEYS)} Gemini API keys from environment.")
MAX_JUDGE_RETRIES = 20

CURRENT_GEMINI_KEY_INDEX = 0

# --- Judge Model Configuration ---
lora_path_name = ""#"final_0_05"
judge_model_instance = None
judge_tokenizer = None,

# NEW: GGUF Judge Model Configuration
# See the USE_GGUF_JUDGE variable in the import section.
if USE_GGUF_JUDGE:
    USE_UNSLOTH_JUDGE = False
    JUDGE_MODEL_PATH = f"models/{GGUF_MODEL}" # Path to your GGUF judge model
    JUDGE_GGUF_N_CTX = 8192#
else:
    USE_UNSLOTH_JUDGE = True
    #"""
    JUDGE_MODEL_PATH = "deepseek-ai/deepseek-llm-7b-chat"
    #JUDGE_MODEL_PATH = "deepseek-ai/deepseek-llm-67b-chat"
    JUDGE_LOAD_IN_4BIT = True # Set to True to enable 4-bit quantization for non-quantized models
    #"""

# Hugging Face Token (optional, but recommended)
HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"] if "HUGGING_FACE_TOKEN" in os.environ else ""

# --- Input and Output Files ---
# The evaluation script will now dynamically generate these paths based on loops
# So these paths are just defaults or not used in the main logic.
# INFERENCE_RESULTS_CSV = "./results/evaluation_results.csv"

# --- Looping Parameters ---
LORA_RANKS_TO_EVALUATE = [0, 16, 32, 64]
DATASET_TYPES_TO_EVALUATE = ["structured", "unstructured", "summarized"]
TIERS_TO_EVALUATE = ["8B", "13B", "33B"]
#TIERS_TO_EVALUATE = ["8b"]

# --- Model and Adapter Paths ---
# The script will now dynamically construct these paths inside the loops
# The catalog is kept here for easy reference and dynamic lookup
MODEL_CATALOG_8B = {
    "llama_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen_8b": "Qwen/Qwen3-8B",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
}
MODEL_CATALOG_13B = {
    "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
    "qwen_14b": "Qwen/Qwen3-14B",
    "deepseek_16b": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
}
MODEL_CATALOG_33B = {
    "codellama_34b": "codellama/CodeLlama-34b-Instruct-hf",
    "qwen_32b": "Qwen/Qwen3-32B",
    "deepseek_33b": "deepseek-ai/deepseek-coder-33b-instruct",
}

# Consolidate catalogs for easier access
ALL_MODEL_CATALOGS = {
    "8b": MODEL_CATALOG_8B,
    "13b": MODEL_CATALOG_13B,
    "33b": MODEL_CATALOG_33B,
}

REFERENCE_DATASET_CSV = f"./datasets/{DATASET_NAME}"


# --- 3. Define Chat Template ---
# This logic is for the *local* LLM judge. Gemini has its own API handling.
chat_template = None
if "coder" in JUDGE_MODEL_PATH.lower():
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '### Instruction:\\n' + message['content'] + '\\n### Response:\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + '<|EOT|>' }}"
        "{% endif %}"
        "{% endfor %}"
    )
elif "deepseek" in JUDGE_MODEL_PATH.lower():
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ 'User: ' + message['content'] + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ 'Assistant: ' + message['content'] + '<|end_of_sentence|>' }}"
        "{% endif %}"
        "{% endfor %}"
    )

def is_garbage_output(text):
    """
    Detects if the text is likely model collapse/garbage.
    Returns True if garbage, False if it looks okay.
    """
    if not isinstance(text, str) or len(text.strip()) < 5:
        return True # Too short or empty

    # 1. Check for Repetition (e.g., "............" or ",,,,,,,")
    # If unique characters are < 5% of total characters, it's likely repetitive garbage
    if len(set(text)) / len(text) < 0.05:
        return True

    # 2. Check for Token/Word Repetition (e.g., "importimportimport")
    # A simple heuristic: if the most common word appears > 50% of the time
    words = text.split()
    if len(words) > 10:
        from collections import Counter
        counts = Counter(words)
        most_common = counts.most_common(1)[0][1]
        if most_common / len(words) > 0.5:
            return True

    # 3. Check for Encoding Artifacts / Non-printable spam
    # The example "ï»¿ï»¿" is often UTF-8 BOM garbage. 
    # If the text has high density of non-alphanumeric chars (excluding standard punctuation)
    # This regex matches anything that ISN'T a letter, number, or standard punctuation
    garbage_chars = len(re.findall(r'[^\w\s.,!?\'"-]', text))
    if garbage_chars / len(text) > 0.3: # If > 30% of text is weird symbols
        return True

    return False

# ===================================================================================
# --- 4. Perplexity Calculation ---
# ===================================================================================
def calculate_perplexity(results_df, reference_df, lora_rank, dataset_type, model_catalog):
    """
    Calculates perplexity for each fine-tuned model on a held-out test set.
    """
    print("\n--- Calculating Perplexity ---")
    if not CALCULATE_PERPLEXITY:
        print("Skipping perplexity calculation as per configuration.")
        return results_df
    
    #if USE_GGUF_JUDGE:
    #    print("Perplexity calculation with GGUF judge model is not implemented. Skipping.")
    #    return results_df
    
    # === FIX: Ensure all required columns exist before use ===
    required_cols = {
        'model_key': pd.NA,
        'lora_rank': pd.NA, #lora_rank, # Use the passed-in value as the default
        'dataset_type': pd.NA, #dataset_type, # Use the passed-in value as the default
        'perplexity': pd.NA
    }
    for col, default_val in required_cols.items():
        if col not in results_df.columns:
            results_df[col] = default_val

    # If it encounters a saved large perplexity (e.g. > 100), set it to 100.0.
    # This flags the model as "poor quality" but "evaluated," preventing infinite reruns.
    if 'perplexity' in results_df.columns:
        # Convert column to numeric, forcing non-numeric to NaN for safety
        ppl_numeric = pd.to_numeric(results_df['perplexity'], errors='coerce')
        
        # Identify rows where perplexity is absurdly high (e.g., > 100)
        high_ppl_mask = ppl_numeric > 100.0 
        
        if high_ppl_mask.any():
            count = high_ppl_mask.sum()
            print(f"Found {count} existing perplexity values > 100.0. Capping them at 100.0 to avoid recalculation.")
            results_df.loc[high_ppl_mask, 'perplexity'] = 100.0

    # Use a fraction of the reference data as a held-out test set for perplexity
    if (len(reference_df) > 20): # If we have more than 20 samples, use a subset for testing
        test_df = reference_df.sample(frac=0.1, random_state=42)
    else:
        test_df = reference_df
    if dataset_type == "structured" or dataset_type == "unstructured":
        test_texts = test_df['bio'].dropna().tolist() #always use bio field for perplexity
    else: #summarized
        test_texts = test_df['bio_summary'].dropna().tolist()



    # CORRECTED LINE: Iterate over unique combinations of the three keys
    unique_configs = results_df[['model_key', 'lora_rank', 'dataset_type']].drop_duplicates()
    
    perplexity_scores = {}

    for _, row in tqdm(unique_configs.iterrows(), desc="Calculating Perplexity for Models"):
        model_key = row['model_key']
        lora_rank = row['lora_rank']
        dataset_type = row['dataset_type']
        
        # CORRECTED LINE: Check for existing perplexity score on all three keys
        if pd.notnull(results_df[
            (results_df['model_key'] == model_key) & 
            (results_df['lora_rank'] == lora_rank) &
            (results_df['dataset_type'] == dataset_type)
        ]['perplexity']).any():
            print(f"Perplexity for {model_key} (R{lora_rank}, {dataset_type}) already exists. Skipping.")
            continue
        try:
            base_model_id = model_catalog.get(model_key)
            
            # <<< --- START OF RANK 0 FIX --- >>>
            adapter_path = None # Initialize adapter_path

            if lora_rank > 0:
            # Dynamically construct the path
                #if no lora_path_name is provided, search for the lastest checkpoint
                if lora_path_name == "":
                    lora_dir = f"results_{TARGET_FOLDER}/R{lora_rank}_{model_key}_{dataset_type}"
                    # Otherwise, look for the latest checkpoint within the determined training_output_dir.
                    checkpoints = list(glob.glob(os.path.join(lora_dir, "checkpoint-*")))
                    if len(checkpoints) > 0:
                        checkpoints.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
                        adapter_path = checkpoints[-1]
                    else:
                        print(f"Warning: No checkpoints found in '{lora_dir}' for '{model_key}' with rank {lora_rank} and dataset '{dataset_type}'. Skipping.")
                        continue
                else:
                    # Dynamically construct the path to the LoRA adapter
                    adapter_path = f"results_{TARGET_FOLDER}/R{lora_rank}_{model_key}_{dataset_type}/{lora_path_name}"

                if not base_model_id or not os.path.exists(adapter_path):
                    print(f"Warning: Model or adapter path not found for {model_key}. Skipping perplexity calculation.")
                    continue
            elif lora_rank == 0:
                print(f"Info: Calculating perplexity for base model (Rank 0) {model_key}.")
                if not base_model_id:
                    print(f"Warning: Base model ID not found for {model_key}. Skipping.")
                    continue

            # Define a fixed max_length and use it consistently.
            max_length = 1024

            MUST_AVOID_UNSLOTH = False # Example models that Unsloth may not support well
            if model_key == "deepseek_33b":
                print(f"Model '{model_key}' is known to have issues with Unsloth. Disabling Unsloth for this model.")
                MUST_AVOID_UNSLOTH = True

            if USE_UNSLOTH and not MUST_AVOID_UNSLOTH:
                # Load base model and tokenizer from Unsloth
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model_id,
                    token=HUGGING_FACE_TOKEN,
                    max_seq_length=max_length, # Use the fixed value here
                    dtype=None,
                    load_in_4bit=not DISABLE_QUANTIZATION,
                    trust_remote_code=True
                )
                # Conditionally load the LoRA adapter
                if lora_rank > 0 and adapter_path:
                    model.load_adapter(adapter_path)

            else:
                print("Unsloth disabled (or not used for perplexity). Using standard Transformers/Peft loading...")
                # Define quantization configuration for 4-bit loading
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
                tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=HUGGING_FACE_TOKEN, trust_remote_code=True)
                
                # Load base model
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    token=HUGGING_FACE_TOKEN,
                    # max_length argument is deprecated in from_pretrained for some versions, 
                    # but harmless if ignored. We set it in tokenizer/generation usually.
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Conditionally apply the LoRA adapter
                if lora_rank > 0 and adapter_path:
                    model = PeftModel.from_pretrained(model, adapter_path)
            # ==========================================================================
            
            model = model.eval()

            encodings = tokenizer("\n\n".join(test_texts), return_tensors="pt")
            # This next line is intentionally NOT using model.config.max_position_embeddings
            stride = 512
            seq_len = encodings.input_ids.size(1)

            nlls = []
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                target_ids[target_ids >= model.config.vocab_size] = -100


                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss
                if(neg_log_likelihood.item() > 100):
                    print(f"High perplexity detected for {model_key} (R{lora_rank}, {dataset_type}): {neg_log_likelihood.item():.4f}, ignoring")
                    continue

                current_nll = neg_log_likelihood.item()
                if np.isnan(current_nll) or np.isinf(current_nll) or current_nll > 100.0:
                    print(f"High perplexity detected ({current_nll:.4f}). Capping at 100.0.")
                    nlls.append(torch.tensor(100.0)) # Append the cap value
                else:
                    nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
            
            ppl = torch.exp(torch.stack(nlls).mean())
            
            # Correctly store the perplexity with the full set of keys
            perplexity_scores[(model_key, lora_rank, dataset_type)] = ppl.item()
            print(f"Perplexity for {model_key} (R{lora_rank}, {dataset_type}): {ppl.item():.4f}")

        except Exception as e:
            print(f"Error calculating perplexity for {model_key}: {e}")
            perplexity_scores[(model_key, lora_rank, dataset_type)] = None
        if 'model' in locals():
            # Clean up memory
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
    
    # ========================= CORRECTED SECTION =========================
    # First, create a temporary DataFrame from the new scores
    ppl_df = pd.DataFrame(
        list(perplexity_scores.items()),
        columns=['key_tuple', 'perplexity']
    )
    
    # Check if there are any new scores to add
    if ppl_df.empty:
        print("No new perplexity scores to add. Returning original DataFrame.")
        return results_df
        
    ppl_df[['model_key', 'lora_rank', 'dataset_type']] = pd.DataFrame(
        ppl_df['key_tuple'].tolist(), 
        index=ppl_df.index
    )

    # Set the index on both DataFrames for an efficient update
    ppl_df = ppl_df.set_index(['model_key', 'lora_rank', 'dataset_type'])
    
    # Ensure the results_df has the correct columns before setting index
    if 'lora_rank' not in results_df.columns:
        results_df['lora_rank'] = lora_rank
    if 'dataset_type' not in results_df.columns:
        results_df['dataset_type'] = dataset_type
    
    results_df = results_df.set_index(['model_key', 'lora_rank', 'dataset_type'])
    
    # Use update on the DataFrame itself to fill NaN values from the new scores.
    # This correctly modifies the DataFrame in place.
    results_df.update(ppl_df)

    # Reset the index on the results_df before returning
    results_df = results_df.reset_index()

    return results_df



def calculate_rouge_bleu(results_df, reference_df, lora_rank, dataset_type, model_catalog):
    """
    Calculates ROUGE and BLEU scores for each generated response.
    """

    # FIX: Ensure the score columns exist before trying to access them.
    if 'rouge_score' not in results_df.columns:
        results_df['rouge_score'] = pd.NA
    if 'bleu_score' not in results_df.columns:
        results_df['bleu_score'] = pd.NA

    print("\n--- Calculating Quantitative Metrics (ROUGE & BLEU) ---")
    
 
    rouge_evaluator = evaluate.load('rouge')
    bleu_evaluator = evaluate.load('sacrebleu')

    # Create a lookup for ground truth from the reference DataFrame
    #reference_lookup = {row['name']: row['bio'] for _, row in reference_df.iterrows()}
    if dataset_type == "structured" or dataset_type == "unstructured":
        reference_lookup = {row['name']: row['bio'] for _, row in reference_df.iterrows()}
    else: #summarized
        reference_lookup = {row['name']: row['bio_summary'] for _, row in reference_df.iterrows()}

    # This list will hold the calculated scores for each model
    score_records = []

    # Iterate over each unique model to ensure scores are calculated per model
    for model_key in results_df['model_key'].unique():

        # Skip if scores already exist for this model
        if pd.notna(results_df[(results_df['model_key'] == model_key) & 
                               (results_df['lora_rank'] == lora_rank) & 
                               (results_df['dataset_type'] == dataset_type)]['rouge_score']).any():
            print(f"ROUGE/BLEU scores for {model_key} already exist. Skipping.")
            continue

        model_df = results_df[results_df['model_key'] == model_key]
        predictions = []
        references = []
        
        for _, row in model_df.iterrows():
            char_name = row['character_name']
            # Safely get the biography and reference, defaulting to an empty string if null
            generated_bio = str(row['biography']) if pd.notnull(row['biography']) else ""
            reference_bio = reference_lookup.get(char_name, None)

            # Only append if both the generated biography and the reference exist and are not empty
            if reference_bio and generated_bio:
                predictions.append(generated_bio)
                references.append([reference_bio])
     
     
        if predictions:
            rouge_result = rouge_evaluator.compute(predictions=predictions, references=references)
            bleu_result = bleu_evaluator.compute(predictions=predictions, references=references)
            
            score_records.append({
                'model_key': model_key,
                'lora_rank': lora_rank,
                'dataset_type': dataset_type,
                'rouge_score': rouge_result['rougeL'],
                'bleu_score': bleu_result['score']
            })
    
    if score_records:
        # Convert scores to a temporary DataFrame
        scores_to_merge = pd.DataFrame(score_records)
        # Merge the new scores back into the main DataFrame
        results_df = pd.merge(results_df, scores_to_merge, 
                               on=['model_key', 'lora_rank', 'dataset_type'], 
                               how='left', suffixes=('', '_new'))
        # Update the original columns
        results_df['rouge_score'] = results_df['rouge_score'].fillna(results_df['rouge_score_new'])
        results_df['bleu_score'] = results_df['bleu_score'].fillna(results_df['bleu_score_new'])
        results_df = results_df.drop(columns=['rouge_score_new', 'bleu_score_new'])

    print("ROUGE/BLEU calculation complete.")
    return results_df



def get_judge_prompt(question, response_pairs, ground_truth_data, perform_reasoning):
    """
    Creates the prompt for the LLM-as-a-Judge,
    including ground truth and instructions for field-specific evaluation.
    """
    # Extract ground truth fields
    gt_bio = ground_truth_data.get('bio', 'Unknown')
    gt_home_city = ground_truth_data.get('home_city', 'Unknown')
    gt_race = ground_truth_data.get('race', 'Unknown')
    gt_gender = ground_truth_data.get('gender', 'Unknown')
    gt_location = ground_truth_data.get('location', 'Unknown')
    # Assuming 'predicted_action' is also in ground_truth_data for evaluation
    gt_predicted_action = ground_truth_data.get('predicted_action', 'Unknown')

    prompt_header = (
        "You are an expert evaluator of Large Language Models, specializing in creative writing and game lore analysis. "
        "Your task is to compare the following responses to a specific question about the game Skyrim. "
        "Each response is provided in a structured format containing 'character_name', 'home_city','race', 'gender', 'location', and 'biography'. "
        "Evaluate each response based on the following criteria:\n"
        "1. **Factual Accuracy of 'Home City', 'Race', 'Gender', 'Location'**: Compare these fields to the ground truth provided. Mark 'Correct' or 'Incorrect' for each. \n"
        "2. **Quality of 'Biography'**: How well does the 'biography' description align with the character's overall biography and established Skyrim lore, and how well does it address the part of the question related to biography? Assess its detail, accuracy, and coherence.\n"
        "3. **Overall Response Coherence and Completeness**: Does the entire response make sense? Are all requested fields present and well-formatted? Is the tone and language consistent with Elder Scrolls lore and the ground truth?\n"
        "4. **Overall Best Response**: Which model provides the most accurate, coherent, and complete response overall for this character?\n\n"
        f"--- Ground-Truth Character Data ---\n"
        f"Biography: {gt_bio}\n"
        f"Home City: {gt_home_city}\n"
        f"Race: {gt_race}\n"
        f"Gender: {gt_gender}\n"
        f"Location: {gt_location}\n"
        # We might also add the ground truth predicted action here if the question implies it
        # f"Expected Predicted Action: {gt_predicted_action}\n" # Only if the question asks for predicted action directly
        f"--- End Ground-Truth ---\n\n"
        f"The original question asked was: \"{question}\"\n\n"
        "Here are the responses from the models:\n"
    )
    responses_text = ""
    for i, (model_key, response_json_str) in enumerate(response_pairs):
        try:
            parsed_response = json.loads(response_json_str)
            gen_char_name = parsed_response.get('character_name', 'N/A')
            gen_home_city = parsed_response.get('home_city', 'N/A')
            gen_race = parsed_response.get('race', 'N/A')
            gen_gender = parsed_response.get('gender', 'N/A')
            gen_location = parsed_response.get('location', 'N/A')
            gen_biography = parsed_response.get('biography', 'N/A')

            responses_text += f"--- Response from Model '{model_key}' ---\n"
            responses_text += f"Character Name: {gen_char_name}\n"
            responses_text += f"Home City: {gen_home_city}\n"
            responses_text += f"Race: {gen_race}\n"
            responses_text += f"Gender: {gen_gender}\n"
            responses_text += f"Location: {gen_location}\n"
            responses_text += f"Biography: {gen_biography}\n\n"

        except (json.JSONDecodeError, Exception):
            responses_text += f"--- Response from Model '{model_key}' (JSON Decode Error) ---\n{response_json_str}\n\n"
            # The judge model should be robust enough to handle this, but it's good to indicate.
    # City: ... \n Race: ... \n Gender: ... \n Location: ... \n Biography:
    if perform_reasoning:
        # Requesting detailed reasoning and individual field judgments
        prompt_footer = (
            "Please perform a detailed evaluation of each response based on the criteria. "
            "For each model's response, explicitly state 'Home City: Correct' or 'Home City: Incorrect', 'Race: Correct' or 'Race: Incorrect', 'Gender: Correct' or 'Gender: Incorrect', and 'Location: Correct' or 'Location: Incorrect' by comparing them to the Ground-Truth. "
            "Then, provide an biography quality score for each model's response based on the quality, correctness based on ground truth, and coherence of the 'Biography', and also provide the overall coherence and quality.\n"
            "Finally, provide an overall ranking of the models from best to worst.\n"
            "Your final output MUST be a single JSON object with the following keys: "
            "'reasoning': (string, your detailed reasoning for all judgments), "
            "'model_evaluations': (list of objects, one for each model, with keys 'model_key', 'home_city_correct', 'race_correct', 'gender_correct', 'location_correct', 'biography_quality_score' (1-5), "
            "'overall_ranking': (list of model keys, ordered from best to worst)."
            "\n\nJSON Output:"
        )
    else:
        # Requesting just the factual judgments and overall ranking
        prompt_footer = (
            "For each model's response, indicate 'Correct' or 'Incorrect' for 'Home City', 'Correct' or 'Incorrect' for 'Race', 'Gender', and 'Location' based on the Ground-Truth. "
            "Then, provide an biography quality score for each model's response based on the quality, correctness based on ground truth, and coherence of the 'Biography', and also provide the overall coherence and quality.\n"
            "Finally, provide an overall ranking of the models from best to worst based on all criteria.\n"
            "Your final output MUST be a single JSON object with the following keys: "
            "'model_evaluations': (list of objects, one for each model, with keys 'model_key', 'home_city_correct', 'race_correct', 'gender_correct', 'location_correct', 'biography_quality_score' (1-5),  "
            "'overall_ranking': (list of model keys, ordered from best to worst)."
            "\n\nJSON Output:"
        )
    return prompt_header + responses_text + prompt_footer

def run_llm_as_judge_evaluation(results_df, reference_df):
    """Uses a powerful LLM (local or Gemini) to judge the quality of responses."""
    print("\n--- Performing LLM-as-a-Judge Evaluation ---")
    global CURRENT_GEMINI_KEY_INDEX
    global PERFORM_LLM_AS_JUDGE_EVAL
    global judge_model_instance, judge_tokenizer
    global FORCE_REGENERATE_RANK_JUDGE 

    if not PERFORM_LLM_AS_JUDGE_EVAL:
        print("Skipping LLM-as-a-Judge evaluation as per configuration.")
        return results_df


    # <<< --- CHANGED: Define column names based on the judge type --- >>>
    if USE_GGUF_JUDGE:
        print("Using GGUF Judge. Results will be saved to '_2' columns.")
        judge_cols = {
            'score': 'overall_model_score_2',
            'reasoning': 'judge_reasoning_2',
            'ranking': 'overall_ranking_for_char_2',
            'city': 'home_city_correct_2',
            'race': 'race_correct_2',
            'gender': 'gender_correct_2',
            'location': 'location_correct_2',
            'bio': 'biography_quality_score_2'
        }
    else:
        print("Using Gemini Judge. Results will be saved to standard columns.")
        judge_cols = {
            'score': 'overall_model_score',
            'reasoning': 'judge_reasoning',
            'ranking': 'overall_ranking_for_char',
            'city': 'home_city_correct',
            'race': 'race_correct',
            'gender': 'gender_correct',
            'location': 'location_correct',
            'bio': 'biography_quality_score'
        }

    # Remove duplicates from the reference dataframe, keeping the first entry
    reference_df.drop_duplicates(subset=['name'], keep='first', inplace=True)

    # Create a lookup dictionary for full ground truth data by character name
    reference_data_lookup = reference_df.set_index('name').to_dict('index')


    if judge_model_instance is None:
        if USE_GEMINI_JUDGE:
            if not GEMINI_API_KEYS:
                raise ValueError("GEMINI_API_KEY is not set for Gemini judge. Please set it in config or environment.")
            genai.configure(api_key=GEMINI_API_KEYS[CURRENT_GEMINI_KEY_INDEX])
            if judge_model_instance is None:
                judge_model_instance = genai.GenerativeModel(JUDGE_GEMINI_MODEL_ID)
            print("Using Google Gemini as the judge model.")
        elif USE_GGUF_JUDGE:
            from llama_cpp import Llama as Llama_cpp
            if judge_model_instance is None:
                judge_model_instance = Llama_cpp(
                    model_path=JUDGE_MODEL_PATH,
                    n_gpu_layers=JUDGE_GGUF_N_GPU_LAYERS,
                    n_ctx=JUDGE_GGUF_N_CTX,
                    n_threads=24,
                    flash_attn=True, # Enable Flash Attention 2.
                    use_mmap=False, # Disable memory mapping.
                    use_mlock=False, # Disable memory locking.
                    offload_kvs=False, # Disable KV offloading.
                    # Corresponds to --tensor-split 30,24,24
                    tensor_split = TENSOR_SPLIT,
                    verbose=False, # Set to True for llama.cpp output during loading
                )
            print(f"Using GGUF model '{JUDGE_MODEL_PATH}' as the judge model.")
        else:
            # Initialize local LLM judge
            if USE_UNSLOTH_JUDGE:
                judge_model_instance, judge_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=JUDGE_MODEL_PATH, token=HUGGING_FACE_TOKEN, device_map="auto", max_seq_length=4096, dtype=None, load_in_4bit=JUDGE_LOAD_IN_4BIT)
                FastLanguageModel.for_inference(judge_model_instance)
            else:
                if JUDGE_LOAD_IN_4BIT:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                    judge_model_instance = AutoModelForCausalLM.from_pretrained(
                        JUDGE_MODEL_PATH, device_map="auto", token=HUGGING_FACE_TOKEN, torch_dtype=torch.bfloat16,
                        quantization_config=quantization_config, trust_remote_code=True,)
                else:
                    judge_model_instance = AutoModelForCausalLM.from_pretrained(
                        JUDGE_MODEL_PATH, device_map="auto", load_in_4bit=False, token=HUGGING_FACE_TOKEN,
                        torch_dtype=torch.bfloat16, trust_remote_code=True,)
                judge_model_instance.eval()
                judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_PATH, trust_remote_code=True)
            if judge_tokenizer.pad_token is None:
                judge_tokenizer.pad_token = judge_tokenizer.eos_token
            if judge_tokenizer.chat_template is None and chat_template:
                judge_tokenizer.chat_template = chat_template
            print(f"Using local LLM '{JUDGE_MODEL_PATH}' as the judge model.")
 
     # <<< --- CHANGED: Use dynamic column name --- >>>
    if judge_cols['score'] not in results_df.columns:
        results_df[judge_cols['score']] = pd.NA

    # ========================= START OF FIX =========================
    # Identify characters that have at least one model needing re-evaluation.
    incomplete_characters = results_df.loc[
        results_df[judge_cols['score']].isnull(),
        'character_name'
    ].unique()
    

    if incomplete_characters.any():
        print(f"Detected {len(incomplete_characters)} characters with incomplete judge results. Resetting scores for these characters.")
        
        judge_score_cols_list = list(judge_cols.values())
        results_df.loc[results_df['character_name'].isin(incomplete_characters), judge_score_cols_list] = np.nan
        
    judgments_to_run = results_df[
        (results_df[judge_cols['score']].isnull())
    ].copy()
    
    # <<< --- CHANGED: Logic to include forced ranks --- >>>
    condition = results_df[judge_cols['score']].isnull()
    
    if FORCE_REGENERATE_RANK_JUDGE:
        force_ranks_list = str(FORCE_REGENERATE_RANK_JUDGE).split(',')
        # Check if the current rows match the forced rank. 
        # Convert column to string to ensure matching works against split strings
        rank_match = results_df['lora_rank'].astype(str).isin(force_ranks_list)
        if rank_match.any():
            print(f"Forcing regeneration for {rank_match.sum()} rows matching rank(s) {FORCE_REGENERATE_RANK_JUDGE}")
            condition = condition | rank_match

    judgments_to_run = results_df[condition].copy()

    if judgments_to_run.empty:
        print("All judgments already computed. Skipping LLM-as-a-Judge.")
        return results_df

    # Validate required keys
    required_eval_keys = [
        'model_key',  'biography_quality_score',
        'home_city_correct', 'race_correct', 'gender_correct', 'location_correct'
    ]
    
    # Group the remaining tasks by character_name
    grouped_tasks = judgments_to_run.groupby('character_name')
    stop_execution = False # Flag to stop execution if any error occurs

    for character_name, group in tqdm(grouped_tasks, desc="Judging responses"):
        if len(group) < 2: 
            print(f"Skipping {character_name}: Not enough models (less than 2) to compare.")
            continue # Need at least two models to compare
        
        if stop_execution:
            print("Stopping execution.")
            break

        ground_truth_data = reference_data_lookup.get(character_name, {})
        if not ground_truth_data:
            print(f"Warning: No ground truth data found for {character_name}. Skipping judgment.")
            continue
        
        question = group['question'].iloc[0]
        response_pairs = list(zip(group['model_key'], group['generated_response']))
        
        judge_prompt = get_judge_prompt(question, response_pairs, ground_truth_data, PERFORM_LLM_AS_JUDGE_EVAL_REASONING)
         
        parsed_judgment = {}
        judgment_text = ""
        temperature = 0.1
        for i in range(MAX_JUDGE_RETRIES): # Retry attempts for judge response
            #default: fixed temperature at 0.1
            temperature = 0.1
            try:
                if USE_GEMINI_JUDGE:
                    gemini_response = judge_model_instance.generate_content(judge_prompt)
                    # Access text from candidate, which is the generated content
                    judgment_text = gemini_response.text.strip()
                    # Generate a random integer between 1 and 3 (inclusive)
                    random_number = random.randint(1, 3)
                    time.sleep(random_number) # Basic rate limit avoidance
                else:
                    # Handle GGUF model inference
                    if USE_GGUF_JUDGE:
                        output = judge_model_instance(
                            judge_prompt,
                            max_tokens=1500, # Max tokens for judge response
                            temperature=temperature,
                        )
                        judgment_text = output["choices"][0]["text"].strip()
                    else: # Existing Hugging Face model inference
                        messages = [{"role": "user", "content": judge_prompt}]
                        prompt_string = judge_tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True)
                        inputs = judge_tokenizer(prompt_string, return_tensors="pt").to(judge_model_instance.device)
                        outputs = judge_model_instance.generate(**inputs, max_new_tokens=1500, use_cache=False) # Increased tokens for detailed reasoning
                        judgment_text = judge_tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
                # --- START OF MODIFIED PARSING LOGIC ---                
                try:
                    json_indicator = judgment_text.find('JSON Output:')
                    if json_indicator != -1:
                        judgment_text = judgment_text[json_indicator+12:]
                    #Remove reasoning <think> tags if present
                    if judgment_text.find('<think>') != -1:
                        judgment_text = re.sub(r'<think>.*?</think>', '', judgment_text, flags=re.DOTALL).strip()
                        # Also remove any leftover whitespace or empty lines
                        judgment_text = '\n'.join([line for line in judgment_text.split('\n') if line.strip()])

                    # Use regex to find the main JSON block.
                    json_start = judgment_text.find('{')
                    json_end = judgment_text.rfind('}') + 1
                    if json_end == 0:
                        json_end = len(judgment_text)  # If no closing brace found, use the end of the string
                    if json_start != -1 and json_end != 0:
                        json_str = judgment_text[json_start:json_end]
                        
                        # Use dirtyjson.loads() which is more tolerant of errors
                        parsed_judgment = dirtyjson.loads(json_str)
                        
                        # Get the list of model keys we EXPECT to find (from the input pairs)
                        expected_model_keys = set([pair[0] for pair in response_pairs])
                        
                        is_valid = False
                        
                        # 1. Structure Check
                        if ('overall_ranking' in parsed_judgment and 
                            isinstance(parsed_judgment['overall_ranking'], list) and
                            'model_evaluations' in parsed_judgment and 
                            isinstance(parsed_judgment['model_evaluations'], list)):
                            
                            # 2. Key Matching Check (Crucial for preventing infinite re-runs)
                            # Get keys returned by the judge
                            generated_model_keys = set([item.get('model_key') for item in parsed_judgment['model_evaluations']])
                            
                            # Check if the judge returned exactly the models we asked for
                            if expected_model_keys == generated_model_keys:
                                
                                # 3. Content Completeness Check
                                # Ensure every evaluation object has ALL required fields
                                if all(all(k in item for k in required_eval_keys) for item in parsed_judgment['model_evaluations']):
                                    is_valid = True
                                else:
                                    print(f"Warning: Missing specific fields in judge output (e.g. bio score or correctness). Retrying...")
                            else:
                                print(f"Warning: Model Key mismatch. Expected {expected_model_keys}, got {generated_model_keys}. Retrying...")

                        if is_valid:
                            break # Successfully parsed and validated
                        else:
                            print(f"Warning: Judge output missing required keys for {character_name}. Retrying...")
                            parsed_judgment = {} # Reset to retry
                    else:
                        print(f"Warning: Judge did not output valid JSON for {character_name}. Retrying...")
                        parsed_judgment = {} # Reset to retry

                except (dirtyjson.Error, AttributeError, KeyError, Exception) as e:
                    print(f"Error parsing judge response for {character_name} (attempt {i+1}): {e}.")
                    parsed_judgment = {} # Reset to retry
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                print(f"Error parsing judge response for {character_name} (attempt {i+1}): {e}. Raw response:\n{judgment_text}")
                parsed_judgment = {} # Reset to retry
            except Exception as e:
                print(f"General error during judge inference for {character_name} (attempt {i+1}): {e}")
                parsed_judgment = {} # Reset to retry
                if str(e).find('PROHIBITED_CONTEN') != -1:
                    print(f"Prohibited content error detected in judge response for {character_name}. Skipping further attempts.")
                    break # No point in retrying if content is prohibited
                

            #we are goin to assume that Gemini wont make mistakes for now
            if i == MAX_JUDGE_RETRIES-1: # After last attempt
                print(f"Failed to get valid judge response for {character_name} after multiple retries. Skipping.")
                parsed_judgment = {} # Ensure it's empty if all retries fail
                if USE_GEMINI_JUDGE:
                    stop_execution = True
                    PERFORM_LLM_AS_JUDGE_EVAL = False
                    print("Stopping further execution of LLM as Judge due to repeated judge failures using gemini.")
                break
            elif judgment_text.find('PROHIBITED_CONTENT') != -1:
                print(f"Prohibited content detected in judge response for {character_name}. Skipping.")
                parsed_judgment = {} # Ensure it's empty if prohibited content is found
                break

            if  USE_GEMINI_JUDGE:
                print(f"Fail using google API. Trying a new key and waiting a few seconds.")
                CURRENT_GEMINI_KEY_INDEX += 1
                if CURRENT_GEMINI_KEY_INDEX >= len(GEMINI_API_KEYS):
                    CURRENT_GEMINI_KEY_INDEX = 0 # Reset to first key if we run out
                wait_time = random.randint(5, 15)
                #wait_time = CURRENT_GEMINI_KEY_INDEX*15 # Increase wait time with each key switch
                print(f"Waiting {wait_time} seconds before retrying with new key.")
                time.sleep(wait_time) # Basic rate limit avoidance
                #Lets try the next key
                print(f"Switching to Gemini API key index {CURRENT_GEMINI_KEY_INDEX}. Retrying with new key.")        
                genai.configure(api_key=GEMINI_API_KEYS[CURRENT_GEMINI_KEY_INDEX])
                judge_model_instance = genai.GenerativeModel(JUDGE_GEMINI_MODEL_ID) # Reinitialize with new key
                continue # Retry with the new key
        if parsed_judgment:
            overall_ranking = parsed_judgment.get('overall_ranking', [])
            model_evaluations = parsed_judgment.get('model_evaluations', [])
            reasoning = parsed_judgment.get('reasoning', '') if PERFORM_LLM_AS_JUDGE_EVAL_REASONING else ''

            # <<< --- CHANGED: Use dynamic column names for updating --- >>>
            for model_eval in model_evaluations:
                model_key = model_eval.get('model_key', 'N/A')
                # Find the row in the main results_df to update
                update_idx = results_df[
                    (results_df['character_name'] == character_name) &
                    (results_df['model_key'] == model_key)
                ].index
                if not update_idx.empty:
                    results_df.loc[update_idx, judge_cols['reasoning']] = reasoning
                    results_df.loc[update_idx, judge_cols['ranking']] = json.dumps(overall_ranking) # Store as JSON string
                    results_df.loc[update_idx, judge_cols['city']] = model_eval.get('home_city_correct', None)
                    results_df.loc[update_idx, judge_cols['race']] = model_eval.get('race_correct', None)
                    results_df.loc[update_idx, judge_cols['gender']] = model_eval.get('gender_correct', None)
                    results_df.loc[update_idx, judge_cols['location']] = model_eval.get('location_correct', None)
                    results_df.loc[update_idx, judge_cols['bio']] = model_eval.get('biography_quality_score', None)
                    results_df.loc[update_idx, judge_cols['score']] = "1" #Used here just to mark as completed. It Will be calculated later.
                    

        else:
            print(f"No valid judgment parsed for {character_name}. Skipping update for this character.")
            pass # The original null values will remain, so this is correct behavior.
 
 


    # Clean up GPU memory for local judge
    if not USE_GEMINI_JUDGE and judge_model_instance is not None and not USE_GGUF_JUDGE:
        del judge_model_instance, judge_tokenizer
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: error clearing GPU cache: {e}")
    # Now, all the judgment scores are directly in the results_df
    # We will compute the aggregate scores after this function returns

    print("LLM-as-a-Judge evaluation complete.")
    return results_df

# ===================================================================================
# --- 7. Main Execution ---
# ===================================================================================

def main():
    """Main function to run the entire evaluation pipeline."""
    # Initialize the FileLocker for the main output CSV
    locker = FileLocker(FINAL_OUTPUT_CSV)
    
    # NEW: Load the main evaluation DataFrame ONCE before the loops.
    locker.acquire_lock()
    try:
        if os.path.exists(FINAL_OUTPUT_CSV):
            final_evaluation_df = pd.read_csv(FINAL_OUTPUT_CSV)
            print(f"Found existing evaluation results at '{FINAL_OUTPUT_CSV}'. Loading to resume evaluation.")
            
            # ==============================================================================
            # <<< --- NEW LOGIC: Smart Force Regenerate (Rank/Type Aware) --- >>>
            # ==============================================================================
            if args.force_regenerate_model:
                model_to_purge = args.force_regenerate_model
                initial_len = len(final_evaluation_df)
                
                # 1. Determine which Ranks and Datasets are actually active in this run
                # (We only want to delete results that we are about to regenerate)
                active_ranks = [r for r in LORA_RANKS_TO_EVALUATE if str(r) not in ignore_lora_rank]
                active_datasets = [d for d in DATASET_TYPES_TO_EVALUATE if d not in ignore_dataset_type]

                # 2. Ensure columns exist to avoid KeyErrors
                cols_to_check = ['model_key', 'lora_rank', 'dataset_type']
                for col in cols_to_check:
                    if col not in final_evaluation_df.columns:
                        final_evaluation_df[col] = pd.NA

                # 3. Create a mask for rows to delete
                # Condition: Model matches AND Rank is active AND Dataset Type is active
                rows_to_delete_mask = (
                    (final_evaluation_df['model_key'] == model_to_purge) &
                    (final_evaluation_df['lora_rank'].isin(active_ranks)) &
                    (final_evaluation_df['dataset_type'].isin(active_datasets))
                )

                # 4. Filter the DataFrame (Keep rows that are NOT in the mask)
                final_evaluation_df = final_evaluation_df[~rows_to_delete_mask]
                
                removed_count = initial_len - len(final_evaluation_df)
                
                if removed_count > 0:
                    print(f"Force regeneration active: Removed {removed_count} rows for model '{model_to_purge}'")
                    print(f"   (Restricted to Active Ranks: {active_ranks} and Active Types: {active_datasets})")
                    # Save immediately so the purge persists even if script crashes
                    final_evaluation_df.to_csv(FINAL_OUTPUT_CSV, index=False)
                else:
                    print(f"Force regeneration active: No matching rows found for model '{model_to_purge}' in the current configuration scope.")
            # ==============================================================================
        else:
            final_evaluation_df = pd.DataFrame()
            print(f"No existing evaluation results found at '{FINAL_OUTPUT_CSV}'. Starting from scratch.")
    finally:
        locker.release_lock()

    # Define columns that are products of evaluation
    evaluation_cols = [
        'perplexity', 'rouge_score', 'bleu_score', 
        'judge_reasoning', 'overall_ranking_for_char', 'home_city_correct',
        'race_correct', 'gender_correct', 'location_correct', 
        'biography_quality_score', 'overall_model_score',
        'judge_reasoning_2', 'overall_ranking_for_char_2', 'home_city_correct_2',
        'race_correct_2', 'gender_correct_2', 'location_correct_2', 
        'biography_quality_score_2', 'overall_model_score_2'
    ]
    # Also define the configuration columns that are used for filtering
    config_cols = ['character_name', 'model_key', 'biography', 'lora_rank', 'dataset_type', 'tier']

    # Ensure ALL necessary columns exist in the main df to avoid KeyErrors
    for col in evaluation_cols + config_cols:
        if col not in final_evaluation_df.columns:
            final_evaluation_df[col] = pd.NA

    for lora_rank in LORA_RANKS_TO_EVALUATE:
        if str(lora_rank) in ignore_lora_rank:
            print(f"Skipping LoRA rank '{lora_rank}' as it is set to be ignored.")
            continue
        
        for dataset_type in DATASET_TYPES_TO_EVALUATE:
            if dataset_type in ignore_dataset_type:
                print(f"Skipping dataset type '{dataset_type}' as it is set to be ignored.")
                continue
            for tier, model_catalog in ALL_MODEL_CATALOGS.items():
                print(f"\n--- Starting Evaluation for Rank: {lora_rank}, Dataset: {dataset_type}, Tier: {tier} ---")

                if str(tier) in ignore_tier:
                    print(f"Skipping tier '{tier}' as it is set to be ignored.")
                    continue

                # Check if this specific configuration has already been fully processed
                is_completed = False
                if not final_evaluation_df.empty and tier in final_evaluation_df['tier'].unique():
                    models_in_tier = list(model_catalog.keys())
                    config_df = final_evaluation_df[
                        (final_evaluation_df['lora_rank'] == lora_rank) &
                        (final_evaluation_df['dataset_type'] == dataset_type) &
                        (final_evaluation_df['tier'] == tier)
                    ]
                    
                    if len(config_df) > 0:
                        all_models_present = set(models_in_tier).issubset(set(config_df['model_key'].unique()))
                        if USE_GGUF_JUDGE:
                            score_cols = [col for col in ['overall_model_score_2', 'perplexity', 'rouge_score', 'bleu_score'] if col in config_df.columns]
                        else:
                            score_cols = [col for col in ['overall_model_score', 'perplexity', 'rouge_score', 'bleu_score'] if col in config_df.columns]
                        all_scores_present = config_df[score_cols].notnull().all().all()

                        valid_perplexity = True
                        if 'perplexity' in config_df.columns:
                            # Coerce to numeric to handle any string weirdness safely
                            ppl_check = pd.to_numeric(config_df['perplexity'], errors='coerce')
                            # If any model in this tier has a perplexity > 100, mark as NOT completed
                            if (ppl_check > 1000).any():
                                valid_perplexity = False
                                print(f"Detected invalid perplexity (>100) in saved results for {tier} {dataset_type}. Forcing re-evaluation.")
                        # ==============================================================================
                        
                        if all_models_present and all_scores_present and valid_perplexity:
                            is_completed = True

                if is_completed and FORCE_REGENERATE_RANK_JUDGE != str(lora_rank):
                    print(f"Configuration (Rank: {lora_rank}, Dataset: {dataset_type}, Tier: {tier}) already completed. Skipping.")
                    continue  # Skip to the next iteration of the loop
                
                # Dynamically construct the path to the inference results CSV
                INFERENCE_RESULTS_CSV_DYNAMIC = f"results_{TARGET_FOLDER}/evaluation_results_R{lora_rank}_{tier}_{dataset_type}.csv"
                print(f"Loading inference results from '{INFERENCE_RESULTS_CSV_DYNAMIC}'")
                
                if not os.path.exists(INFERENCE_RESULTS_CSV_DYNAMIC) or not os.path.exists(REFERENCE_DATASET_CSV):
                    print(f"Warning: Required input file not found. Skipping evaluation for this configuration. "
                          f"Check paths for '{INFERENCE_RESULTS_CSV_DYNAMIC}' and '{REFERENCE_DATASET_CSV}'.")
                    continue

                results_df = pd.read_csv(INFERENCE_RESULTS_CSV_DYNAMIC)
                reference_df = pd.read_csv(REFERENCE_DATASET_CSV)
                print(f"Loaded {len(results_df)} generated responses from {len(results_df['model_key'].unique())} models.")
                
                models_in_tier = list(model_catalog.keys())
                results_df_filtered = results_df[results_df['model_key'].isin(models_in_tier)].copy()

                if results_df_filtered.empty:
                    print(f"No results found for models in tier {tier}. Skipping.")
                    continue

                results_df_filtered['lora_rank'] = lora_rank
                results_df_filtered['dataset_type'] = dataset_type
                results_df_filtered['tier'] = tier
                
                # Merge existing scores from the main results file into the current working DataFrame.
                if not final_evaluation_df.empty:
                    merge_keys = ['character_name', 'model_key', 'lora_rank', 'dataset_type', 'tier']
                    existing_scores_df = final_evaluation_df[merge_keys + evaluation_cols].copy()
                    
                    results_df_filtered = pd.merge(
                        results_df_filtered,
                        existing_scores_df,
                        on=merge_keys,
                        how='left'
                    )

                results_with_ppl = calculate_perplexity(results_df_filtered, reference_df, lora_rank, dataset_type, model_catalog)
                results_with_rouge_bleu = calculate_rouge_bleu(results_with_ppl, reference_df, lora_rank, dataset_type, model_catalog)
                
                final_results_for_run = run_llm_as_judge_evaluation(results_with_rouge_bleu, reference_df)

                final_results_for_run['lora_rank'] = lora_rank
                final_results_for_run['dataset_type'] = dataset_type
                final_results_for_run['tier'] = tier

                # Use the lock to safely update rows in-place
                locker.acquire_lock()
                try:
                    # Define the key columns that uniquely identify each row
                    key_cols = ['character_name', 'model_key', 'lora_rank', 'dataset_type', 'tier']
                    # Check if the file already exists
                    if os.path.exists(FINAL_OUTPUT_CSV):
                        # Load the existing main DataFrame
                        main_df = pd.read_csv(FINAL_OUTPUT_CSV)
                        
                        # Combine the old data with the new results
                        combined_df = pd.concat([main_df, final_results_for_run], ignore_index=True)
                        
                        # Remove duplicate rows based on the keys, keeping the *last* (most recent) entry
                        combined_df.drop_duplicates(subset=key_cols, keep='last', inplace=True)
                        
                        # Save the updated DataFrame
                        combined_df.to_csv(FINAL_OUTPUT_CSV, index=False)
                        print(f"Incremental save complete. Results for (Rank: {lora_rank}, Dataset: {dataset_type}, Tier: {tier}) merged into '{FINAL_OUTPUT_CSV}'.")
                    else:
                        # If the file does NOT exist, this is the first write. Save the current results directly.
                        final_results_for_run.to_csv(FINAL_OUTPUT_CSV, index=False)
                        print(f"Created new results file. Results for (Rank: {lora_rank}, Dataset: {dataset_type}, Tier: {tier}) saved to '{FINAL_OUTPUT_CSV}'.")
                except Exception as e:
                    print(f"Error during locked file write: {e}")
                finally:
                    locker.release_lock()

    print(f"\nEvaluation pipeline complete. All results saved to '{FINAL_OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()
