# ===================================================================================
# Advanced LoRa Fine-Tuning Script for Skyrim Character Dataset
# ===================================================================================

# --- 1. Install Unsloth ---

import os
import glob
import shutil # <<< ADD THIS IMPORT
# --- 2. Import Necessary Libraries ---
import warnings
import argparse
import wandb  # Optional: For Weights & Biases integration


#Default values
DEFAULT_CUDA_VISIBLE_DEVICES = None #Values: None, "0,1,2", "0", "1", "2"
DEFAULT_SELECTED_MODEL_KEY = "llama_8b" # Values: "deepseek_7b", "llama_8b", "qwen_8b", "deepseek_16b", "llama2_13b", 
                                           # "qwen_14b", "deepseek_33b", "codellama_34b", "qwen_32b"
DEFAULT_LORA_RANK = 32 # Values: 16, 32, 64
DEFAULT_DATASET_TYPE = "summarized"  #Values: summarized, unstructured, structured,
DEFAULT_TARGET_FOLDER = "seed_1"#"


DEFAULT_EARLY_STOPPPING_PATIENCE = 4 #CHANGE ME BACK TO 4!
DEFAULT_SAVE_EVAL_STEPS = 20  # Save and evaluate every 20 steps. 

DEFAULT_DATASET_TYPE_EXTRA = "" #Values: "","_example"
DEFAULT_FORCE_REGENERATE_LORA = False #Values: True, False
DEFAULT_CONTINUE_FROM_CHECKPOINT = False # only valid for FORCE_REGENERATE_LORA = False
DEFAULT_OVERWRITE_CHECKPOINT_PATH = None
DEFAULT_LOW_VRAM = False
DEFAULT_NUM_TRAIN_EPOCHS = 100 #it has early stopping, so it will stop when the validation loss stops improving
DEFAULT_SAVE_PRETRAINED_MODEL = False
DEFAULT_SEED = 4315468 # Define the default seed
DEFAULT_USE_UNSLOTH = True 

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Advanced LoRa Fine-Tuning Script for Skyrim Character Dataset")


# Required arguments
parser.add_argument("--model_key", type=str, default=DEFAULT_SELECTED_MODEL_KEY,
                    help="Key for the model to use (e.g., 'deepseek_7b', 'llama_8b').")
parser.add_argument("--lora_rank", type=int, default=DEFAULT_LORA_RANK, choices=[16, 32, 64],
                    help="Rank for the LoRA adapter (e.g., 16, 32, 64).")
parser.add_argument("--dataset_type", type=str, default=DEFAULT_DATASET_TYPE, choices=["structured", "unstructured", "summarized"],
                    help="Type of dataset to use: 'structured', 'unstructured', or 'summarized'.")
# Optional arguments with default values
parser.add_argument("--cuda_devices", type=str, default=DEFAULT_CUDA_VISIBLE_DEVICES,
                    help="Comma-separated list of CUDA device IDs to make visible (e.g., '0,1,2'). If not provided, os.environ['CUDA_VISIBLE_DEVICES'] will not be set by the script.")
parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS,
                    help="Number of training epochs. Defaults to 100.")
parser.add_argument("--save_eval_steps", type=int, default=DEFAULT_SAVE_EVAL_STEPS,
                    help="Number of steps after which to save a checkpoint and evaluate. Defaults to 100.")
parser.add_argument("--dataset_type_extra", type=str, default=DEFAULT_DATASET_TYPE_EXTRA,
                    help="Extra suffix for the dataset filename (e.g., '_example'). Defaults to ''.")
parser.add_argument("--force_regenerate_lora", action="store_true", default=DEFAULT_FORCE_REGENERATE_LORA,
                    help="If set, forces regeneration of the LoRA adapter, ignoring existing checkpoints. Defaults to False.")
parser.add_argument("--continue_from_checkpoint", action="store_true", default=DEFAULT_CONTINUE_FROM_CHECKPOINT,
                    help="If set to True, attempts to continue training from the latest checkpoint. Defaults to False.")
parser.add_argument("--overwrite_checkpoint_path", type=str, default=DEFAULT_OVERWRITE_CHECKPOINT_PATH,
                    help="Path to a specific checkpoint to resume training from. Overrides automatic checkpoint detection. Defaults to None.")
parser.add_argument("--low_vram", action="store_true", default=DEFAULT_LOW_VRAM,
                    help="If set to True, attempts to use low VRAM settings. Defaults to False.")
parser.add_argument("--target_folder", type=str, default=DEFAULT_TARGET_FOLDER,
                    help="Target folder. ")  
parser.add_argument("--disable_unsloth", action="store_true", # <<< ADDED
                    help="If set, disables Unsloth and uses standard Transformers/Peft loading.")
parser.add_argument("--disable_quantization", action="store_true", 
                    help="If set, disables 4-bit quantization and loads the model in 16-bit precision. Defaults to False.")
parser.add_argument("--training_split_col", type=str, default="for_training_1",
                    help="The name of the column used to split data into training (1/True) and validation (0/False).")
parser.add_argument("--save_pretrained_model", action="store_true", default=DEFAULT_SAVE_PRETRAINED_MODEL,
                    help="If set, saves the pretrained model after training. Defaults to False.")
parser.add_argument("--vl", action="store_true", default=False,
                    help="If set to True, use val. loss. Defaults to False.")
parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for reproducibility. Defaults to 4315468.")
parser.add_argument("--original_target_folder", type=str, default=None,
                    help="The original target folder if resuming from a copied checkpoint (e.g., '0_03'). Defaults to None.")
args = parser.parse_args()

# Apply parsed arguments
if args.cuda_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    # If not provided, ensure it's not set by the script if it was previously
    if DEFAULT_CUDA_VISIBLE_DEVICES is None and "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


SELECTED_MODEL_KEY = args.model_key
LORA_RANK = args.lora_rank
DATASET_TYPE = args.dataset_type
NUM_TRAIN_EPOCHS = args.num_train_epochs
SAVE_EVAL_STEPS = args.save_eval_steps
DATASET_TYPE_EXTRA = args.dataset_type_extra
FORCE_REGENERATE_LORA = args.force_regenerate_lora
CONTINUE_FROM_CHECKPOINT = args.continue_from_checkpoint
OVERWRITE_CHECKPOINT_PATH = args.overwrite_checkpoint_path
LBM = args.vl
LOW_VRAM = args.low_vram
SEED = args.seed
TARGET_FOLDER = args.target_folder
save_pretrained_model = args.save_pretrained_model
ORIGINAL_TARGET_FOLDER = args.original_target_folder
USE_UNSLOTH = not args.disable_unsloth if DEFAULT_USE_UNSLOTH else False # <<< ADDED
DISABLE_QUANTIZATION = args.disable_quantization

# Ensure CONTINUE_FROM_CHECKPOINT logic is consistent with FORCE_REGENERATE_LORA
if FORCE_REGENERATE_LORA:
    CONTINUE_FROM_CHECKPOINT = False # If regenerating, don't continue from old checkpoint

# This will be the base directory for ALL training outputs (checkpoints, logs, final adapter).
# The trainer will create subdirectories like `checkpoint-XXX` inside this.
training_output_dir = f"results_{TARGET_FOLDER}/R{LORA_RANK}_{SELECTED_MODEL_KEY}_{DATASET_TYPE}"

# If OVERWRITE_CHECKPOINT_PATH is set, it means we are trying to resume from a specific checkpoint,
# so the training_output_dir should be set to the parent directory of that checkpoint
# (i.e., the base directory where checkpoints are saved).
if OVERWRITE_CHECKPOINT_PATH is not None:
    training_output_dir = os.path.dirname(OVERWRITE_CHECKPOINT_PATH) #parent directory of that checkpoin


# The path where the *final, best* LoRA adapter (weights only) will be saved for inference.
# This is separate from the full training checkpoints managed by the Trainer.
FINAL_LORA_SAVE_PATH = os.path.join(training_output_dir, "final_lora_adapter")

# Setting variables
if OVERWRITE_CHECKPOINT_PATH is not None and os.path.isdir(OVERWRITE_CHECKPOINT_PATH):
    # If a specific checkpoint path is provided, use it directly.
    # The user might be just using another path (no guarantees we wants to train again or continue training)
    TRAINING_DIRECTORY = OVERWRITE_CHECKPOINT_PATH
    print(f"Attempting to load from specified checkpoint: '{TRAINING_DIRECTORY}'")
elif os.path.isdir(training_output_dir):
    # Otherwise, look for the latest checkpoint within the determined training_output_dir.
    checkpoints = list(glob.glob(os.path.join(training_output_dir, "checkpoint-*")))
    if len(checkpoints) > 0:
        # Sort checkpoints from oldest to newest based on step number
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
        
        TRAINING_DIRECTORY = None # Reset
        for checkpoint_path in reversed(checkpoints):
            trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
            if os.path.isfile(trainer_state_file):
                # Found a valid checkpoint
                TRAINING_DIRECTORY = checkpoint_path
                print(f"✅ Found valid latest checkpoint: '{TRAINING_DIRECTORY}'")
                break # Stop searching
            else:
                # This checkpoint is corrupt, print a warning and attempt to delete
                print(f"⚠️ Found corrupt checkpoint (missing 'trainer_state.json'): '{checkpoint_path}'.")
                
                # --- NEW: Attempt to delete the corrupt folder ---
                try:
                    shutil.rmtree(checkpoint_path)
                    print(f"    🗑️  Successfully deleted corrupt folder: '{checkpoint_path}'")
                except Exception as e:
                    print(f"    ❌ ERROR: Could not delete corrupt folder '{checkpoint_path}'. Error: {e}")
                # --- END NEW ---
        
        # If after checking all, none were valid
        if TRAINING_DIRECTORY is None:
            print(f"INFO: No *valid* checkpoint found in '{training_output_dir}'. Starting new training.")
            CONTINUE_FROM_CHECKPOINT = False
            TRAINING_DIRECTORY = None
            
    else:
        print(f"INFO: No full checkpoint found in '{training_output_dir}'. Starting new training.")
        CONTINUE_FROM_CHECKPOINT = False # Reset flag if no checkpoint found
        TRAINING_DIRECTORY = None # Since didn't find the directory
else:
    if CONTINUE_FROM_CHECKPOINT:
        print(f"INFO: CONTINUE_FROM_CHECKPOINT is True but no checkpoint found in '{training_output_dir}'. Starting new training.")
        CONTINUE_FROM_CHECKPOINT = False # since we have no directory to being with
    TRAINING_DIRECTORY = None

# --- New logic to check progress difference from an original checkpoint ---
if ORIGINAL_TARGET_FOLDER is not None and TRAINING_DIRECTORY is not None:
    print(f"INFO: Original target folder '{ORIGINAL_TARGET_FOLDER}' provided. Checking checkpoint progress...")
    
    # 1. Construct the original directory path
    original_training_output_dir = f"results_{ORIGINAL_TARGET_FOLDER}/R{LORA_RANK}_{SELECTED_MODEL_KEY}_{DATASET_TYPE}"
    
    # 2. Find the latest checkpoint in the *original* directory
    ORIGINAL_TRAINING_DIRECTORY = None
    if os.path.isdir(original_training_output_dir):
        original_checkpoints = list(glob.glob(os.path.join(original_training_output_dir, "checkpoint-*")))
        if len(original_checkpoints) > 0:
            original_checkpoints.sort(key=lambda x: int(os.path.basename(x).split('-')[-1]))
            ORIGINAL_TRAINING_DIRECTORY = original_checkpoints[-1]
            print(f"    Found latest *original* checkpoint: '{ORIGINAL_TRAINING_DIRECTORY}'")
        else:
            print(f"    INFO: No checkpoints found in original directory '{original_training_output_dir}'. Skipping progress check.")
    else:
            print(f"    INFO: Original directory '{original_training_output_dir}' not found. Skipping progress check.")

    # 3. Compare step numbers if both checkpoints were found
    if ORIGINAL_TRAINING_DIRECTORY is not None:
        try:
            # Extract step numbers
            new_step_num = int(os.path.basename(TRAINING_DIRECTORY).split('-')[-1])
            original_step_num = int(os.path.basename(ORIGINAL_TRAINING_DIRECTORY).split('-')[-1])
            
            difference = new_step_num - original_step_num
            print(f"    Comparing new checkpoint ({new_step_num}) to original ({original_step_num}). Difference: {difference}")

            # 4. Apply the logic: If difference is >= 80, set CONTINUE_FROM_CHECKPOINT to False
            #    This will prevent DO_TRAINING from becoming True (if FORCE_REGENERATE_LORA is also False)
            if difference >= 80:
                print(f"    Difference ({difference}) is >= 80. Sufficient progress already made. Setting CONTINUE_FROM_CHECKPOINT to False to prevent re-run.")
                CONTINUE_FROM_CHECKPOINT = False
        except Exception as e:
            print(f"    WARNING: Could not parse step numbers from checkpoint directories. Skipping progress check. Error: {e}")


# Refined DO_TRAINING logic (either new or continuing):
# If we want to continue a checkpoint or to train a new one (forced or not) or we do not have the training directory
DO_TRAINING = FORCE_REGENERATE_LORA or CONTINUE_FROM_CHECKPOINT or TRAINING_DIRECTORY is None

if not DO_TRAINING:
    print("No training will be performed. Exiting script.")
    exit()

# The most straightforward way to conceptualize this is:
# We train if FORCE_REGENERATE_LORA is true.
# OR (if FORCE_REGENERATE_LORA is false) AND (we are either starting a new run or successfully found a checkpoint to resume)
if FORCE_REGENERATE_LORA:
    print("FORCE_REGENERATE_LORA is True. Training will proceed to generate a new LoRA adapter.")
elif CONTINUE_FROM_CHECKPOINT:
    print(f"Continuing the checkpoint at '{training_output_dir}'.")
elif TRAINING_DIRECTORY is not None:
    print("Skip training.")


# Import Unsloth's FastLanguageModel
import torch
if USE_UNSLOTH:
    from unsloth import FastLanguageModel
    from transformers import EarlyStoppingCallback, TrainingArguments
else:
    from transformers import EarlyStoppingCallback, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers import AutoTokenizer
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, get_peft_model # Make sure PeftModel is imported

from datasets import Dataset
from huggingface_hub import login
from unsloth import unsloth_train # For Unsloth's specific training function
from transformers import TrainerCallback, TrainerState, TrainerControl
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

class LossThresholdCallback(TrainerCallback):
    def __init__(self, loss_threshold=0.06):
        self.loss_threshold = loss_threshold

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called after each evaluation step. Checks if the evaluation loss is below the threshold.
        """
        if state.log_history and 'eval_loss' in state.log_history[-1]:
            if state.log_history[-1]['eval_loss'] < self.loss_threshold:
                print(f"Evaluation loss ({state.log_history[-1]['eval_loss']:.4f}) is below threshold ({self.loss_threshold}). Stopping training.")
                control.should_training_stop = True
                


NUM_GPUS = torch.cuda.device_count()


# --- Function to calculate necessary max_seq_length ---
def calculate_max_seq_length(dataset, tokenizer, formatting_function):
    """
    Calculates the maximum sequence length required by scanning all entries
    from the dataset file and applying the chosen prompt formatting.

    Args:
        dataset (Dataset): The Hugging Face dataset object.
        tokenizer (PreTrainedTokenizer): The tokenizer loaded for the model.
        formatting_function (callable): The function used to format each sample
                                        into the model's expected chat format.

    Returns:
        int: The maximum sequence length found across all formatted samples,
             with a buffer and rounded up for GPU efficiency.
    """
    max_length = 0
    print("Calculating optimal max_seq_length...")
    for i, sample in enumerate(dataset):
        # Apply the formatting function to get the text in the desired prompt format
        formatted_output = formatting_function(sample, tokenizer)
        text = formatted_output["text"]

        # Tokenize the formatted text and get the length
        # add_special_tokens=True ensures BOS/EOS tokens are included in length calculation
        tokenized_output = tokenizer(text, return_length=True, add_special_tokens=True)
        current_length = tokenized_output['length'][0] # Access the length from the returned dictionary

        if current_length > max_length:
            max_length = current_length
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} samples. Current max length: {max_length}")

    # Add a small buffer to the max_length to be safe (e.g., 10%)
    # Then, round up to the nearest multiple of 64 for GPU memory alignment/efficiency.
    buffered_max_length = int(max_length * 1.10)
    buffered_max_length = ((buffered_max_length + 63) // 64) * 64
    print(f"Calculated max_seq_length: {max_length} (raw) -> {buffered_max_length} (buffered and rounded)")
    return buffered_max_length


# --- 3. Setup Authentication and Configuration ---
HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"] if "HUGGING_FACE_TOKEN" in os.environ else ""
WANDB_API_KEY = None#"YOUR_WANDB_API_KEY"          # Optional: Replace with your W&B key

if HUGGING_FACE_TOKEN:
    login(token=HUGGING_FACE_TOKEN)

if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb_project_name = "skyrim-lora-comparative-analysis"
    os.environ["WANDB_PROJECT"] = wandb_project_name
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
else:
    wandb_project_name = None

warnings.filterwarnings("ignore")

# ===================================================================================
# --- 4. EXPERIMENT CONFIGURATION ---
# ===================================================================================

# "auto" works for most cases, but "sequential" if you want to use Unsloth's sequential loading.
DEVICE_MAP = "auto"#"auto"#"sequential" if USE_UNSLOTH else "auto"





if "_7b" in SELECTED_MODEL_KEY or "_8b" in SELECTED_MODEL_KEY:
    if LORA_RANK == 16:
        if not LOW_VRAM:
            PER_DEVICE_TRAIN = 64 #worked with 7b. Attempts with 70 failed with killed process.
        else:
            PER_DEVICE_TRAIN = 8
    elif LORA_RANK == 32:
        if not LOW_VRAM:
            PER_DEVICE_TRAIN = 32 #7b worked with 32, trying with 40
        else:
            PER_DEVICE_TRAIN = 4
    elif LORA_RANK == 64:
        if not LOW_VRAM:
            PER_DEVICE_TRAIN = 32 #7b worked with 32, trying with 40
        else:
            PER_DEVICE_TRAIN = 4
elif "_13b" in SELECTED_MODEL_KEY or "_14b" in SELECTED_MODEL_KEY or "_16b" in SELECTED_MODEL_KEY:
    if LORA_RANK == 16:
        if not LOW_VRAM:
            if "_16b" in SELECTED_MODEL_KEY:
                PER_DEVICE_TRAIN = 16 #might need to reduce. 45gb vram without validation.
            else:
                PER_DEVICE_TRAIN = 32
        else:
            PER_DEVICE_TRAIN = 8
    elif LORA_RANK == 32:
        if not LOW_VRAM:
            if "_16b" in SELECTED_MODEL_KEY:
                PER_DEVICE_TRAIN = 8 #might need to reduce. 45gb vram without validation.
            else:
                PER_DEVICE_TRAIN = 16
        else:
            PER_DEVICE_TRAIN = 8
    elif LORA_RANK == 64:
        if not LOW_VRAM:
            PER_DEVICE_TRAIN = 8
        else:
            PER_DEVICE_TRAIN = 4
elif "_32b" in SELECTED_MODEL_KEY or "_33b" in SELECTED_MODEL_KEY or "_34b" in SELECTED_MODEL_KEY:
    DEVICE_MAP = "auto" # use auto for 32b models
    if LORA_RANK == 16:
        PER_DEVICE_TRAIN = 16 #Had to use 16
    elif LORA_RANK == 32:
        PER_DEVICE_TRAIN = 16
    elif LORA_RANK == 64:
        PER_DEVICE_TRAIN = 8 
GRADIENT_ACCUMULATION_STEPS=int(128/PER_DEVICE_TRAIN)


print("Performing training with the following configuration:")
print(f"  - Model: {SELECTED_MODEL_KEY}")
print(f"  - Dataset Type: {DATASET_TYPE}{DATASET_TYPE_EXTRA}")
print(f"  - LoRA Rank: {LORA_RANK}")
print(f"  - Per Device Train Batch Size: {PER_DEVICE_TRAIN}")
print(f"  - Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Low VRAM: {LOW_VRAM}")

#"""
# --- Model Catalog ---
# This catalog points to the base/instruct models, not pre-quantized ones.
# The script will apply 4-bit quantization on the fly, which is the correct QLoRA process.
MODEL_CATALOG = {
    # ~8B Tier
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    "llama_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen_8b": "Qwen/Qwen3-8B",
    # ~13B Tier
    "deepseek_16b": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
    "qwen_14b": "Qwen/Qwen3-14B",
    # ~32B Tier
    "deepseek_33b": "deepseek-ai/deepseek-coder-33b-instruct",
    "codellama_34b": "codellama/CodeLlama-34b-Instruct-hf",
    "qwen_32b": "Qwen/Qwen3-32B",
}
#"""

# --- 5. Dynamic Model and Prompt Setup ---
MODEL_ID = MODEL_CATALOG.get(SELECTED_MODEL_KEY)
if not MODEL_ID:
    raise ValueError(f"Model key '{SELECTED_MODEL_KEY}' not found in MODEL_CATALOG.")

print(f"Selected Model: {SELECTED_MODEL_KEY} ({MODEL_ID})")
print(f"Selected Dataset Type: {DATASET_TYPE}{DATASET_TYPE_EXTRA}")

MODEL_FAMILY = "unknown"
if "deepseek" in SELECTED_MODEL_KEY:
    MODEL_FAMILY = "deepseek"
elif "llama" in SELECTED_MODEL_KEY:
    MODEL_FAMILY = "llama"
elif "qwen" in SELECTED_MODEL_KEY:
    MODEL_FAMILY = "qwen"


chat_template = None
if "coder" in MODEL_ID.lower():
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '### Instruction:\\n' + message['content'] + '\\n### Response:\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + '<|EOT|>' }}"
        "{% endif %}"
        "{% endfor %}"
    )
elif "deepseek" in MODEL_ID.lower():
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
elif "llama" in MODEL_ID.lower() and ("3" in MODEL_ID or "3.1" in MODEL_ID):
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
elif "qwen" in MODEL_ID.lower():
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

# These functions remain the same
def format_prompt_unstructured(sample, tokenizer):
    instruction = f"Provide a detailed description of the following character from Skyrim: {sample['name']}"
    response = sample['bio']
    messages = [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}

# --- Prompt Formatting Functions ---
def format_prompt_structured(sample, tokenizer):
    """
    Formats the prompt for the structured dataset to extract character attributes
    and include the biography in the response.
    """

    instruction = (
        f"Based on the character {sample['name']} from the Skyrim universe, "
        f"identify their home city, house, race, gender, class, current location and biography. "
        f"If a specific piece of information is not available or explicitly stated in the biography, "
        f"please state 'Unknown' for that attribute.\n\n"
        f"Please provide the character's attributes and their biography in a structured format, "
        f"filling in details or 'Unknown' as appropriate.\n"
    )

    # The response now includes the character's biography along with the extracted attributes.
    response = (
        f"Home City: {sample['home_city']}\n"
        f"House: {sample['house']}\n"
        f"Race: {sample['race']}\n"
        f"Gender: {sample['gender']}\n"
        f"Class: {sample['class']}\n"
        f"Location: {sample['location']}\n\n"
        f"Character Biography:\n---\n{sample['bio']}\n---"
    )

    # The messages format remains, with the instruction as the user's prompt
    # and the structured attribute extraction + bio as the assistant's target response.
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]

    # Use the tokenizer's chat template to apply the correct model-specific formatting.
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}

def format_prompt_summarized(sample, tokenizer):
     """
     Formats the prompt for the structured dataset using the summarized biography
     to extract character attributes.
     """

     instruction = (
         f"Based on the character {sample['name']} from the Skyrim universe, "
         f"identify their home city, house, race, gender, class, current location and biography. "
         f"If a specific piece of information is not available or explicitly stated in the biography, "
         f"please state 'Unknown' for that attribute.\n\n"
         f"Please provide the character's attributes and their biography in a structured format, "
         f"filling in details or 'Unknown' as appropriate.\n"
     )

     # The response uses the summarized biography (bio_summary) instead of the full bio.
     response = (
         f"Home City: {sample['home_city']}\n"
         f"House: {sample['house']}\n"
         f"Race: {sample['race']}\n"
         f"Gender: {sample['gender']}\n"
         f"Class: {sample['class']}\n"
         f"Location: {sample['location']}\n\n"
         f"Character Biography:\n---\n{sample['bio_summary']}\n---"
     )
     messages = [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]
     return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}

# --- 6. Load and Prepare the Dataset ---
try:
    dataset_path = f"datasets/skyrim_characters_all{DATASET_TYPE_EXTRA}.csv"
    training_split_col = args.training_split_col # Get the column name from args
    print(f"Loading dataset from: {dataset_path}")
    print(f"Using training split column: '{training_split_col}'")

    if DATASET_TYPE == "unstructured":
        formatting_function = format_prompt_unstructured
        eval_formatting_function = format_prompt_unstructured # Use same for eval
    elif DATASET_TYPE == "structured":
        formatting_function = format_prompt_structured
        eval_formatting_function = format_prompt_structured # Use same for eval
    elif DATASET_TYPE == "summarized":
        formatting_function = format_prompt_summarized
        eval_formatting_function = format_prompt_summarized # Use same for eval
    else: # Fallback for unknown dataset types
         raise ValueError("DATASET_TYPE must be 'unstructured', 'structured', or 'summarized'")
    #eval_formatting_function = format_prompt_eval #this evaluation format did not work well with the structured dataset

    # Load the entire dataset from the CSV file
    full_dataset = Dataset.from_csv(dataset_path)
    print(f"Successfully loaded full dataset with {len(full_dataset)} rows.")

    # Filter the dataset based on the 'for_training' and 'for_evaluation' columns
    # This assumes these columns contain boolean (True/False) or integer (1/0) values.
    print(f"Splitting dataset using the '{training_split_col}' column.")
    train_dataset = full_dataset.filter(lambda example: bool(example[training_split_col]))
    validation_dataset = full_dataset.filter(lambda example: not bool(example[training_split_col]))

    print(f"Split dataset into {len(train_dataset)} training samples and {len(validation_dataset)} validation samples.")

except FileNotFoundError:
    print(f"Error: Dataset file '{dataset_path}' not found. Please create it and ensure it's in the correct path.")
    exit()
except (KeyError, TypeError) as e:
    print(f"Error during dataset filtering: {e}.")
    print("Please ensure your CSV file contains columns with boolean-like values (True/False or 1/0).")
    exit()




if DO_TRAINING:
    torch.cuda.empty_cache()
    print("Start training.")

    # --- 7. Load Tokenizer FIRST (for max_seq_length calculation) ---
    # Tokenizer can be loaded independently.
    if True:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        # Ensure tokenizer has chat_template and pad_token set before calculating max_seq_length
        if tokenizer.chat_template is None:
            tokenizer.chat_template = chat_template
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        calculated_max_seq_length = calculate_max_seq_length(full_dataset, tokenizer, formatting_function)
        #calculate = 2597 -> 2880 rounded
    else:
        calculated_max_seq_length = 2048 # Set a default value for max_seq_length if not calculating dynamically

    

    MUST_AVOID_UNSLOTH = False # Example models that Unsloth may not support well
    if SELECTED_MODEL_KEY == "deepseek_33b":
        print(f"Model '{SELECTED_MODEL_KEY}' is known to have issues with Unsloth. Disabling Unsloth for this model.")
        MUST_AVOID_UNSLOTH
    # ========================= MODIFIED LOADING BLOCK =========================
    if USE_UNSLOTH and not MUST_AVOID_UNSLOTH:
        #--- 7. Load Model and Tokenizer with Unsloth ---
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = MODEL_ID,
                max_seq_length = calculated_max_seq_length, # Use the calculated length
                device_map = DEVICE_MAP,
                dtype = None,
                load_in_4bit = not DISABLE_QUANTIZATION,
                trust_remote_code = True,
            )
        except Exception as e:
            print(f"Error loading model with 4-bit quantization: {e}. Trying to load without 4-bit quantization.")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = MODEL_ID,
                max_seq_length = calculated_max_seq_length, # Use the calculated length
                device_map = DEVICE_MAP,
                dtype = None,
                trust_remote_code = True,
                load_in_4bit = not DISABLE_QUANTIZATION,
            )
    else:
        # --- 7. Load Model with Standard Transformers ---
        if not DISABLE_QUANTIZATION:
            print("Loading with 4-bit quantization (BitsAndBytes)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            print("Quantization disabled. Loading in full/half precision...")
            quantization_config = None # <<< CHANGED: No config if disabled
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map=DEVICE_MAP,
            trust_remote_code=True,
        )
        # For standard transformers, max_seq_length is often handled by the tokenizer's truncation
        # during dataset preparation or directly by the model's forward pass.
        # We ensure the tokenizer is configured to handle the max length.

    # --- IMPORTANT: LoRA Adapter Initialization for Training (Always happens) ---
    # The base model MUST be wrapped with PEFT before passing to SFTTrainer,
    # regardless of whether we are starting new training or resuming.
    # The Trainer's `resume_from_checkpoint` handles loading the specific weights and optimizer states.
    if USE_UNSLOTH:
        model = FastLanguageModel.get_peft_model(
            model,
            r = LORA_RANK,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = LORA_RANK*2,
            bias = "none",
            use_gradient_checkpointing="unsloth", # Keep this False or set to "unsloth" if you want to use it
            use_rslora = False,
            loftq_config = None,
        )
    else:
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK*2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)


    if tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply formatting to the dataset
    formatted_train_dataset = train_dataset.map(lambda sample: formatting_function(sample, tokenizer))
    formatted_validation_dataset = validation_dataset.map(lambda sample: formatting_function(sample, tokenizer))

    training_args = TrainingArguments(
        output_dir=training_output_dir, # <--- Use the consolidated output directory
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN, #decrease if you run out of memory, and increase gradient_accumulation_steps to compensate
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing="unsloth" if USE_UNSLOTH else False, # Keep this False or set to "unsloth" if you want to use it
        optim="paged_adamw_8bit",
        remove_unused_columns=True,
        save_steps=SAVE_EVAL_STEPS,
        logging_steps=10,
        weight_decay=0.001,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        #max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        learning_rate=1e-4,
        #lr_scheduler_type="constant",
        lr_scheduler_type="cosine", # Use cosine decay for better convergence
        report_to="wandb" if wandb_project_name else "none",
        # evaluation parameters
        eval_strategy="steps",  # Evaluate every `eval_steps`
        eval_steps=SAVE_EVAL_STEPS,                # Or eval_steps=trainer.args.save_steps
        load_best_model_at_end=LBM,  # Load the best model found during training at the end
        metric_for_best_model="eval_loss", # Monitor validation loss to determine the best model
        greater_is_better=False,    # For loss, smaller is better (this is default for eval_loss)
        save_total_limit=1 if LBM else 2,           # Necessary to ensure that we are able to load the newest model when load_best_model_at_end=False
        # end of evaluation parameters
    )

    # --- 8. Configure LoRA with Unsloth ---
    if USE_UNSLOTH:
        try:
            trainer = SFTTrainer(
                model=model, # 'model' is already a PEFT model
                train_dataset=formatted_train_dataset,
                eval_dataset=formatted_validation_dataset,
                tokenizer=tokenizer,
                dataset_text_field="text",
                packing=False,
                use_cache=True,# if NUM_GPUS > 1 else True,  # Set to False to avoid problems with multi-gpu, Set use_cache=True for faster generation, but can get OOM with multi-gpu
                args=training_args,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=DEFAULT_EARLY_STOPPPING_PATIENCE, early_stopping_threshold=0.01),
                        LossThresholdCallback(loss_threshold=0.01)],
            )
        except Exception as e:
            print(f"Failed to create SFTTrainer with use_cache=True: {e}. Trying with use_cache=False.")
            trainer = SFTTrainer(
                model=model, # 'model' is already a PEFT model
                train_dataset=formatted_train_dataset,
                eval_dataset=formatted_validation_dataset,
                tokenizer=tokenizer,
                dataset_text_field="text",
                packing=False,
                use_cache=False,# if NUM_GPUS > 1 else True,  # Set to False to avoid problems with multi-gpu, Set use_cache=True for faster generation, but can get OOM with multi-gpu
                args=training_args,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=DEFAULT_EARLY_STOPPPING_PATIENCE, early_stopping_threshold=0.01),
                        LossThresholdCallback(loss_threshold=0.01)],
            )
    else:
        # lora_config should be defined if not USE_UNSLOTH (it is defined just above)
        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=formatted_train_dataset,
                eval_dataset=formatted_validation_dataset,
                peft_config=lora_config,
                processing_class =tokenizer,
                args=training_args,
                use_cache=False if NUM_GPUS > 1 else True,  # Set to False to avoid problems with multi-gpu, Set use_cache=True for faster generation, but can get OOM with multi-gpu
                callbacks=[EarlyStoppingCallback(early_stopping_patience=DEFAULT_EARLY_STOPPPING_PATIENCE, early_stopping_threshold=0.01),
                           LossThresholdCallback(loss_threshold=0.01)],
            )
        except Exception as e:
            #for older version support using tokenizer instead of processing_class
            trainer = SFTTrainer(
                model=model,
                train_dataset=formatted_train_dataset,
                eval_dataset=formatted_validation_dataset,
                peft_config=lora_config,
                tokenizer=tokenizer,
                dataset_text_field="text",
                packing=False,
                args=training_args,
                use_cache=False if NUM_GPUS > 1 else True,  # Set to False to avoid problems with multi-gpu, Set use_cache=True for faster generation, but can get OOM with multi-gpu
                callbacks=[EarlyStoppingCallback(early_stopping_patience=DEFAULT_EARLY_STOPPPING_PATIENCE, early_stopping_threshold=0.01),
                           LossThresholdCallback(loss_threshold=0.01)],
            )
    #with torch.autograd.detect_anomaly():
    #    trainer.train()

    # --- 10. Start Fine-Tuning ---
    print("Starting LoRA fine-tuning...")
    # The resume_from_checkpoint argument will handle loading the full state
    # We still must make sure we only use the TRAINING_DIRECTORY if the CONTINUE_FROM_CHECKPOINT is set to True
    if USE_UNSLOTH:
        # Use Unsloth's training method
        unsloth_train(trainer=trainer, resume_from_checkpoint=TRAINING_DIRECTORY if CONTINUE_FROM_CHECKPOINT else None)
        #trainer.train(resume_from_checkpoint=TRAINING_DIRECTORY if CONTINUE_FROM_CHECKPOINT else None)
    else:
        trainer.train(resume_from_checkpoint=TRAINING_DIRECTORY if CONTINUE_FROM_CHECKPOINT else None)

    print("Fine-tuning complete.")

    if save_pretrained_model:
        # --- 11. Save the Fine-Tuned LoRA Model ---
        # The final LoRA adapter is saved to the specified path.
        # When load_best_model_at_end=True, the Trainer will load the best model (which includes the LoRA adapter)
        # back into `trainer.model`. So `trainer.model.save_pretrained()` here will save the best adapter.
        trainer.model.save_pretrained(FINAL_LORA_SAVE_PATH)
        print(f"LoRA adapter (final best) saved to ./{FINAL_LORA_SAVE_PATH}")
        
    del model
    del trainer
    torch.cuda.empty_cache()


print(f"\nProcess Completed.")