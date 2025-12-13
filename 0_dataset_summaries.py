import pandas as pd
import numpy as np
import os
import glob
import warnings
import argparse
import random
import json
import google.generativeai as genai
import time
from tqdm import tqdm
import dirtyjson
import argparse

# ===================================================================================
# --- SCRIPT CONFIGURATION ---
# ===================================================================================


# Gemini API Keys (replace with your own)
GEMINI_API_KEYS = os.environ["GEMINI_API_KEYS"].split(",") if "GEMINI_API_KEYS" in os.environ else []
CURRENT_GEMINI_KEY_INDEX = 0

# Gemini Model ID
#if you don't have a paid account, use "models/gemini-2.0-flash-001", otherwise, "models/gemini-2.5-pro"
DEFAULT_GEMINI_MODEL_ID = "models/gemini-2.0-flash-001" #"models/gemini-1.5-pro-latest"

parser = argparse.ArgumentParser(description="Skyrim Character Biography Summarization using Gemini")
parser.add_argument("--gemini_model_id", type=str, default=DEFAULT_GEMINI_MODEL_ID,
                    help="Gemini model ID to use as the judge. If not provided, the script will not set a specific Gemini model ID.")
args = parser.parse_args()




# File Paths
INPUT_CSV_PATH = 'datasets/skyrim_characters_all_raw.csv'
OUTPUT_CSV_PATH = f'datasets/skyrim_characters_all.csv'
TEMP_SAVE_PATH = f'datasets/temp_summaries_on_error.csv'
JUDGE_GEMINI_MODEL_ID = args.gemini_model_id


# Processing settings
BATCH_SIZE = 1
MAX_RETRIES = 5

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)


def get_gemini_model():
    """Initializes and returns a Gemini model instance with the current API key."""
    global CURRENT_GEMINI_KEY_INDEX
    
    # Configure with a new key on each retry or call to this function
    try:
        genai.configure(api_key=GEMINI_API_KEYS[CURRENT_GEMINI_KEY_INDEX])
        return genai.GenerativeModel(JUDGE_GEMINI_MODEL_ID)
    except Exception as e:
        print(f"Error configuring Gemini with key index {CURRENT_GEMINI_KEY_INDEX}: {e}")
        # Try the next key if the current one fails
        CURRENT_GEMINI_KEY_INDEX = (CURRENT_GEMINI_KEY_INDEX + 1) % len(GEMINI_API_KEYS)
        print(f"Switching to new key at index {CURRENT_GEMINI_KEY_INDEX}.")
        return get_gemini_model()


def build_batch_prompt(characters):
    """
    Builds a single prompt to summarize a batch of characters.
    """
    
    prompt = (
        "You are an expert in The Elder Scrolls V: Skyrim lore. Your task is to read the biographies "
        "of several characters and provide a concise summary for each. "
        "The summary for each character should be a single paragraph (maximum 3 sentences and 200 characters) "
        "describing where they live, what their job is, their main relationships, and their core personality.\n"
        "Your final output MUST be a single JSON object containing a list of objects, "
        "where each object has a 'name' and a 'biography_summary' key.\n\n"
    )
    
    for i, char in enumerate(characters):
        prompt += f"--- Character {i+1} ---\n"
        prompt += f"Name: {char['name']}\n"
        prompt += f"Biography: {char['bio']}\n"
        prompt += f"--- End Character {i+1} ---\n"
        
    prompt += "\n\nJSON Output:\n"
    return prompt


def summarize_biographies():
    """
    Orchestrates the batch summarization of character biographies using Gemini.
    """
    global CURRENT_GEMINI_KEY_INDEX

    # --- 1. Load Data with Resumption Logic ---
    try:
        if os.path.exists(OUTPUT_CSV_PATH):
            print(f"Found existing output file '{OUTPUT_CSV_PATH}'. Resuming from a previous run.")
            df = pd.read_csv(OUTPUT_CSV_PATH)
            if 'bio_summary' not in df.columns:
                df['bio_summary'] = np.nan
        else:
            print(f"Output file '{OUTPUT_CSV_PATH}' not found. Starting from scratch.")
            df = pd.read_csv(INPUT_CSV_PATH)
            df['bio_summary'] = np.nan
            df.to_csv(OUTPUT_CSV_PATH, index=False) # Save the initial file structure
            
        characters_to_summarize = df[df['bio_summary'].isnull()]
        print(f"Found {len(characters_to_summarize)} characters to summarize.")
        
        if characters_to_summarize.empty:
            print("All characters already have a summary. No work to do.")
            return

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{INPUT_CSV_PATH}' exists.")
        return

    # --- 2. Process in Batches with Retry Logic ---
    model = get_gemini_model()
    
    total_to_process = len(characters_to_summarize)
    for i in tqdm(range(0, total_to_process, BATCH_SIZE), desc="Processing batches"):
        
        # Get the next batch of characters
        batch_to_process = characters_to_summarize.iloc[i : i+BATCH_SIZE]
        
        # Check if the batch has already been processed in another run
        if batch_to_process['bio_summary'].notna().all():
            continue

        # Create a list of dictionaries for the prompt
        character_data = batch_to_process.to_dict('records')
        
        # Build the prompt for the current batch
        prompt = build_batch_prompt(character_data)
        
        # Retry loop for Gemini API call
        for attempt in range(MAX_RETRIES):
            print(f"Attempt {attempt + 1} to summarize batch starting with '{batch_to_process.iloc[0]['name']}'")
            
            try:
                # --- Gemini API Call ---
                response = model.generate_content(prompt)
                response_text = response.text.strip()

                # --- Parsing and Validation ---
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                if json_start != -1 and json_end != 0:
                    json_str = response_text[json_start:json_end]
                    # Use dirtyjson for a more forgiving parse
                    parsed_json_list = dirtyjson.loads(json_str)
                    
                    # Ensure it's a list and contains the correct number of items
                    if isinstance(parsed_json_list, list) and len(parsed_json_list) == len(character_data):
                        # Update the main DataFrame with the summaries
                        for item in parsed_json_list:
                            name = item.get('name')
                            summary = item.get('biography_summary')
                            if name and summary:
                                df.loc[df['name'] == name, 'bio_summary'] = summary
                        
                        # Save the updated DataFrame after a successful batch
                        df.to_csv(OUTPUT_CSV_PATH, index=False)
                        print(f"Successfully summarized batch. Saved to '{OUTPUT_CSV_PATH}'.")
                        time.sleep(1)
                        break # Exit the retry loop

                    else:
                        raise ValueError("Parsed JSON does not match expected format or size.")
                else:
                    raise ValueError("No valid JSON list found in response.")

            except (dirtyjson.Error, ValueError, Exception) as e:
                print(f"Error processing Gemini response (attempt {attempt + 1}): {e}")
                print(f"Raw response: {response_text[:200]}...") # Print a snippet for debugging

                if attempt < MAX_RETRIES - 1:
                    # Switch to the next API key and wait before retrying
                    wait_time = random.randint(5, 15)
                    print(f"Waiting {wait_time} seconds before retrying.")
                    time.sleep(wait_time)
                    CURRENT_GEMINI_KEY_INDEX = (CURRENT_GEMINI_KEY_INDEX + 1) % len(GEMINI_API_KEYS)
                    model = get_gemini_model() # Re-initialize with the new key
                    continue
                else:
                    print(f"Max retries reached for batch. Skipping this batch.")
                    break # Exit the retry loop after all retries fail
    
    print("\nAll tasks completed.")

# --- 3. Main Execution and Safety Net ---
if __name__ == "__main__":
    try:
        summarize_biographies()
    except Exception as e:
        print(f"\nAn unhandled exception occurred: {e}")
        # Final save of the current progress before exiting
        try:
            temp_df = pd.read_csv(OUTPUT_CSV_PATH)
            temp_df.to_csv(TEMP_SAVE_PATH, index=False)
            print(f"Work in progress saved to '{TEMP_SAVE_PATH}'.")
        except Exception as save_e:
            print(f"Failed to save temporary work: {save_e}")
        finally:
            exit(1)