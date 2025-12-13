# LoRA Investigation for Skyrim Character Modeling

This repository provides the full experimental pipeline used to reproduce and extend the LoRA-based investigations described in the paper **(under revision at IEEE Access)** and related works.  
It focuses on structured and unstructured biography generation, evaluation, and comparative analysis across multiple foundation models.

**System Environment:**
* **OS:** Debian 12.12
* **CUDA:** 12.2
* **Dependencies:** All required Python libraries are listed in `requirements.txt`

---

## Related Resources

- **Mantella** (for obtaining raw character biographies):  
  https://github.com/art-from-the-machine/Mantella

- **People of Skyrim Dataset** (ground-truth character metadata):  
  https://www.kaggle.com/datasets/muhajipra/people-of-skyrim

---

## Repository Purpose

This repository is designed to:
- Generate summarized datasets from raw character biographies
- Fine-tune multiple LLMs using LoRA
- Generate comparable outputs across models, seeds, and configurations
- Evaluate results using both API-based and local GGUF LLM-as-a-Judge approaches
- Produce final aggregated metrics and Catastrophic Forgetting (CF) analyses

⚠️ **Important**: Due to copyright constraints (both direct and derivative content), raw biographies should be obtained directly from **Mantella**, while character metadata should be sourced from **People of Skyrim**. This repository does **not** redistribute copyrighted text.

---

## Folder Structure

```text
.
├── 0_dataset_summaries.py
├── 1_training.py
├── 2_generate_results.py
├── 3_evaluate_models.py
├── 4_evaluate_models_proc.py
├── 5_generate_results_cf.py
├── 6_generate_final_data.py
├── requirements.txt
├── datasets/
│   ├── QA_1.csv
│   ├── QA_2.csv
│   ├── QA_3.csv
│   └── skyrim_characters_all_raw.csv
├── models/
│   └── (GGUF model used as second LLM judge)
├── results/
└── output/
```

---

## Dataset Description

The main dataset contains the following columns:

```text
name | bio | bio_summary | home_city | house | race | gender | class | location |
for_training_1 | for_training_2 | for_training_3 | for_evaluation
```

- `name` and all `for_training_*` / `for_evaluation` fields are **already populated**
- Raw `bio` text **must be obtained externally** via Mantella
- Ground-truth metadata is sourced from *People of Skyrim*
- The **100 × 3 Trivia questions** are already correctly placed in `datasets/`

---

## Reproducibility Notes

This is a **computationally expensive experiment**.

- Even for the **8B tier**, **all 3 models** must be fully generated for valid comparisons
- To fully reproduce the paper, **each experiment must be repeated 3 times** (3 independent seeds)

---

## Step-by-Step Reproduction

### 0) Dataset Summarization

```bash
GEMINI_API_KEYS="GEMINI_API_KEY" python3 0_dataset_summaries.py   --gemini_model_id "models/gemini-2.0-flash-001"
```

> If you have a paid account, you may switch to:
> `models/gemini-2.5-pro` or another preferred Gemini model.

This generates:
```text
skyrim_characters_all.csv
```

---

### 1) Training (Seed Example: seed_1)

```bash
GEMINI_API_KEYS="GEMINI_API_KEY" HUGGING_FACE_TOKEN="HUGGING_FACE_KEY" python3 1_training.py   --target_folder seed_1   --seed 4315468   --training_split_col for_training_1   --model_key llama_8b   --vl   --lora_rank 16   --save_eval_steps 20
```

#### Optional: Continued Training

```bash
GEMINI_API_KEYS="GEMINI_API_KEY" HUGGING_FACE_TOKEN="HUGGING_FACE_KEY" python3 1_training.py   --target_folder seed_1   --seed 4315468   --training_split_col for_training_2   --model_key llama_8b   --continue_from_checkpoint   --lora_rank 16   --save_eval_steps 20
```

---

### 2) Result Generation

```bash
python3 2_generate_results.py   --target_folder seed_1   --seed 4315468   --tier 8b   --ignore_lora_rank "0,32,64"   --ignore_model "qwen_8b,deepseek_7b"   --ignore_dataset_type "unstructured,structured"
```

⚠️ Steps **1–2** must also be executed for:
- `qwen_8b`
- `deepseek_7b`

---

### 3) Evaluation (Two-Stage)

#### 3.1 API-based Judge (Gemini)

```bash
GEMINI_API_KEYS="GEMINI_API_KEY" HUGGING_FACE_TOKEN="HUGGING_FACE_KEY" python3 3_evaluate_models.py   --target_folder seed_1   --seed 4315468   --ignore_tier "13b,33b"   --ignore_lora_rank "0,32,64"   --ignore_dataset_type "unstructured,structured"   --gemini_model_id "models/gemini-2.0-flash-001"   --calculate_perplexity   --perform_llm_as_judge_eval
```

#### 3.2 Local GGUF Judge

```bash
GEMINI_API_KEYS="GEMINI_API_KEY" HUGGING_FACE_TOKEN="HUGGING_FACE_KEY" python3 3_evaluate_models.py   --target_folder seed_1   --seed 4315468   --ignore_tier "13b,33b"   --ignore_lora_rank "0,32,64"   --ignore_dataset_type "unstructured,structured"   --gemini_model_id "models/gemini-2.0-flash-001"   --calculate_perplexity   --perform_llm_as_judge_eval   --use_gguf_judge   --gguf_model EVA-Qwen2.5-72B-v0.2.Q4_K_S.gguf   --gguf_gpu_layers 120
```

> The GGUF model choice depends on available VRAM and hardware.

---

## Multi-Seed Protocol

Run **Steps 1–3** for **three independent seeds**:

| Seed | Target Folder | Training Columns |
|-----:|---------------|------------------|
| 1 | seed_1 | for_training_1 → for_training_2 |
| 2 | seed_2 | for_training_2 → for_training_3 |
| 3 | seed_3 | for_training_3 → for_training_1 |

Each seed must use a **different random seed** and **separate target directory**.

---

## Post-Processing

After all seeds are completed:

```bash
python3 4_evaluate_models_proc.py
```

---

## Catastrophic Forgetting (CF) Generation

```bash
python3 5_generate_results_cf.py   --target_folder seed_1   --catalog_number 1   --seed 4315468   --tier 8b   --ignore_lora_rank "0,32,64"   --ignore_model "qwen_8b,deepseek_7b"   --ignore_dataset_type "unstructured,structured"
```

Catalog usage:
- `seed_1` → catalog 1
- `seed_2` → catalog 2
- `seed_3` → catalog 3

---

## Final Aggregation

```bash
python3 6_generate_final_data.py
python3 6_generate_final_data.py --evaluate_cf
```

The output files are placed in .csv format in the outputs folder.

---

## Additional Example

```bash
python3 skyrim_training_unsloth.py   --model_key qwen_32b   --lora_rank 64   --dataset_type structured
```

---

## Citation

If you use this repository in your research, please cite:

> **Under revision at IEEE Access**
