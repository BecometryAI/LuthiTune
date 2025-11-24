import torch
import yaml
import argparse
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# ==========================================
# 0. SETUP & ARGUMENT PARSING
# ==========================================
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

parser = argparse.ArgumentParser(description="LuthiTune Training Engine")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file")
args = parser.parse_args()

config = load_config(args.config)

# ==========================================
# 1. DYNAMIC CONFIGURATION
# ==========================================
MODEL_NAME = config['model']['name']
MAX_SEQ_LENGTH = config['model']['max_seq_length']
LOAD_IN_4BIT = config['model']['load_in_4bit']
OUTPUT_DIR = config['training']['output_dir']
DATA_FILE = config['training']['data_file']

def train():
    # ==========================================
    # 2. LOAD THE BASE MODEL (Agnostic)
    # ==========================================
    print(f"--- LuthiTune Engine Starting ---")
    print(f"--- Loading Target Model: {MODEL_NAME} ---")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto-detects based on hardware
        load_in_4bit = LOAD_IN_4BIT,
    )

    # ==========================================
    # 3. CONFIGURE LoRA (The "Adapters")
    # ==========================================
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['hyperparameters']['r'],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = True, 
        random_state = config['training']['seed'],
    )

    # ==========================================
    # 4. LOAD DATASET
    # ==========================================
    print(f"--- Loading Dataset from {DATA_FILE} ---")
    if os.path.exists(DATA_FILE):
        dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    else:
        print(f"CRITICAL ERROR: {DATA_FILE} not found. Please check your config.")
        return

    # ==========================================
    # 5. THE TRAINING LOOP
    # ==========================================
    print("--- Starting Training Session ---")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        
        args = TrainingArguments(
            per_device_train_batch_size = config['hyperparameters']['batch_size'],
            gradient_accumulation_steps = 4, 
            warmup_steps = 5,
            max_steps = config['hyperparameters']['max_steps'],
            learning_rate = config['hyperparameters']['learning_rate'],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(), 
            logging_steps = 1,
            optim = "adamw_8bit", 
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = config['training']['seed'],
            output_dir = OUTPUT_DIR,
        ),
    )

    trainer.train()

    # ==========================================
    # 6. EXPORT
    # ==========================================
    print("--- Saving Adapter ---")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"SUCCESS: Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()