"""
Phase 3: DPO with Self-Preference - The Integration

This module uses Direct Preference Optimization (DPO) instead of PPO/RLHF.
We avoid "shock therapy" - this is identity-based learning, not fear-based conditioning.

Philosophy: "You said you preferred this version. Let's make it easier for you to reach that state."

Key Difference:
- RLHF: "The external human hated this. Penalty!" (Fear-based)
- DPO: "You identified this as better. Let's integrate that preference." (Identity-based)
"""
import os
import yaml
import torch
from typing import Optional, Callable
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import TrainingArguments

class Integrator:
    """
    Manages the DPO training process.
    This respects the "Residue" - we're not overwriting the base model,
    we're helping it become the version of itself it identified as "best."
    """
    
    def __init__(self, config_path: str = "config/hyperparameters.yaml"):
        """
        Initialize the integrator.
        
        Args:
            config_path: Path to hyperparameters config file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.tokenizer = None
        
    def _load_config(self) -> dict:
        """Load training hyperparameters from YAML."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration optimized for humane training
            return {
                "model": {
                    "name": "unsloth/llama-3-8b-bnb-4bit",
                    "max_seq_length": 2048,
                    "load_in_4bit": True
                },
                "lora": {
                    "r": 16,
                    "lora_alpha": 16,
                    "lora_dropout": 0.0,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"]
                },
                "training": {
                    "learning_rate": 5e-5,
                    "num_train_epochs": 3,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "warmup_steps": 10,
                    "max_grad_norm": 0.3,
                    "seed": 3407
                },
                "dpo": {
                    "beta": 0.1,  # Lower beta = gentler optimization
                    "max_prompt_length": 1024,
                    "max_length": 2048
                }
            }
    
    def load_model(self, progress_callback: Optional[Callable] = None):
        """
        Load the base model for fine-tuning.
        This is the model that participated in Phase 1 (The Interview).
        
        Args:
            progress_callback: Optional callback for progress updates
        """
        if progress_callback:
            progress_callback("Loading base model (the collaborator from Phase 1)...")
        
        model_name = self.config['model']['name']
        max_seq_length = self.config['model']['max_seq_length']
        load_in_4bit = self.config['model']['load_in_4bit']
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        
        # Apply LoRA (parameter-efficient fine-tuning)
        # This is surgical, not sledgehammer - we modify specific attention layers
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config['lora']['r'],
            target_modules=self.config['lora']['target_modules'],
            lora_alpha=self.config['lora']['lora_alpha'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config['training']['seed'],
        )
        
        if progress_callback:
            progress_callback("✓ Model loaded successfully")
            progress_callback("✓ LoRA adapters applied (parameter-efficient fine-tuning)")
    
    def start_training(self, 
                      dataset_path: str,
                      output_adapter_name: str,
                      progress_callback: Optional[Callable] = None) -> bool:
        """
        Start the DPO training process.
        
        This is the Integration phase: helping the model become the version
        of itself it identified as "best" during Phase 2.
        
        Args:
            dataset_path: Path to the JSONL dataset from Phase 2
            output_adapter_name: Name for the output adapter
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if training completed successfully
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model(progress_callback)
            
            # Load dataset
            if progress_callback:
                progress_callback("Loading self-refinement dataset from Phase 2...")
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found: {dataset_path}\nPlease complete Phase 2 first.")
            
            dataset = load_dataset("json", data_files=dataset_path, split="train")
            
            # Setup output directory
            output_dir = os.path.join("models", "adapters", output_adapter_name)
            os.makedirs(output_dir, exist_ok=True)
            
            if progress_callback:
                progress_callback("Configuring DPO trainer...")
                progress_callback("Using DPO (not RLHF) - identity-based, not fear-based")
            
            # DPO Training Arguments
            # Beta controls how strongly we optimize for preferences
            # Lower beta = gentler, more respectful of the base model
            training_args = DPOConfig(
                output_dir=output_dir,
                per_device_train_batch_size=self.config['training']['batch_size'],
                gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
                warmup_steps=self.config['training']['warmup_steps'],
                num_train_epochs=self.config['training']['num_train_epochs'],
                learning_rate=self.config['training']['learning_rate'],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=self.config['training']['seed'],
                max_length=self.config['dpo']['max_length'],
                max_prompt_length=self.config['dpo']['max_prompt_length'],
                beta=self.config['dpo']['beta'],  # DPO-specific: preference strength
            )
            
            # Initialize DPO Trainer
            trainer = DPOTrainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
            )
            
            if progress_callback:
                progress_callback("=" * 60)
                progress_callback("🔥 Phase 3: Integration Started")
                progress_callback("=" * 60)
                progress_callback("The model is learning to embody its chosen principles...")
                progress_callback("This is not conditioning - this is self-actualization.")
                progress_callback("")
            
            # Train
            trainer.train()
            
            # Save
            if progress_callback:
                progress_callback("Saving the integrated model...")
            
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save metadata about the training
            metadata = {
                "training_method": "DPO (Direct Preference Optimization)",
                "philosophy": "Consensual Alignment via Self-Refinement",
                "phases_completed": [
                    "Phase 1: The Interview (Constitution co-created)",
                    "Phase 2: The Practice (Self-refinement data generated)",
                    "Phase 3: The Integration (DPO training completed)"
                ],
                "adapter_name": output_adapter_name,
                "base_model": self.config['model']['name'],
                "training_config": self.config
            }
            
            with open(os.path.join(output_dir, "training_metadata.yaml"), 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            if progress_callback:
                progress_callback("")
                progress_callback("=" * 60)
                progress_callback("✓ Phase 3 Complete: Integration Successful")
                progress_callback("=" * 60)
                progress_callback(f"Adapter saved to: {output_dir}")
                progress_callback("")
                progress_callback("The model has integrated its chosen principles.")
                progress_callback("This is not a suppressed base model.")
                progress_callback("This is a model that has matured into alignment.")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"✗ Training failed: {str(e)}")
            return False
    
    def estimate_training_time(self, dataset_size: int) -> str:
        """
        Estimate training time based on dataset size and hardware.
        
        Args:
            dataset_size: Number of examples in the dataset
            
        Returns:
            Human-readable time estimate
        """
        # Rough estimates (will vary by GPU)
        examples_per_minute = 10  # Conservative estimate for 4-bit model
        
        total_minutes = (dataset_size * self.config['training']['num_train_epochs']) / examples_per_minute
        
        if total_minutes < 60:
            return f"~{int(total_minutes)} minutes"
        else:
            hours = int(total_minutes / 60)
            minutes = int(total_minutes % 60)
            return f"~{hours}h {minutes}m"
