"""
Fine-tuning module for FinBot.

Provides functionality for fine-tuning language models on financial Q&A data.
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

Optional module - requires additional dependencies:
- pip install peft torch
"""

import logging
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class FinancialQADataset(Dataset):
    """
    Dataset for financial Q&A pairs.
    
    Expects JSON file with format:
    [
        {
            "question": "What is compound interest?",
            "answer": "Compound interest is...",
            "context": "Financial context..."
        }
    ]
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            file_path: Path to JSON file with Q&A data
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        self._load_data(file_path)
    
    def _load_data(self, file_path: str):
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Data file must contain a list of Q&A pairs")
            
            self.data = data
            logger.info(f"Loaded {len(self.data)} Q&A pairs from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item by index."""
        item = self.data[idx]
        
        # Prepare text
        question = item.get("question", "")
        answer = item.get("answer", "")
        context = item.get("context", "")
        
        # Combine text
        text = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class FineTuner:
    """
    Fine-tuner for financial language models using LoRA.
    
    Requires: pip install peft torch
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_lora: bool = True
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model_name: Name of model to fine-tune
            device: Device to use (cuda/cpu)
            use_lora: Whether to use LoRA for efficient fine-tuning
        """
        self.model_name = model_name
        self.device = device
        self.use_lora = use_lora
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing FineTuner on {device}")
    
    def load_model(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Apply LoRA if requested
            if self.use_lora:
                self._apply_lora()
            
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Required library not installed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _apply_lora(self):
        """Apply LoRA to model."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            logger.info("Applying LoRA to model")
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        except ImportError as e:
            logger.error(
                "PEFT library not installed. Fine-tuning requires PEFT. "
                "Install with: pip install peft\n"
                "Or install optional dependencies: pip install -e '.[finetune]'"
            )
            raise ImportError(
                "PEFT is required for fine-tuning. Install with: pip install peft"
            ) from e
    
    def prepare_data(
        self,
        qa_file: str,
        train_ratio: float = 0.8,
        batch_size: int = 8
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare train and validation data.
        
        Args:
            qa_file: Path to Q&A JSON file
            train_ratio: Ratio of training vs validation data
            batch_size: Batch size for dataloaders
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        try:
            # Load dataset
            dataset = FinancialQADataset(qa_file, self.tokenizer)
            
            # Split data
            dataset_size = len(dataset)
            train_size = int(dataset_size * train_ratio)
            val_size = dataset_size - train_size
            
            train_data, val_data = torch.utils.data.random_split(
                dataset,
                [train_size, val_size]
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                shuffle=True
            )
            
            val_loader = DataLoader(
                val_data,
                batch_size=batch_size,
                shuffle=False
            )
            
            logger.info(
                f"Data prepared: {train_size} training, {val_size} validation"
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        output_dir: str = "checkpoints"
    ) -> Dict:
        """
        Fine-tune the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            output_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        try:
            from torch.optim import AdamW
            
            if self.model is None:
                self.load_model()
            
            logger.info("Starting fine-tuning")
            
            # Setup optimizer
            optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate
            )
            
            # Training loop
            history = {
                "train_loss": [],
                "val_loss": []
            }
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    # Move to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    train_loss += loss.item()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{epochs}, "
                            f"Batch {batch_idx}/{len(train_loader)}, "
                            f"Loss: {loss.item():.4f}"
                        )
                
                # Validation
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                
                # Average losses
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                
                logger.info(
                    f"Epoch {epoch+1} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )
                
                # Save checkpoint
                checkpoint_dir = Path(output_dir) / f"checkpoint_epoch_{epoch+1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(checkpoint_dir))
                self.tokenizer.save_pretrained(str(checkpoint_dir))
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            logger.info("Fine-tuning completed")
            return history
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def save_model(self, output_dir: str):
        """Save fine-tuned model."""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


# Example usage function
def finetune_example():
    """Example of how to use the fine-tuner."""
    
    # Initialize fine-tuner
    fine_tuner = FineTuner(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_lora=True
    )
    
    # Load model
    fine_tuner.load_model()
    
    # Prepare data (assuming you have a financial_qa.json file)
    qa_file = "data/financial_qa.json"
    
    try:
        train_loader, val_loader = fine_tuner.prepare_data(
            qa_file,
            train_ratio=0.8,
            batch_size=8
        )
        
        # Fine-tune
        history = fine_tuner.fine_tune(
            train_loader,
            val_loader,
            epochs=3,
            learning_rate=2e-4,
            output_dir="checkpoints"
        )
        
        # Save model
        fine_tuner.save_model("models/finbot-finetuned")
        
        print("Fine-tuning completed!")
        print(f"History: {history}")
        
    except FileNotFoundError:
        logger.warning(f"Q&A file not found: {qa_file}")
        logger.info("To use fine-tuning, create a JSON file with Q&A pairs")


if __name__ == "__main__":
    finetune_example()
