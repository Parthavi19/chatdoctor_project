import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up checkpoint directory
checkpoint_dir = "checkpoints/tinylama-chatdoctor"
os.makedirs(checkpoint_dir, exist_ok=True)

# Load 5 samples from the dataset
logger.info("Loading 5 samples from ChatDoctor-HealthCareMagic-100k...")
try:
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train[:5]")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Load tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise

# Force CPU usage
device = torch.device("cpu")
logger.info(f"Using device: {device}")
try:
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Lightweight LoRA config
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM"
)

try:
    model = get_peft_model(model, lora_config)
except Exception as e:
    logger.error(f"Failed to apply LoRA: {e}")
    raise

# Format chat-style examples
def format_example(example):
    messages = [
        {"role": "system", "content": "You are a medical assistant providing helpful and accurate medical information."},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Apply chat formatting
logger.info("Formatting dataset...")
try:
    dataset = dataset.map(format_example)
except Exception as e:
    logger.error(f"Failed to format dataset: {e}")
    raise

# Tokenize
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

logger.info("Tokenizing dataset...")
try:
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
except Exception as e:
    logger.error(f"Failed to tokenize dataset: {e}")
    raise

# Minimal training arguments
training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    fp16=False,
    bf16=False,
    save_strategy="no",
    logging_steps=1,
    report_to="none",
    max_grad_norm=1.0,
    warmup_steps=0,
    remove_unused_columns=True
)

# Data collator
try:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
except Exception as e:
    logger.error(f"Failed to initialize data collator: {e}")
    raise

# Initialize trainer
try:
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator
    )
except Exception as e:
    logger.error(f"Failed to initialize SFTTrainer: {e}")
    raise

# Start fine-tuning
logger.info("Starting fine-tuning...")
try:
    trainer.train()
except Exception as e:
    logger.error(f"Failed during training: {e}")
    raise

# Save model and tokenizer
logger.info(f"Saving fine-tuned model to {checkpoint_dir}...")
try:
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
except Exception as e:
    logger.error(f"Failed to save model: {e}")
    raise

logger.info("Fine-tuning complete.")
