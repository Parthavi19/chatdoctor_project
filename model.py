# model.py
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
import os

logger = logging.getLogger(__name__)

class ChatDoctorModel:
    def __init__(self, checkpoint_dir="checkpoints/tinylama-chatdoctor", device_map="auto"):
        self.checkpoint_dir = checkpoint_dir
        self.device_map = device_map
        self.pipe = None
        self.is_fine_tuned = False
        self.model_loading = False

    def load_model(self):
        self.model_loading = True
        logger.info(f"Loading model from {self.checkpoint_dir}...")
        try:
            if os.path.exists(self.checkpoint_dir) and os.path.isdir(self.checkpoint_dir):
                logger.info("Loading fine-tuned ChatDoctor model...")
                self.pipe = pipeline(
                    "text-generation",
                    model=self.checkpoint_dir,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device_map
                )
                self.is_fine_tuned = True
            else:
                logger.info("Loading base TinyLlama model...")
                model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.pipe = pipeline(
                    "text-generation",
                    model=model_id,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device_map
                )
                self.pipe.tokenizer.pad_token = self.pipe.tokenizer.pad_token or self.pipe.tokenizer.eos_token
            logger.info("Model loaded.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        finally:
            self.model_loading = False

    def generate_response(self, messages, max_new_tokens=150, temperature=0.7, top_p=0.95):
        if not self.pipe:
            raise ValueError("Model pipeline not loaded.")
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        return outputs[0]["generated_text"][len(prompt):].strip()

    def format_medical_prompt(self, question):
        return [
            {"role": "system", "content": "You are a medical assistant providing helpful and accurate medical information."},
            {"role": "user", "content": question}
        ]
