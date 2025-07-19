import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
import os

logger = logging.getLogger(__name__)

class ChatDoctorModel:
    def __init__(self, use_cpu=True):
        self.use_cpu = use_cpu
        self.pipe = None
        self.model_loading = False
        self.model_loaded = False
        self.is_fine_tuned = False
        self.checkpoint_dir = "checkpoints/tinylama-chatdoctor"

    def load_model(self):
        logger.info("Starting model load process...")
        if self.model_loaded:
            logger.info("Model already loaded, skipping.")
            return
            
        self.model_loading = True
        model_id = os.path.exists(self.checkpoint_dir) and "TinyLlama/TinyLlama-1.1B-Chat-v1.0" or "microsoft/DialoGPT-small"
        logger.info(f"Loading model from {model_id} or checkpoint {self.checkpoint_dir}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir if os.path.exists(self.checkpoint_dir) else model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_dir if os.path.exists(self.checkpoint_dir) else model_id,
                torch_dtype=torch.float32,
                device_map="cpu" if self.use_cpu else "auto",
                low_cpu_mem_usage=True
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1 if self.use_cpu else 0,
                max_length=200,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            self.model_loaded = True
            self.is_fine_tuned = os.path.exists(self.checkpoint_dir)
            logger.info(f"Model loaded successfully (fine-tuned: {self.is_fine_tuned}).")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            try:
                logger.info("Falling back to gpt2...")
                self.pipe = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=-1,
                    max_length=150
                )
                self.model_loaded = True
                self.is_fine_tuned = False
                logger.info("Fallback model (GPT-2) loaded successfully.")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raise e2
        finally:
            self.model_loading = False

    def generate_response(self, messages, max_new_tokens=100, temperature=0.7, top_p=0.95):
        if not self.pipe:
            raise ValueError("Model pipeline not loaded.")
        
        if isinstance(messages, list) and len(messages) > 0:
            prompt = messages[-1].get("content", "")
        else:
            prompt = str(messages)
            
        medical_prompt = f"Medical Assistant: Please provide helpful medical information about: {prompt}\nResponse:"
        
        try:
            outputs = self.pipe(
                medical_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]["generated_text"]
            response = generated_text[len(medical_prompt):].strip()
            
            if not response:
                response = "I understand you're asking about a medical concern. Please consult with a healthcare professional for proper medical advice."
                
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again or consult a healthcare professional."

    def format_medical_prompt(self, question):
        return [
            {"role": "system", "content": "You are a helpful medical information assistant."},
            {"role": "user", "content": question}
        ]
