import os
import time
import threading
import logging
import gc
from flask import Flask, request, jsonify, render_template

from model import ChatDoctorModel

app = Flask(__name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
chat_doctor = ChatDoctorModel()

# CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    gc.collect()
    return response

# Lazy load model in background
def load_model_lazy():
    if chat_doctor.pipe is not None:
        return chat_doctor.is_fine_tuned
    if chat_doctor.model_loading:
        logger.info("Model is already loading...")
        return False
    thread = threading.Thread(target=chat_doctor.load_model)
    thread.daemon = True
    thread.start()
    return False

@app.route('/')
def home():
    try:
        return render_template("index.html")
    except:
        return jsonify({"message": "ChatDoctor API is running."})

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@app.route('/readiness')
def readiness_check():
    if chat_doctor.pipe:
        return jsonify({"status": "ready", "model_loaded": True}), 200
    return jsonify({"status": "not_ready", "model_loaded": False}), 503

@app.route('/status', methods=['GET'])
def get_status():
    model_loaded = chat_doctor.pipe is not None
    fine_tuned_exists = os.path.exists(chat_doctor.checkpoint_dir)
    status = (
        "ready" if model_loaded and fine_tuned_exists else
        "base_model_ready" if model_loaded else
        "loading" if chat_doctor.model_loading else
        "not_ready"
    )
    message = {
        "ready": "ChatDoctor is fine-tuned and ready",
        "base_model_ready": "Base model loaded (not fine-tuned)",
        "loading": "Model is currently loading...",
        "not_ready": "Model not loaded yet"
    }[status]
    return jsonify({
        "status": status,
        "message": message,
        "model_loaded": model_loaded,
        "model_loading": chat_doctor.model_loading,
        "fine_tuned_exists": fine_tuned_exists,
        "is_fine_tuned": chat_doctor.is_fine_tuned,
        "model_type": "ChatDoctor" if chat_doctor.is_fine_tuned else "Base TinyLlama"
    })

@app.route('/medical_consultation', methods=['POST'])
def medical_consultation():
    try:
        load_model_lazy()
        if chat_doctor.model_loading or not chat_doctor.pipe:
            return jsonify({"detail": "Model is still loading. Please try again later."}), 503
        data = request.get_json()
        if not data or not data.get("question"):
            return jsonify({"detail": "Patient question is required"}), 400

        messages = chat_doctor.format_medical_prompt(data["question"])
        start_time = time.time()
        response = chat_doctor.generate_response(messages, max_new_tokens=150, temperature=0.7)
        end_time = time.time()

        disclaimer = ""
        if not chat_doctor.is_fine_tuned:
            disclaimer = " (Note: This is a general AI model. Always consult a professional for medical advice.)"

        return jsonify({
            "doctor_response": response + disclaimer,
            "time_taken": round(end_time - start_time, 2),
            "model_status": "chatdoctor_fine_tuned" if chat_doctor.is_fine_tuned else "base_model",
            "consultation_type": data.get('type', 'general'),
            "disclaimer": "This AI output is informational and not a substitute for professional medical advice."
        })

    except Exception as e:
        logger.error(f"Error during consultation: {str(e)}", exc_info=True)
        return jsonify({"detail": f"Error during consultation: {str(e)}"}), 500

@app.route('/infer', methods=['POST'])
def infer():
    try:
        load_model_lazy()
        if chat_doctor.model_loading or not chat_doctor.pipe:
            return jsonify({"detail": "Model is still loading. Please try again later."}), 503
        data = request.get_json()
        if not data or (not data.get('instruction') and not data.get('input_text')):
            return jsonify({"detail": "instruction or input_text is required"}), 400

        prompt = data.get('instruction', '').strip() or data.get('input_text', '').strip()

        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. For medical questions, recommend seeing a professional." if not chat_doctor.is_fine_tuned
                else "You are a professional medical assistant."
            )},
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()
        response = chat_doctor.generate_response(messages, max_new_tokens=100, temperature=0.8)
        end_time = time.time()

        return jsonify({
            "generated_answer": response,
            "time_taken": round(end_time - start_time, 2),
            "model_status": "chatdoctor_fine_tuned" if chat_doctor.is_fine_tuned else "base_model"
        })

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        return jsonify({"detail": f"Error during inference: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"detail": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"detail": "Internal server error"}), 500

# ✅ Entry point for local run & Cloud Run
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting ChatDoctor on 0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port)
