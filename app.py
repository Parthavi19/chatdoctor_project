# app.py
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

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    gc.collect()
    return response

# Lazy load model in a background thread
def load_model_lazy():
    if chat_doctor.pipe is not None:
        return chat_doctor.is_fine_tuned
    if chat_doctor.model_loading:
        start_time = time.time()
        while chat_doctor.model_loading and time.time() - start_time < 300:  # 5-minute timeout
            time.sleep(1)
        if chat_doctor.model_loading:
            logger.error("Model loading timed out.")
        return chat_doctor.is_fine_tuned
    thread = threading.Thread(target=chat_doctor.load_model)
    thread.daemon = True
    thread.start()
    start_time = time.time()
    while chat_doctor.model_loading and time.time() - start_time < 300:
        time.sleep(1)
    if chat_doctor.model_loading:
        logger.error("Model loading timed out.")
    return chat_doctor.is_fine_tuned

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@app.route('/readiness')
def readiness_check():
    if chat_doctor.pipe is not None:
        return jsonify({"status": "ready", "model_loaded": True}), 200
    elif chat_doctor.model_loading:
        return jsonify({"status": "loading", "model_loaded": False}), 202
    else:
        return jsonify({"status": "not_ready", "model_loaded": False}), 503

@app.route('/status', methods=['GET'])
def get_status():
    model_loaded = chat_doctor.pipe is not None
    fine_tuned_exists = os.path.exists(chat_doctor.checkpoint_dir) and os.path.isdir(chat_doctor.checkpoint_dir)
    status = "loading" if chat_doctor.model_loading else \
             "ready" if fine_tuned_exists and model_loaded else \
             "base_model_ready" if model_loaded else "not_ready"
    message = "Model is currently loading..." if chat_doctor.model_loading else \
              "ChatDoctor model is fine-tuned and ready" if fine_tuned_exists and model_loaded else \
              "Base model loaded (not fine-tuned for medical use)" if model_loaded else \
              "Model not loaded yet"
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
        is_fine_tuned = load_model_lazy()
        if chat_doctor.model_loading or chat_doctor.pipe is None:
            return jsonify({"detail": "Model failed to load. Please try again later."}), 503
        data = request.get_json()
        if not data or not data.get('question'):
            return jsonify({"detail": "Patient question is required"}), 400
        messages = chat_doctor.format_medical_prompt(data['question'])
        start_time = time.time()
        response = chat_doctor.generate_response(messages, max_new_tokens=150, temperature=0.7)
        end_time = time.time()
        disclaimer = " (Note: This is a general AI model, not specifically trained for medical advice. Please consult healthcare professionals.)" if not is_fine_tuned else ""
        return jsonify({
            "doctor_response": response + disclaimer,
            "time_taken": round(end_time - start_time, 2),
            "model_status": "chatdoctor_fine_tuned" if is_fine_tuned else "base_model",
            "consultation_type": data.get('type', 'general'),
            "disclaimer": "This AI response is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment."
        })
    except Exception as e:
        logger.error(f"Error during medical consultation: {str(e)}", exc_info=True)
        gc.collect()
        return jsonify({"detail": f"Error during consultation: {str(e)}"}), 500

@app.route('/infer', methods=['POST'])
def infer():
    try:
        is_fine_tuned = load_model_lazy()
        if chat_doctor.model_loading or chat_doctor.pipe is None:
            return jsonify({"detail": "Model failed to load. Please try again later."}), 503
        data = request.get_json()
        if not data or (not data.get('instruction') and not data.get('input_text')):
            return jsonify({"detail": "Either instruction or input_text field is required"}), 400
        instruction = data.get('instruction', '').strip()
        input_text = data.get('input_text', '').strip()
        messages = [
            {"role": "system", "content": "You are a helpful assistant." if is_fine_tuned else "You are a helpful assistant. Provide general information and recommend consulting professionals for medical advice."},
            {"role": "user", "content": instruction or input_text}
        ]
        start_time = time.time()
        response = chat_doctor.generate_response(messages, max_new_tokens=100, temperature=0.8)
        end_time = time.time()
        return jsonify({
            "generated_answer": response,
            "time_taken": round(end_time - start_time, 2),
            "model_status": "chatdoctor_fine_tuned" if is_fine_tuned else "base_model"
        })
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        gc.collect()
        return jsonify({"detail": f"Error during inference: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"detail": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"detail": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting ChatDoctor Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
