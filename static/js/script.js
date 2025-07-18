document.addEventListener('DOMContentLoaded', () => {
    const medicalTab = document.getElementById('medicalTab');
    const generalTab = document.getElementById('generalTab');
    const medicalForm = document.getElementById('medicalForm');
    const generalForm = document.getElementById('generalForm');
    const medicalConsultationForm = document.getElementById('medicalConsultationForm');
    const generalInferenceForm = document.getElementById('generalInferenceForm');
    const responseArea = document.getElementById('responseArea');
    const responseText = document.getElementById('responseText');
    const responseMeta = document.getElementById('responseMeta');
    const disclaimer = document.getElementById('disclaimer');
    const loading = document.getElementById('loading');
    const checkStatus = document.getElementById('checkStatus');
    const statusText = document.getElementById('statusText');

    // Toggle between tabs
    medicalTab.addEventListener('click', () => {
        medicalTab.classList.add('bg-blue-600', 'text-white');
        medicalTab.classList.remove('bg-gray-300', 'text-gray-800');
        generalTab.classList.add('bg-gray-300', 'text-gray-800');
        generalTab.classList.remove('bg-blue-600', 'text-white');
        medicalForm.classList.remove('hidden');
        generalForm.classList.add('hidden');
    });

    generalTab.addEventListener('click', () => {
        generalTab.classList.add('bg-blue-600', 'text-white');
        generalTab.classList.remove('bg-gray-300', 'text-gray-800');
        medicalTab.classList.add('bg-gray-300', 'text-gray-800');
        medicalTab.classList.remove('bg-blue-600', 'text-white');
        generalForm.classList.remove('hidden');
        medicalForm.classList.add('hidden');
    });

    // Handle medical consultation form submission
    medicalConsultationForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = document.getElementById('medicalQuestion').value.trim();
        const consultationType = document.getElementById('consultationType').value;
        if (!question) {
            alert('Please enter a medical question.');
            return;
        }

        responseArea.classList.add('hidden');
        loading.classList.remove('hidden');

        try {
            const response = await fetch('/medical_consultation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, type: consultationType })
            });
            const data = await response.json();

            if (response.ok) {
                responseText.textContent = data.doctor_response;
                responseMeta.textContent = `Time taken: ${data.time_taken}s | Model: ${data.model_status} | Type: ${data.consultation_type}`;
                disclaimer.textContent = data.disclaimer;
                responseArea.classList.remove('hidden');
            } else {
                responseText.textContent = `Error: ${data.detail}`;
                responseMeta.textContent = '';
                disclaimer.textContent = '';
                responseArea.classList.remove('hidden');
            }
        } catch (error) {
            responseText.textContent = `Error: ${error.message}`;
            responseMeta.textContent = '';
            disclaimer.textContent = '';
            responseArea.classList.remove('hidden');
        } finally {
            loading.classList.add('hidden');
        }
    });

    // Handle general inference form submission
    generalInferenceForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const inputText = document.getElementById('generalInput').value.trim();
        if (!inputText) {
            alert('Please enter a question.');
            return;
        }

        responseArea.classList.add('hidden');
        loading.classList.remove('hidden');

        try {
            const response = await fetch('/infer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input_text: inputText })
            });
            const data = await response.json();

            if (response.ok) {
                responseText.textContent = data.generated_answer;
                responseMeta.textContent = `Time taken: ${data.time_taken}s | Model: ${data.model_status}`;
                disclaimer.textContent = data.disclaimer || 'This AI response is for informational purposes only.';
                responseArea.classList.remove('hidden');
            } else {
                responseText.textContent = `Error: ${data.detail}`;
                responseMeta.textContent = '';
                disclaimer.textContent = '';
                responseArea.classList.remove('hidden');
            }
        } catch (error) {
            responseText.textContent = `Error: ${error.message}`;
            responseMeta.textContent = '';
            disclaimer.textContent = '';
            responseArea.classList.remove('hidden');
        } finally {
            loading.classList.add('hidden');
        }
    });

    // Check server status
    checkStatus.addEventListener('click', async () => {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            statusText.textContent = `Status: ${data.status} | Message: ${data.message} | Model Loaded: ${data.model_loaded} | Fine-Tuned: ${data.is_fine_tuned}`;
        } catch (error) {
            statusText.textContent = `Error checking status: ${error.message}`;
        }
    });
});
