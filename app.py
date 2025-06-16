from flask import Flask, request, render_template, jsonify, session
from dotenv import load_dotenv
import os
import logging
from functools import wraps
from typing import Dict, Any, List
import google.generativeai as genai
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Chat history management
class ChatHistory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history = []

# Initialize chat history
chat_histories = {}

def get_chat_history(user_id: str) -> ChatHistory:
    if user_id not in chat_histories:
        chat_histories[user_id] = ChatHistory()
    return chat_histories[user_id]

# Error handling decorator
def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({'error': 'An internal server error occurred'}), 500
    return decorated_function

def analyze_health_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze symptoms using Gemini API for disease predictions and detailed information.
    
    Args:
        data: Dictionary containing user's symptoms
        
    Returns:
        Dictionary containing predictions, risk factors, and recommendations
    """
    try:
        # Ensure symptoms are provided
        if not data['symptoms']:
            return {
                'results': [],
                'risk_factors': ["No symptoms provided to analyze"],
                'recommendations': ["Please provide symptoms for analysis"]
            }

        # Prepare prompt for Gemini
        prompt = f"""
        You are a medical expert. Based on these symptoms, provide a detailed medical analysis.
        
        Patient Symptoms: {', '.join(data['symptoms'])}

        Your task is to:
        1. Analyze the symptoms
        2. Identify potential medical conditions
        3. Provide detailed information for each condition

        Format your response EXACTLY like this:

        DIAGNOSES:
        - [Disease Name] ([Probability]%)
          * Symptoms: [List the specific symptoms that match this condition]
          * Tests: [List the specific tests needed to confirm]
          * Treatment: [List the specific treatment options]

        RECOMMENDATIONS:
        - [List immediate actions]
        - [List lifestyle changes]
        - [List follow-up steps]

        RISK FACTORS:
        - [List relevant risk factors]

        Rules:
        1. Only include conditions with >30% probability
        2. Be specific and detailed
        3. Use proper medical terminology
        4. Consider symptom combinations
        5. Include both common and serious conditions
        6. Provide evidence-based recommendations

        Example:
        DIAGNOSES:
        - Influenza (85%)
          * Symptoms: fever, cough, fatigue, body aches
          * Tests: rapid flu test, complete blood count
          * Treatment: rest, fluids, antiviral medication
        - Common Cold (70%)
          * Symptoms: runny nose, sore throat, mild fever
          * Tests: physical examination
          * Treatment: rest, fluids, over-the-counter medications

        RECOMMENDATIONS:
        - Rest and stay hydrated
        - Monitor temperature
        - Seek medical attention if symptoms worsen

        RISK FACTORS:
        - Age and immune status
        - Exposure to sick individuals
        - Underlying health conditions
        """

        # Configure Gemini model parameters for more focused responses
        generation_config = {
            "temperature": 0.3,  # Lower temperature for more focused responses
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        # Get analysis from Gemini with configured parameters
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        analysis_text = response.text

        # Parse the response
        predictions = []
        risk_factors = []
        recommendations = []
        current_section = None

        # Split response into lines and process
        lines = analysis_text.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Check for section headers
            if line == 'DIAGNOSES:':
                current_section = 'predictions'
                i += 1
                continue
            elif line == 'RECOMMENDATIONS:':
                current_section = 'recommendations'
                i += 1
                continue
            elif line == 'RISK FACTORS:':
                current_section = 'risk_factors'
                i += 1
                continue

            # Process predictions
            if current_section == 'predictions' and line.startswith('-'):
                try:
                    # Extract condition and probability
                    condition_part = line[1:].strip()
                    if '(' in condition_part:
                        condition, prob = condition_part.split('(')
                        condition = condition.strip()
                        prob = float(prob.strip('%')) / 100

                        # Initialize disease data
                        disease_data = {
                            'disease': condition,
                            'probability': prob,
                            'symptoms': [],
                            'tests': [],
                            'treatment': []
                        }

                        # Process subsequent lines for symptoms, tests, and treatment
                        i += 1
                        while i < len(lines) and lines[i].strip().startswith('*'):
                            content = lines[i].strip()[1:].strip()
                            if content.startswith('Symptoms:'):
                                disease_data['symptoms'] = [s.strip() for s in content[9:].split(',')]
                            elif content.startswith('Tests:'):
                                disease_data['tests'] = [t.strip() for t in content[6:].split(',')]
                            elif content.startswith('Treatment:'):
                                disease_data['treatment'] = [t.strip() for t in content[10:].split(',')]
                            i += 1

                        # Only include predictions with probability > 30%
                        if 0.3 < prob <= 1.0:
                            predictions.append(disease_data)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing prediction line: {line}, Error: {str(e)}")
                    i += 1
                    continue

            # Process recommendations
            elif current_section == 'recommendations' and line.startswith('-'):
                recommendations.append(line[1:].strip())
                i += 1

            # Process risk factors
            elif current_section == 'risk_factors' and line.startswith('-'):
                risk_factors.append(line[1:].strip())
                i += 1
            else:
                i += 1

        # Sort predictions by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        # If no predictions were found, try to get at least one
        if not predictions:
            # Try to get at least one prediction with a simpler prompt
            simple_prompt = f"""
            Based on these symptoms: {', '.join(data['symptoms'])}
            What is the most likely medical condition? Format as:
            - [Disease Name] (80%)
              * Symptoms: [List symptoms]
              * Tests: [List tests]
              * Treatment: [List treatment]
            """
            try:
                simple_response = model.generate_content(simple_prompt)
                simple_text = simple_response.text
                # Parse the simple response
                if '-' in simple_text:
                    condition_part = simple_text.split('-')[1].strip()
                    if '(' in condition_part:
                        condition, prob = condition_part.split('(')
                        condition = condition.strip()
                        prob = float(prob.strip('%')) / 100
                        predictions.append({
                            'disease': condition,
                            'probability': prob,
                            'symptoms': data['symptoms'],
                            'tests': ['Physical examination', 'Blood tests'],
                            'treatment': ['Consult healthcare provider']
                        })
            except Exception as e:
                logger.error(f"Error in simple prediction: {str(e)}")

        # Ensure we have at least some recommendations
        if not recommendations:
            recommendations = [
                "Schedule a comprehensive medical evaluation",
                "Maintain detailed symptom documentation",
                "Follow up with primary care physician"
            ]

        # Ensure we have at least some risk factors
        if not risk_factors:
            risk_factors = ["Clinical assessment required for accurate risk factor evaluation"]

        logger.info(f"Health analysis completed: {len(predictions)} conditions identified")

        return {
            'results': predictions,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }

    except Exception as e:
        logger.error(f"Error in health analysis: {str(e)}", exc_info=True)
        return {
            'results': [],
            'risk_factors': ["Unable to complete clinical assessment due to processing error"],
            'recommendations': [
                "Seek immediate medical evaluation",
                "Document all symptoms and their progression",
                "Schedule follow-up with healthcare provider"
            ]
        }

def process_chat_message(message: str, user_id: str) -> str:
    """
    Process chat messages using Gemini API with conversation history.
    
    Args:
        message: User's message
        user_id: Unique identifier for the user's chat session
        
    Returns:
        AI's response
    """
    try:
        # Get chat history for the user
        chat_history = get_chat_history(user_id)
        
        # Add user message to history
        chat_history.add_message('user', message)
        
        # Prepare conversation context
        conversation_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history.get_history()[:-1]  # Exclude the current message
        ])
        
        # Prepare prompt with conversation history
        prompt = f"""
        You are a professional medical doctor providing concise, accurate health advice. 
        Previous conversation:
        {conversation_context}
        
        Current user message: {message}
        
        Provide a professional medical response following these guidelines:
        1. Keep responses to 2-3 lines maximum
        2. Use clear, professional medical terminology
        3. Be direct and informative
        4. Include only essential medical information
        5. If the question is not health-related, politely redirect to health topics
        6. Always maintain a professional, clinical tone
        7. If unsure, recommend consulting a healthcare provider
        
        Format your response as a professional medical consultation.
        """
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Add assistant response to history
        chat_history.add_message('assistant', response_text)
        
        return response_text
    
    except Exception as e:
        logger.error(f"Error in chat processing: {str(e)}")
        return "I apologize, but I'm having trouble processing your request. Please try again later."

@app.route('/')
@handle_errors
def index():
    """Render the main application page."""
    return render_template("index.html")

@app.route('/chat')
@handle_errors
def chat():
    """Render the chat interface page."""
    return render_template("chat.html")

@app.route('/chat', methods=['POST'])
@handle_errors
def chat_response():
    """
    Process chat messages and return responses with conversation history.
    
    Request Body:
        JSON object containing:
            message (str): The user's message
            user_id (str): Unique identifier for the chat session
    
    Returns:
        JSON object containing:
            response (str): The assistant's response
            history (list): Updated conversation history
    """
    data = request.get_json()
    if not data or 'message' not in data or 'user_id' not in data:
        return jsonify({'error': 'Invalid request format'}), 400
    
    message = data.get('message', '')
    user_id = data.get('user_id', '')
    
    response = process_chat_message(message, user_id)
    chat_history = get_chat_history(user_id)
    
    return jsonify({
        'response': response,
        'history': chat_history.get_history()
    })

@app.route('/chat/clear', methods=['POST'])
@handle_errors
def clear_chat():
    """
    Clear chat history for a specific user.
    
    Request Body:
        JSON object containing:
            user_id (str): Unique identifier for the chat session
    
    Returns:
        JSON object confirming history cleared
    """
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({'error': 'Invalid request format'}), 400
    
    user_id = data.get('user_id', '')
    chat_history = get_chat_history(user_id)
    chat_history.clear()
    
    return jsonify({'message': 'Chat history cleared successfully'})

@app.route('/predict', methods=['GET'])
@handle_errors
def predict_page():
    """Render the disease prediction page."""
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
@handle_errors
def predict():
    """
    Process symptom-based health analysis requests and return JSON results.
    
    Request Body:
        Form data containing:
            symptoms (str): Comma-separated list of symptoms
            custom_symptoms (str): Additional symptoms
    
    Returns:
        JSON object with results, risk factors, and recommendations
    """
    try:
        # Process symptoms
        symptoms = []
        if request.form.get("symptoms"):
            symptoms.extend([s.strip().lower() for s in request.form.getlist("symptoms")])
        
        # Process custom symptoms
        custom_symptoms = request.form.get("custom_symptoms", "").strip()
        if custom_symptoms:
            symptoms.extend([s.strip().lower() for s in custom_symptoms.split(',')])
        
        # Remove duplicates and empty strings
        symptoms = list(set([s for s in symptoms if s]))

        data = {
            "symptoms": symptoms
        }

        # Validate input
        if not symptoms:
            return jsonify({"error": 'No symptoms provided'}), 400

        # Get health analysis
        analysis = analyze_health_data(data)
        
        logger.info("Successfully processed symptom analysis request")
        return jsonify(analysis)

    except Exception as e:
        logger.error(f"Error processing health analysis: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route('/result')
@handle_errors
def result():
    """Render the prediction results page."""
    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)