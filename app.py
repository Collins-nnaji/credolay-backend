import os
import time
import random
import logging
from functools import wraps
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import re
import io
from PyPDF2 import PdfReader
from docx import Document
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS
allowed_origins = [
    'https://credolay.com',
    'https://credolay.netlify.app'
    'https://www.credolay.netlify.app'
    'https://credolay.azurewebsites.net'
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_ENDPOINT")
)

# Rate limiting
RATE_LIMIT = 5  # requests per minute
RATE_LIMIT_PERIOD = 60  # seconds

def rate_limited(max_per_period, period):
    calls = []
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [c for c in calls if c > now - period]
            if len(calls) >= max_per_period:
                raise Exception("Rate limit exceeded")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise
                    sleep = (backoff_in_seconds * 2 ** x +
                             random.uniform(0, 1))
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def extract_json(text):
    """Attempt to extract JSON from text, even if it's not perfectly formatted."""
    try:
        # Try to find JSON-like content
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Try to parse it as JSON
            return json.loads(json_str)
    except:
        pass
    return None

def salvage_data(content):
    """Attempt to salvage data from non-JSON content."""
    data = {}
    
    # Try to extract match score
    match = re.search(r'Job Match Score:?\s*(\d+(?:\.\d+)?)%', content)
    if match:
        data['matchScore'] = float(match.group(1))
    
    # Try to extract match score explanation
    match = re.search(r'Match Score Explanation:(.*?)(?:Skills Analysis:|$)', content, re.DOTALL)
    if match:
        data['matchScoreExplanation'] = match.group(1).strip()
    
    # Try to extract skills
    skills_section = re.search(r'Skills Analysis:(.*?)(?:Skills Gap Analysis:|$)', content, re.DOTALL)
    if skills_section:
        skills = re.findall(r'(\w+(?:\s+\w+)*)\s*(?:\(match\)|\(not match\))', skills_section.group(1))
        data['skills'] = [{"name": skill, "match": "match" in line} for skill, line in zip(skills, skills_section.group(1).split('\n')) if skill]
    
    # Try to extract skills gap analysis
    skills_gap_section = re.search(r'Skills Gap Analysis:(.*?)(?:Recommendations:|$)', content, re.DOTALL)
    if skills_gap_section:
        data['skillsGapAnalysis'] = skills_gap_section.group(1).strip()
    
    # Try to extract recommendations
    recommendations_section = re.search(r'Recommendations:(.*?)(?:Suggested Job Titles:|$)', content, re.DOTALL)
    if recommendations_section:
        data['recommendations'] = []
        for rec in recommendations_section.group(1).split('\n'):
            if ':' in rec:
                action, time_estimate = rec.split(':', 1)
                data['recommendations'].append({
                    "action": action.strip(),
                    "timeEstimate": time_estimate.strip()
                })
    
    # Try to extract suggested job titles
    job_titles_section = re.search(r'Suggested Job Titles:(.*?)$', content, re.DOTALL)
    if job_titles_section:
        data['suggestedJobTitles'] = []
        for title in job_titles_section.group(1).split('\n'):
            if ':' in title:
                job_title, explanation = title.split(':', 1)
                data['suggestedJobTitles'].append({
                    "title": job_title.strip(),
                    "explanation": explanation.strip()
                })
    
    return data

def extract_text_from_file(file):
    filename = secure_filename(file.filename)
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension == '.pdf':
        pdf_reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension in ['.doc', '.docx']:
        doc = Document(io.BytesIO(file.read()))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")
    
    return text

@app.route('/')
def home():
    return "Welcome to the CV Analysis API. The backend is running correctly!"

@app.route('/api/analyze', methods=['POST'])
@rate_limited(RATE_LIMIT, RATE_LIMIT_PERIOD)
@retry_with_backoff()
def analyze():
    try:
        if 'resume' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['resume']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        job_description = request.form.get('jobDescription')
        if not job_description:
            return jsonify({"error": "Please provide a job description."}), 400

        resume_text = extract_text_from_file(file)

        messages = [
            {"role": "system", "content": "You are an AI assistant that helps analyze CVs for job fit and skills. Always respond in valid JSON format."},
            {"role": "user", "content": f"""
                Analyze the following CV and job description:

                CV: {resume_text}
                Job Description: {job_description}

                Provide the following information:
                1. Job Match Score: Give a percentage indicating how well the CV matches the job description.
                2. Match Score Explanation: Provide a short paragraph explaining the reasons for the given match score.
                3. Skills Analysis: List the key skills from the job description and indicate whether they are present in the CV (match) or missing (not match).
                4. Skills Gap Analysis: For skills that are not matched, provide a brief explanation of each skill and its importance.
                5. Recommendations: Suggest specific actions or skills the candidate should acquire to better fit the job. Include realistic time estimates for learning each skill based on the candidate's current skill level.
                6. Suggested Job Titles: Propose three job titles the candidate would be well suited for based on their current CV, along with a brief explanation for each.

                Format the response as a JSON object with the following structure:
                {{
                  "matchScore": number,
                  "matchScoreExplanation": string,
                  "skills": [{{ "name": string, "match": boolean }}],
                  "skillsGapAnalysis": string,
                  "recommendations": [{{ "action": string, "timeEstimate": string }}],
                  "suggestedJobTitles": [{{ "title": string, "explanation": string }}]
                }}
            """}
        ]

        full_response = ""
        for response in client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            timeout=120,
            stream=True
        ):
            if response.choices[0].delta.content is not None:
                full_response += response.choices[0].delta.content

        # Log the full response for debugging
        logging.info(f"Full API response: {full_response}")

        # Attempt to parse the JSON content
        try:
            analysis = json.loads(full_response)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            logging.error(f"Problematic content: {full_response}")
            
            # Attempt to extract JSON or salvage data
            extracted_json = extract_json(full_response)
            if extracted_json:
                analysis = extracted_json
            else:
                analysis = salvage_data(full_response)
            
            if not analysis:
                return jsonify({
                    "error": "Error parsing AI response. Please try again.",
                    "rawContent": full_response  # Include the raw content for debugging
                }), 500

        # Validate the parsed data
        required_keys = ["matchScore", "matchScoreExplanation", "skills", "skillsGapAnalysis", "recommendations", "suggestedJobTitles"]
        missing_keys = [key for key in required_keys if key not in analysis]
        
        if missing_keys:
            logging.error(f"Missing keys in parsed data: {missing_keys}")
            return jsonify({
                "error": f"Incomplete data received. Missing: {', '.join(missing_keys)}. Please try again.",
                "partialData": analysis
            }), 500

        return jsonify(analysis)

    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}. Please try again later."}), 500

@app.route('/api/optimize', methods=['POST'])
@rate_limited(RATE_LIMIT, RATE_LIMIT_PERIOD)
@retry_with_backoff()
def optimize():
    try:
        data = request.json
        resume_text = data.get('resumeText')
        job_description = data.get('jobDescription')
        analysis = data.get('analysis')

        if not resume_text or not job_description or not analysis:
            return jsonify({"error": "Please provide CV text, job description, and analysis results."}), 400

        messages = [
            {"role": "system", "content": "You are an AI assistant that helps optimize CVs for specific job descriptions. Always respond in valid JSON format."},
            {"role": "user", "content": f"""
                Given the following CV, job description, and analysis:

                CV: {resume_text}
                Job Description: {job_description}
                Analysis: {json.dumps(analysis)}

                Please provide tips to optimize the CV to better match the job description. Focus on the following:
                1. Highlight skills that match the job description
                2. Suggest how to address skill gaps
                3. Recommend ways to reword experiences to better align with the job requirements
                4. Suggest improvements to the CV structure to emphasize relevant experiences

                Format the response as a JSON object with the following structure:
                {{
                  "optimizationTips": [
                    {{
                      "category": string,
                      "tips": [string]
                    }}
                  ]
                }}
            """}
        ]

        full_response = ""
        for response in client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            timeout=120,
            stream=True
        ):
            if response.choices[0].delta.content is not None:
                full_response += response.choices[0].delta.content

        # Log the full response for debugging
        logging.info(f"Full API response: {full_response}")

        # Attempt to parse the JSON content
        try:
            optimization_tips = json.loads(full_response)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            logging.error(f"Problematic content: {full_response}")
            
            # Attempt to extract JSON
            extracted_json = extract_json(full_response)
            if extracted_json:
                optimization_tips = extracted_json
            else:
                return jsonify({
                    "error": "Error parsing AI response for CV optimization tips. Please try again.",
                    "rawContent": full_response  # Include the raw content for debugging
                }), 500

        # Validate the parsed JSON
        if "optimizationTips" not in optimization_tips:
            logging.error("Missing 'optimizationTips' key in parsed JSON")
            return jsonify({
                "error": "Incomplete data received. Missing optimization tips. Please try again.",
                "partialData": optimization_tips
            }), 500

        return jsonify(optimization_tips)

    except Exception as e:
        logging.error(f"An error occurred during CV optimization: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))