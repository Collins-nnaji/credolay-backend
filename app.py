# app.py

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
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS
allowed_origins = [
    'https://credolay.com',
    'https://credolay-eydvckgkcmbycsdg.eastus-01.azurewebsites.net'
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
    
    # Try to extract skills
    skills_section = re.search(r'Skills Analysis:(.*?)(?:Recommendations:|$)', content, re.DOTALL)
    if skills_section:
        skills = re.findall(r'(\w+(?:\s+\w+)*)\s*(?:\(match\)|\(gap\))', skills_section.group(1))
        data['skills'] = [{"name": skill, "match": "match" in line} for skill, line in zip(skills, skills_section.group(1).split('\n')) if skill]
    
    # Try to extract recommendations
    recommendations_section = re.search(r'Recommendations:(.*?)(?:Suggested Job Titles:|$)', content, re.DOTALL)
    if recommendations_section:
        data['recommendations'] = [rec.strip() for rec in recommendations_section.group(1).split('\n') if rec.strip()]
    
    # Try to extract suggested job titles
    job_titles_section = re.search(r'Suggested Job Titles:(.*?)$', content, re.DOTALL)
    if job_titles_section:
        data['suggestedJobTitles'] = [title.strip() for title in job_titles_section.group(1).split('\n') if title.strip()]
    
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
                2. Skills Analysis: List the key skills from the job description and indicate whether they are present in the CV (match) or missing (gap).
                3. Recommendations: Suggest specific actions or skills the candidate should acquire to better fit the job.
                4. Suggested Job Titles: Propose three job titles the candidate would be well suited for based on their current CV.

                Format the response as a JSON object with the following structure:
                {{
                  "matchScore": number,
                  "skills": [{{ "name": string, "match": boolean }}],
                  "recommendations": [string],
                  "suggestedJobTitles": [string]
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
        required_keys = ["matchScore", "skills", "recommendations", "suggestedJobTitles"]
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
@retry_with_backoff(retries=5, backoff_in_seconds=2)
def optimize():
    try:
        data = request.json
        resume_text = data.get('resumeText')
        job_description = data.get('jobDescription')
        analysis = data.get('analysis')

        if not resume_text or not job_description or not analysis:
            return jsonify({"error": "Please provide CV text, job description, and analysis results."}), 400

        messages = [
            {"role": "system", "content": "You are an AI assistant that helps optimize CVs for specific job descriptions. Provide the optimized CV in a clear, structured format with sections."},
            {"role": "user", "content": f"""
                Given the following CV, job description, and analysis:

                CV: {resume_text}
                Job Description: {job_description}
                Analysis: {json.dumps(analysis)}

                Please optimize the CV to better match the job description. Focus on the following:
                1. Highlight skills that match the job description
                2. Add any missing key skills if the candidate possesses them
                3. Reword experiences to better align with the job requirements
                4. Reorganize the CV structure if necessary to emphasize relevant experiences

                Provide the optimized CV in a clear, professional format without the person's name or fictional name.
                Use the following structure, with each section clearly labeled:

                [PROFESSIONAL SUMMARY]
                A brief, impactful summary of the candidate's key qualifications and experience

                [SKILLS]
                • Skill 1
                • Skill 2
                • Skill 3
                ...

                [WORK EXPERIENCE]
                Job Title, Company Name (Start Date - End Date)
                • Key responsibility or achievement
                • Another key responsibility or achievement
                ...

                [EDUCATION]
                Degree, Institution (Year of Graduation)

                [CERTIFICATIONS]
                • Certification 1
                • Certification 2
                ...

                Ensure each section is clearly labeled with square brackets [LIKE THIS].
            """}
        ]

        full_response = ""
        for response in client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            timeout=180,
            stream=True
        ):
            if response.choices[0].delta.content is not None:
                full_response += response.choices[0].delta.content

        # Log the full response for debugging
        logging.info(f"Full API response: {full_response}")

        # Create a PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
        styles.add(ParagraphStyle(name='Left', alignment=TA_LEFT))
        styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='Heading', fontSize=14, spaceAfter=6))
        styles.add(ParagraphStyle(name='Bullet', leftIndent=20, spaceBefore=0, spaceAfter=0))

        story = []

        # Split the response into sections
        sections = re.split(r'\[([^\]]+)\]', full_response)

        for i in range(1, len(sections), 2):
            if i < len(sections):
                title = sections[i].strip()
                content = sections[i+1].strip() if i+1 < len(sections) else ""

                story.append(Paragraph(title, styles['Heading']))
                
                if title == "SKILLS" or title == "CERTIFICATIONS":
                    for item in content.split('•'):
                        if item.strip():
                            story.append(Paragraph(f"• {item.strip()}", styles['Bullet']))
                elif title == "WORK EXPERIENCE":
                    experiences = content.split('\n\n')
                    for exp in experiences:
                        lines = exp.split('\n')
                        story.append(Paragraph(lines[0], styles['Left']))
                        for line in lines[1:]:
                            if line.strip().startswith('•'):
                                story.append(Paragraph(line.strip(), styles['Bullet']))
                            else:
                                story.append(Paragraph(line.strip(), styles['Left']))
                else:
                    for para in content.split('\n'):
                        if para.strip():
                            story.append(Paragraph(para.strip(), styles['Justify']))
                
                story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name='optimized_resume.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        logging.error(f"An error occurred during CV optimization: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))