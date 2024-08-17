import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import logging

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

@app.route('/')
def home():
    return "Welcome to the CV Analysis API. The backend is running correctly!"

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        resume_text = data.get('resumeText')
        job_description = data.get('jobDescription')

        if not resume_text or not job_description:
            return jsonify({"error": "Please provide both CV text and job description."}), 400

        messages = [
            {"role": "system", "content": "You are an AI assistant that helps analyze CVs for job fit and skills."},
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

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            max_tokens=800
        )

        analysis = json.loads(response.choices[0].message.content)
        return jsonify(analysis)

    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}")
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize():
    try:
        data = request.json
        resume_text = data.get('resumeText')
        job_description = data.get('jobDescription')
        analysis = data.get('analysis')

        if not resume_text or not job_description or not analysis:
            return jsonify({"error": "Please provide CV text, job description, and analysis results."}), 400

        messages = [
            {"role": "system", "content": "You are an AI assistant that helps optimize CVs for specific job descriptions."},
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

                Provide the optimized CV in a clear, professional format.
                
                Format the response as a JSON object with the following structure:
                {{
                  "optimizedResume": string
                }}
            """}
        ]

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            max_tokens=1000
        )

        optimized_resume = json.loads(response.choices[0].message.content)
        return jsonify(optimized_resume)

    except Exception as e:
        logging.error(f"An error occurred during CV optimization: {str(e)}")
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))