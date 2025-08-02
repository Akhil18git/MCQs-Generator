import os
from flask import Flask, render_template, request, send_file
import pdfplumber
import docx
from werkzeug.utils import secure_filename
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize LangChain LLM
llm = ChatGroq(
    api_key="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",
    model="llama-3.1-8b-instant",
    temperature=0.0
)

# LangChain prompt template
mcq_prompt = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""
You are an AI assistant helping the user generate multiple-choice questions (MCQs) from the text below:

Text:
{context}

Generate {num_questions} MCQs. Each should include:
- A clear question
- Four answer options labeled A, B, C, and D
- The correct answer clearly indicated at the end

Format each MCQ exactly like this:
### MCQ
Question: [question]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [correct option]
"""
)

mcq_chain = mcq_prompt | llm

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Text extraction
def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            return ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif ext == 'docx':
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

# Parse MCQs into a structured format
def parse_mcqs(mcq_text):
    mcqs = []
    current_mcq = {}
    question_number = 0
    
    for line in mcq_text.split('\n'):
        line = line.strip()
        if line.startswith('### MCQ'):
            if current_mcq:
                current_mcq['q_num'] = question_number
                mcqs.append(current_mcq)
                current_mcq = {}
                question_number += 1
        elif line.startswith('Question:'):
            current_mcq['question'] = line.replace('Question:', '').strip()
            current_mcq['options'] = []
            current_mcq['q_num'] = question_number
        elif line.startswith(('A)', 'B)', 'C)', 'D)')):
            option_text = line[3:].strip()
            option_letter = line[0]
            current_mcq['options'].append({
                'letter': option_letter,
                'text': option_text,
                'id': f"q{question_number+1}_{option_letter}"
            })
        elif line.startswith('Correct Answer:'):
            correct = line.replace('Correct Answer:', '').strip()[0]
            current_mcq['correct'] = correct
    
    if current_mcq:
        current_mcq['q_num'] = question_number
        mcqs.append(current_mcq)
    
    return mcqs

# Generate MCQs with LangChain
def generate_mcqs_with_langchain(text, num_questions):
    response = mcq_chain.invoke({"context": text, "num_questions": num_questions})
    return response.content.strip()

# Save MCQs to text file
def save_mcqs_to_file(mcqs, filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(mcqs)
    return path

# Save MCQs to PDF
'''def create_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("### MCQ"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)

    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    pdf.output(path)
    return path'''

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_mcqs():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = extract_text_from_file(file_path)
        if text:
            num_questions = int(request.form['num_questions'])
            mcq_text = generate_mcqs_with_langchain(text, num_questions)
            mcqs = parse_mcqs(mcq_text)

            # Save output
            #base_name = filename.rsplit('.', 1)[0]
            #txt_file = f"generated_mcqs_{base_name}.txt"
            #pdf_file = f"generated_mcqs_{base_name}.pdf"
            #save_mcqs_to_file(mcq_text, txt_file)
            #create_pdf(mcq_text, pdf_file)'''

            return render_template('results.html', 
                                 mcqs=mcqs, 
                                 mcq_text=mcq_text,
                                 #txt_filename=txt_file, 
                                 #pdf_filename=pdf_file,
                                 results=None,
                                 score=0,
                                 total=len(mcqs))

    return "Invalid file format or upload error."

@app.route('/check_answers', methods=['POST'])
def check_answers():
    user_answers = request.form.to_dict()
    mcq_text = request.form.get('mcq_text')
    mcqs = parse_mcqs(mcq_text)
    
    results = {}
    score = 0
    
    for mcq in mcqs:
        user_answer = request.form.get(f"q_{mcq['q_num']}", "").upper()
        is_correct = user_answer == mcq['correct']
        if is_correct:
            score += 1
        
        results[mcq['q_num']] = {
            'question': mcq['question'],
            'user_answer': user_answer,
            'correct_answer': mcq['correct'],
            'is_correct': is_correct,
            'options': mcq['options']
        }
    
    return render_template('results.html', 
                         mcqs=mcqs,
                         mcq_text=mcq_text,
                         results=results,
                         score=score,
                         total=len(mcqs))

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)