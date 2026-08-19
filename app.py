import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file
import pdfplumber
from fpdf import FPDF
import docx
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

basedir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(basedir, '.env')
load_dotenv(dotenv_path)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise RuntimeError('Missing GROQ_API_KEY environment variable. Set it before running the app.')

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model='llama-3.1-8b-instant',
    temperature=0.0
)

DIFFICULTY_INSTRUCTIONS = {
    'easy': 'Difficulty: EASY — Ask straightforward recall questions about key facts and definitions.',
    'medium': 'Difficulty: MEDIUM — Ask questions that require understanding and application of concepts.',
    'hard': 'Difficulty: HARD — Ask challenging questions requiring deep understanding and analysis.'
}

mcq_prompt = PromptTemplate(
    input_variables=['context', 'num_questions', 'difficulty_instruction'],
    template='''
You are an AI assistant helping the user generate multiple-choice questions (MCQs) from the text below:

Text:
{context}

{difficulty_instruction}

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
'''
)

mcq_chain = LLMChain(llm=llm, prompt=mcq_prompt)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            return ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    if ext == 'docx':
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    if ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None


# MCQ generation
def generate_mcqs_with_langchain(text, num_questions, difficulty='medium'):
    diff_instruction = DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS['medium'])
    response = mcq_chain.run({
        "context": text,
        "num_questions": num_questions,
        "difficulty_instruction": diff_instruction
    })
    return response.strip()


def parse_mcqs(mcq_text):
    """Parse raw LLM MCQ text into a structured list of MCQ objects."""
    if not mcq_text:
        return []

    chunks = re.split(r'(?:###\s*MCQ|##\s*MCQ|--\s*MCQ|\bMCQ\s*\d*:?)', mcq_text, flags=re.IGNORECASE)
    mcqs = []
    q_index = 0

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        q_match = re.search(r'(?:Question\s*\d*:\s*|\*\*\s*Question\s*\d*:\s*\*\*|\d+\.\s+)?(.*?)(?=(?:\n\s*(?:[A-D][\.\)]|\([A-D]\))\s+))', chunk, re.DOTALL | re.IGNORECASE)
        if q_match:
            question_text = q_match.group(1).strip()
            question_text = re.sub(r'^(?:Question\s*\d*:\s*|\*\*\s*Question\s*\d*:\s*\*\*)+', '', question_text, flags=re.IGNORECASE).strip()
            question_text = question_text.strip('*').strip()
        else:
            lines = [line.strip() for line in chunk.split('\n') if line.strip()]
            if lines:
                question_text = re.sub(r'^(?:Question\s*\d*:\s*|\*\*\s*Question\s*\d*:\s*\*\*)+', '', lines[0], flags=re.IGNORECASE).strip()
            else:
                continue

        options = []
        for letter in ['A', 'B', 'C', 'D']:
            opt_pattern = rf'(?:^|\n)\s*(?:\(?{letter}\)|\(?{letter}[\.\:\)])\s*([^\n]+(?:\n(?!\s*(?:\(?[A-D][\)\.\:]|Correct\s+Answer|Answer))[^\n]+)*)'
            opt_match = re.search(opt_pattern, chunk, re.IGNORECASE)
            if opt_match:
                opt_text = opt_match.group(1).strip().strip('*').strip()
                options.append({
                    'letter': letter,
                    'text': opt_text,
                    'id': f'q_{q_index}_opt_{letter}'
                })

        correct_match = re.search(r'(?:Correct\s*Answer|Answer|Correct)\s*:\s*(?:\(?([A-D])\)?|[A-D]\)?\s*([A-D]))', chunk, re.IGNORECASE)
        correct_letter = ''
        if correct_match:
            correct_letter = (correct_match.group(1) or correct_match.group(2) or '').upper()
        else:
            ca_search = re.search(r'(?:Correct\s*Answer|Answer|Correct)[^\n]*?([A-D])\b', chunk, re.IGNORECASE)
            if ca_search:
                correct_letter = ca_search.group(1).upper()

        if question_text and len(options) >= 2:
            mcqs.append({
                'q_num': q_index,
                'question': question_text,
                'options': options,
                'correct': correct_letter
            })
            q_index += 1

    return mcqs


# Save MCQs to text file
def save_mcqs_to_file(mcqs, filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(mcqs)
    return path


# Save MCQs to PDF
def create_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for mcq in mcqs.split("### MCQ"):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)

    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    pdf.output(path)
    return path

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_mcqs():
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        text = extract_text_from_file(file_path)
        if text:
            num_questions = int(request.form['num_questions'])
            difficulty = request.form.get('difficulty', 'medium')
            if difficulty not in DIFFICULTY_INSTRUCTIONS:
                difficulty = 'medium'
            mcq_text = generate_mcqs_with_langchain(text, num_questions, difficulty)
            mcqs = parse_mcqs(mcq_text)
            return render_template('results.html',
                                   mcqs=mcqs,
                                   mcq_text=mcq_text,
                                   difficulty=difficulty,
                                   results=None,
                                   score=0,
                                   total=len(mcqs))

    return 'Invalid file format or upload error.'


@app.route('/check_answers', methods=['POST'])
def check_answers():
    user_answers = request.form.to_dict()
    mcq_text = request.form.get('mcq_text')
    mcqs = parse_mcqs(mcq_text)

    results = {}
    score = 0
    for mcq in mcqs:
        user_answer = request.form.get(f"q_{mcq['q_num']}", '').upper()
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
                           difficulty=request.form.get('difficulty', 'medium'),
                           results=results,
                           score=score,
                           total=len(mcqs))


@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    mcq_text = request.form.get('mcq_text', '')
    if not mcq_text:
        return 'No MCQ content provided.', 400
    pdf_path = create_pdf(mcq_text, 'generated_mcqs.pdf')
    return send_file(pdf_path, as_attachment=True, download_name='generated_mcqs.pdf')


if __name__ == '__main__':
    app.run(debug=True)

