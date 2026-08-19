import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
import pdfplumber
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
                'id': f"q{question_number + 1}_{option_letter}"
            })
        elif line.startswith('Correct Answer:'):
            current_mcq['correct'] = line.replace('Correct Answer:', '').strip()[0]

    if current_mcq:
        current_mcq['q_num'] = question_number
        mcqs.append(current_mcq)

    return mcqs


def generate_mcqs_with_langchain(text, num_questions, difficulty='medium'):
    difficulty = difficulty if difficulty in DIFFICULTY_INSTRUCTIONS else 'medium'
    response = mcq_chain.run({
        'context': text,
        'num_questions': num_questions,
        'difficulty_instruction': DIFFICULTY_INSTRUCTIONS[difficulty],
    })
    return response.strip()


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


if __name__ == '__main__':
    app.run(debug=True)
