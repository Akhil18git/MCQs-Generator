import os
import sys
from dotenv import load_dotenv
import pdfplumber
import docx
from fpdf import FPDF
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

UPLOAD_FILE = 'file-here'
NUM_QUESTIONS = 5
OUTPUT_FOLDER = 'results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise RuntimeError('Missing GROQ_API_KEY environment variable. Set it before running the app.')

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model='llama-3.3-70b-versatile',
    temperature=0.0
)

mcq_prompt = PromptTemplate(
    input_variables=['context', 'num_questions'],
    template='''
You are an AI assistant helping the user generate multiple-choice questions (MCQs) from the text below:

Text:
{context}

Generate {num_questions} MCQs. Each should include:
- A clear question
- Four answer options labeled A, B, C, and D
- The correct answer clearly indicated at the end

Format:
-- MCQ
Question: [question]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [correct option]
'''
)

mcq_chain = mcq_prompt | llm


def extract_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            return ''.join([p.extract_text() for p in pdf.pages if p.extract_text()])
    if ext == 'docx':
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    if ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    raise ValueError(f"Unsupported file type: .{ext}. Supported formats: PDF, DOCX, TXT")


def save_txt(mcqs, filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(mcqs)


def save_pdf(mcqs, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    for mcq in mcqs.split('-- MCQ'):
        if mcq.strip():
            pdf.multi_cell(0, 10, mcq.strip())
            pdf.ln(5)
    path = os.path.join(OUTPUT_FOLDER, filename)
    pdf.output(path)


def main():
    target_file = sys.argv[1] if len(sys.argv) > 1 else UPLOAD_FILE
    if target_file == 'file-here' or not os.path.exists(target_file):
        print("Error: No valid input file specified.")
        print("Usage:")
        print("  py main.py <path_to_file.pdf|docx|txt>")
        print("  Or update UPLOAD_FILE in main.py with your file path.")
        return

    text = extract_text(target_file)
    if not text:
        print('No text extracted from file.')
        return

    print(f"Generating {NUM_QUESTIONS} MCQs from {target_file}...")
    try:
        response = mcq_chain.invoke({'context': text, 'num_questions': NUM_QUESTIONS})
        mcqs = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    except Exception as e:
        error_msg = str(e)
        if '401' in error_msg or 'invalid_api_key' in error_msg.lower():
            print("\n[Groq Auth Error] Your GROQ_API_KEY in .env is invalid or expired.")
            print("Please create a free API key at https://console.groq.com/keys and update .env.")
        else:
            print(f"\nError: {e}")
        return

    base_name = os.path.basename(target_file).rsplit('.', 1)[0]
    txt_filename = f'generated_mcqs_{base_name}.txt'
    pdf_filename = f'generated_mcqs_{base_name}.pdf'
    save_txt(mcqs, txt_filename)
    save_pdf(mcqs, pdf_filename)
    print(f"MCQs successfully saved to:")
    print(f" - {os.path.join(OUTPUT_FOLDER, txt_filename)}")
    print(f" - {os.path.join(OUTPUT_FOLDER, pdf_filename)}")


if __name__ == '__main__':
    main()
