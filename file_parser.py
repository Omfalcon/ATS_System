import PyPDF2
from docx import Document
import pdfplumber
import re


def extract_text_from_file(file):
    filename = file.filename.lower()
    file.seek(0)

    try:
        if filename.endswith('.pdf'):
            text = ""
            # pdfplumber se try karo
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
            except:
                pass

            # Fallback: PyPDF2
            if not text.strip():
                file.seek(0)
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
                except:
                    pass

            return text.strip()

        elif filename.endswith(('.docx', '.doc')):
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text.strip()

        elif filename.endswith('.txt'):
            return file.read().decode('utf-8').strip()

        else:
            return ""

    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\-\+\&]', ' ', text)
    return text.strip()