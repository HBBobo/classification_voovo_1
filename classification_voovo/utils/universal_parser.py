import os
from typing import List
import pdfplumber
from docx import Document
from pptx import Presentation

from utils.paragraph_splitter import split_text_into_paragraphs_with_gemini_async

async def _get_paragraphs_from_pdf_async(file_path: str) -> List[str]:
    """Extracts raw text from a PDF and uses Gemini to split it into paragraphs, asynchronously."""
    print(f"  -> Using PDF method with Gemini splitter...")
    try:
        # pdfplumber is sync, which is fine
        with pdfplumber.open(file_path) as pdf:
            raw_text = " ".join(
                page.extract_text(x_tolerance=1).replace('\n', ' ')
                for page in pdf.pages if page.extract_text()
            )
        # The API call part is async
        return await split_text_into_paragraphs_with_gemini_async(raw_text)
    except Exception as e:
        print(f"    -> Error processing PDF {file_path}: {e}")
        return []

def _get_paragraphs_from_docx(file_path: str) -> List[str]:
    """Extracts paragraphs directly from a .docx file's structure."""
    # ... (This function remains synchronous)
    print("  -> Using DOCX method (direct structure read)...")
    try:
        doc = Document(file_path)
        return [p.text for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        print(f"    -> Error processing DOCX {file_path}: {e}")
        return []

def _get_paragraphs_from_pptx(file_path: str) -> List[str]:
    """Extracts text from each slide of a .pptx file. Each slide becomes one 'paragraph'."""
    print("  -> Using PPTX method (one text chunk per slide)...")
    try:
        prs = Presentation(file_path)
        slide_texts = []
        for slide in prs.slides:
            text_runs = []
            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        text_runs.append(run.text)
            
            slide_text = " ".join(text_runs).strip()
            if slide_text:
                slide_texts.append(slide_text)
        return slide_texts
    except Exception as e:
        print(f"    -> Error processing PPTX {file_path}: {e}")
        return []

async def extract_paragraphs_from_file_async(file_path: str) -> List[str]:
    """
    Universally extracts paragraphs from PDF, DOCX, or PPTX files, using async for PDFs.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    print(f"\nProcessing file: '{os.path.basename(file_path)}'")

    if extension == '.pdf':
        return await _get_paragraphs_from_pdf_async(file_path)
    elif extension == '.docx':
        return _get_paragraphs_from_docx(file_path)
    elif extension == '.pptx':
        return _get_paragraphs_from_pptx(file_path)
    else:
        print(f"  -> Unsupported file type: '{extension}'. Skipping.")
        return []

# The __main__ block for this file is omitted for brevity, as it's a library file.
# If testing is needed, it should be converted to async like the others.