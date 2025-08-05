import os
import pdfplumber
import google.generativeai as genai

from dotenv import load_dotenv
import os
load_dotenv()

def get_raw_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF into a single raw string.
    
    Args:
        pdf_path: The path to the PDF file to be processed.

    Returns:
        The entire text content of the PDF as a single string.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = " ".join(
                page.extract_text(x_tolerance=1) #.replace('\n', ' ')
                for page in pdf.pages if page.extract_text()
            )
        return full_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def split_text_into_paragraphs_with_gemini(raw_text: str) -> list[str]:
    """
    Intelligently splits a raw text into paragraphs using the Gemini API.
    
    Args:
        raw_text: The raw text to be processed.

    Returns:
        A list where each element is a semantically coherent paragraph.
    """
    if not raw_text:
        print("The text to be processed is empty.")
        return []

    try:
        genai.configure()
        
        model = genai.GenerativeModel('gemini-2.5-flash') #gemini-2.5-flash-lite

        prompt = f"""
        You are a text processing expert specializing in making OCR-scanned, 
        unstructured Hungarian documents readable.

        TASK:
        Break down the given raw text into logical, semantically coherent paragraphs.
        The goal is to make the output a well-readable, structured text.

        RULES:
        1.  PRESERVE ORIGINAL TEXT: Do not change, rewrite, or omit anything!
        2.  SEMANTIC GROUPING: Keep sentences belonging to the same train of thought together, 
            even if they are separated by line breaks in the original text.
        3.  HEADINGS AND LIST ITEMS: Treat headings (e.g., "Concept:", "battles:") and their associated 
            short list items as a single logical block.
        4.  Don't split the headings into a separated part!
        5.  SPECIAL DELIMITER: Mark the end of each paragraph with a unique delimiter: |||---|||
            DO NOT use a simple line break, only and exclusively this delimiter!

        RAW TEXT:
        ---
        {raw_text}
        ---
        """

        print("Sending text to Gemini API for processing...")
        response = model.generate_content(prompt)
        
        cleaned_text = response.text
        
        paragraphs = [p.strip() for p in cleaned_text.split("|||---|||") if p.strip()]
        
        return paragraphs

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return []

if __name__ == "__main__":

    if True:
        pdf_file_path = "data/3.Anglia és Franciaország 10–15. században .docx.pdf"
        
        print(f"1. Extracting raw text from '{pdf_file_path}'...")
        raw_document_text = get_raw_text_from_pdf(pdf_file_path)

        if raw_document_text:
            print("2. Intelligently splitting text into paragraphs using Gemini...")
            final_paragraphs = split_text_into_paragraphs_with_gemini(raw_document_text)
            
            if final_paragraphs:
                print("\n--- Intelligently Split Paragraphs ---")
                for i, para in enumerate(final_paragraphs, 1):
                    print(f"[{i}] {para}\n")
                print(f"A total of {len(final_paragraphs)} paragraphs were identified.")
            else:
                print("Failed to split text into paragraphs using the Gemini API.")