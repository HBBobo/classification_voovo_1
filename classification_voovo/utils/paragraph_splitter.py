import os
import pdfplumber
import google.generativeai as genai
import asyncio
from dotenv import load_dotenv
from config import constants

load_dotenv()

def get_raw_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF into a single raw string.
    """
    # ... (This function remains synchronous)
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

async def split_text_into_paragraphs_with_gemini_async(raw_text: str) -> list[str]:
    """
    Intelligently splits a raw text into paragraphs using the Gemini API, asynchronously.
    """
    if not raw_text:
        print("The text to be processed is empty.")
        return []

    try:
        model = genai.GenerativeModel(constants.PARAGRAPH_SPLITTER_MODEL)
        prompt = f"""
        # ... (PROMPT REMAINS THE SAME) ...
        """

        print("Sending text to Gemini API for paragraph splitting...")
        response = await model.generate_content_async(prompt)
        
        cleaned_text = response.text
        paragraphs = [p.strip() for p in cleaned_text.split("|||---|||") if p.strip()]
        return paragraphs

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return []

async def main_test():
    """Async main function for testing this module."""
    pdf_file_path = "data/3.Anglia és Franciaország 10–15. században .docx.pdf"
    
    print(f"1. Extracting raw text from '{pdf_file_path}'...")
    raw_document_text = get_raw_text_from_pdf(pdf_file_path)

    if raw_document_text:
        print("2. Intelligently splitting text into paragraphs using Gemini...")
        final_paragraphs = await split_text_into_paragraphs_with_gemini_async(raw_document_text)
        
        if final_paragraphs:
            print("\n--- Intelligently Split Paragraphs ---")
            for i, para in enumerate(final_paragraphs, 1):
                print(f"[{i}] {para}\n")
            print(f"A total of {len(final_paragraphs)} paragraphs were identified.")
        else:
            print("Failed to split text into paragraphs using the Gemini API.")

if __name__ == "__main__":
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    asyncio.run(main_test())