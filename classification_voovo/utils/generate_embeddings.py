import google.generativeai as genai
from typing import List, Dict, Any
import numpy as np
import os
from dotenv import load_dotenv

from paragraph_splitter import get_raw_text_from_pdf, split_text_into_paragraphs_with_gemini

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

def generate_embeddings_with_gemini(
    paragraphs: List[str], 
    model_name: str = 'gemini-embedding-001'
) -> List[Dict[str, Any]]:
    """
    Generates embedding vectors for a list of paragraphs using a Gemini model.

    Args:
        paragraphs: A list of text paragraphs.
        model_name: The name of the Gemini embedding model to use.

    Returns:
        A list of dictionaries, where each dictionary contains the original
        paragraph text and its corresponding numpy embedding vector.
    """
    if not paragraphs:
        print("Cannot generate embeddings: The paragraph list is empty.")
        return []
    
    print(f"Generating embeddings using Gemini model: '{model_name}'...")
    try:
        genai.configure()

        result = genai.embed_content(
            model=model_name,
            content=paragraphs,
            task_type="CLASSIFICATION"
        )
        
        embeddings = result['embedding']

        embedded_paragraphs = [
            {
                "text": para,
                "vector": np.array(vec)
            } for para, vec in zip(paragraphs, embeddings)
        ]
    
        return embedded_paragraphs

    except Exception as e:
        print(f"An error occurred during Gemini embedding generation: {e}")
        return []


if __name__ == "__main__":
    pdf_file_path = "data/3.Anglia és Franciaország 10–15. században .docx.pdf"
        
    print(f"1. Extracting raw text from '{pdf_file_path}'...")
    raw_document_text = get_raw_text_from_pdf(pdf_file_path)

    if not raw_document_text:
        print("Could not proceed, raw text is empty.")
    else:
        print("2. Intelligently splitting text into paragraphs using Gemini...")
        final_paragraphs = split_text_into_paragraphs_with_gemini(raw_document_text)

        if not final_paragraphs:
            print("Could not proceed, no paragraphs were generated.")
        else:
            print("\n3. Generating embedding vectors for the paragraphs using Gemini...")
            embedding_data = generate_embeddings_with_gemini(final_paragraphs)

            if embedding_data:
                print("\n--- Embedding Generation Complete ---")
                print(f"Successfully generated embeddings for {len(embedding_data)} paragraphs.")
                
                vector_dimension = len(embedding_data[0]['vector'])
                print(f"Vector dimension: {vector_dimension}")

                print("\nSample of the first embedded paragraph:")
                print(f"Text: {embedding_data[0]['text']}")
                print(f"Vector (first 5 elements): {embedding_data[0]['vector'][:5]}")
            else:
                print("Failed to generate embeddings.")