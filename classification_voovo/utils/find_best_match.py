# ========================
#     File not in use!    
# ========================

import json
import google.generativeai as genai
from typing import List, Dict, Any
import numpy as np
import os
from dotenv import load_dotenv

from paragraph_splitter import get_raw_text_from_pdf, split_text_into_paragraphs_with_gemini
from generate_embeddings import generate_embeddings_with_gemini

load_dotenv()
GENERATED_QUERIES_CACHE_PATH = "data/generated_queries_cache.json"

def load_cache(cache_path: str) -> Dict:
    """Loads the generated queries cache file. Returns an empty dict if not found."""
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"Warning: Could not read or decode cache file at {cache_path}. Starting fresh.")
        return {}

def save_cache(cache_path: str, cache_data: Dict):
    """Saves the cache data to the specified file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)

def generate_and_cache_all_queries(all_topic_definitions: List[Dict]) -> Dict:
    """
    Generates rich queries for all topic/subtopic pairs, using a cache to avoid re-generation.
    """
    print(f"\n--- Phase: Generating Rich Queries (using cache at '{GENERATED_QUERIES_CACHE_PATH}') ---")
    cache = load_cache(GENERATED_QUERIES_CACHE_PATH)
    
    genai.configure()
    model = genai.GenerativeModel('gemini-2.5-flash')

    for topic_info in all_topic_definitions:
        main_topic = topic_info.get('mainTopic')
        if main_topic not in cache:
            cache[main_topic] = {}

        for sub_topic in topic_info.get('subTopics', []):
            if sub_topic in cache[main_topic]:
                print(f"Cache HIT for: '{main_topic}' -> '{sub_topic}'")
                continue
            
            print(f"Cache MISS. Generating text for: '{main_topic}' -> '{sub_topic}'...")
            prompt = f"""
            You are a system that generates ideal example texts for search queries. 
            Your only function is to output a single, well-written paragraph in Hungarian.

            TASK:
            Based on the provided TOPIC and SUBTOPIC, generate a single, detailed paragraph. 
            This paragraph should be a perfect example of the kind of text a user would be thrilled 
            to find when searching for the subtopic within the main topic.

            TOPIC: "{main_topic}"
            SUBTOPIC: "{sub_topic}"
            
            RULES:
            - Your output MUST be the generated paragraph and nothing else.
            - DO NOT write any explanation, introduction, or meta-commentary.
            - The output must be written entirely in Hungarian.

            OUTPUT THE PARAGRAPH NOW:
            """
            try:
                response = model.generate_content(prompt)
                generated_text = response.text.strip()
                
                cache[main_topic][sub_topic] = generated_text
                save_cache(GENERATED_QUERIES_CACHE_PATH, cache) # Save progress immediately
                print(f"Successfully generated and cached text.")

            except Exception as e:
                print(f"An error occurred during API call: {e}")

    print("\n--- All rich queries are now generated and cached. ---")
    return cache

def find_best_match_for_query(
    query_text: str, 
    document_embeddings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Finds the best matching document paragraph for a single, pre-generated query text."""
    try:
        query_embedding_result = genai.embed_content(
            model='gemini-embedding-001',
            content=query_text,
            task_type="CLASSIFICATION"
        )
        query_vector = np.array(query_embedding_result['embedding'])
    except Exception as e:
        return {"error": f"Failed to generate query embedding: {e}"}

    best_match = {"score": -1.0, "text": None}
    for item in document_embeddings:
        doc_vector = item['vector']
        
        dot_product = np.dot(query_vector, doc_vector)
        norm_query = np.linalg.norm(query_vector)
        norm_doc = np.linalg.norm(doc_vector)
        similarity = dot_product / (norm_query * norm_doc)
        
        if similarity > best_match["score"]:
            best_match["score"] = similarity
            best_match["text"] = item["text"]
            
    return best_match

if __name__ == "__main__":
    pdf_file_path = "data/3.Anglia és Franciaország 10–15. században .docx.pdf"
    print(f"--- Starting Full Pipeline for '{pdf_file_path}' ---")
    raw_text = get_raw_text_from_pdf(pdf_file_path)
    if not raw_text: exit("Exiting: Could not read PDF.")
    
    paragraphs = split_text_into_paragraphs_with_gemini(raw_text)
    if not paragraphs: exit("Exiting: Could not split text into paragraphs.")

    print("\n--- Phase: Generating Document Embeddings ---")
    document_embeddings = generate_embeddings_with_gemini(
        paragraphs, model_name='gemini-embedding-001'
    )
    if not document_embeddings: exit("Exiting: Could not generate document embeddings.")
    print(f"\n--- Document Processing Complete. Found {len(document_embeddings)} embeddable paragraphs. ---")

    json_path = "data/main_topics_subtopics.json"
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        all_topic_definitions = topics_data.get('topics', [])
        if not all_topic_definitions: exit("Exiting: JSON file is empty or missing 'topics' key.")
    except FileNotFoundError: exit(f"Exiting: JSON file not found at '{json_path}'")
    except json.JSONDecodeError: exit(f"Exiting: Could not decode JSON. Check syntax.")

    all_generated_queries = generate_and_cache_all_queries(all_topic_definitions)

    print("\n\n--- Phase: Finding Best Matches for All Topics ---")
    for main_topic, sub_topics_dict in all_generated_queries.items():
        print(f"\n{'='*60}\nPROCESSING MAIN TOPIC: {main_topic}\n{'='*60}")
        for sub_topic, query_text in sub_topics_dict.items():
            print(f"\n--- Finding match for Subtopic: '{sub_topic}' ---")
            print(f"Using cached query: \"{query_text}\"")
            
            result = find_best_match_for_query(query_text, document_embeddings)

            if "error" in result:
                print(f"An error occurred: {result['error']}")
            else:
                print(f"Best Matching Paragraph (Score: {result['score']:.4f}): '{result['text']}'")