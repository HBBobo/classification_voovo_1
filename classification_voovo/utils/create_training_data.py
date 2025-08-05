import json
import google.generativeai as genai
from typing import List, Dict
import os
import time
from dotenv import load_dotenv

from paragraph_splitter import get_raw_text_from_pdf, split_text_into_paragraphs_with_gemini

load_dotenv()
TRAINING_DATA_UNIVERSAL_CACHE_PATH = "data/training_dataset_universal_scores.json"

def load_cache(cache_path: str) -> Dict:
    """Loads the scored paragraphs cache file."""
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {item['text']: item for item in data}
    except (json.JSONDecodeError, IOError):
        print(f"Warning: Could not read or decode cache file at {cache_path}. Starting fresh.")
        return {}

def save_cache(cache_path: str, cache_data: Dict):
    """Saves the cache data to the specified file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(list(cache_data.values()), f, ensure_ascii=False, indent=4)

def score_paragraph_universally(paragraph: str, all_subtopic_combinations: List[str]) -> Dict:
    """
    Uses Gemini to generate a relevance score (0.0 to 1.0) for a paragraph
    against EVERY possible subtopic combination.
    """
    try:
        topic_options_string = "\n".join(f"- {combo}" for combo in all_subtopic_combinations)

        prompt = f"""
        You are a highly precise document analysis system. Your task is to read a paragraph of
        (maybe Hungarian) text and evaluate its relevance to a comprehensive list of topics
        on a fine-grained, continuous scale.

        TASK:
        For the given PARAGRAPH, you MUST assign a relevance score from 0.00 to 1.00 for 
        EVERY SINGLE topic combination in the provided list. Think carefully and provide
        nuanced, non-rounded scores.

        RULES:
        1. Your response MUST be a single, valid JSON object and nothing else.
        2. The JSON object must contain a single key: "scores".
        3. The value of "scores" must be a dictionary where the keys EXACTLY MATCH the topic
           combinations from the list, and the values are float numbers.
        4. DO NOT round your scores. A score of 0.7 is acceptable, but a score of 0.73 is better.

        EXAMPLE RESPONSE FORMAT (Note the non-rounded, continuous values):
        {{
          "scores": {{
            "Anglia és Franciaország 10–15. században -> Évszámok": 0.95,
            "Anglia és Franciaország 10–15. században -> Fogalmak, személyek": 0.88,
            "A Karoling birodalom felbomlása... -> Évszámok": 0.0,
            "A Karoling birodalom felbomlása... -> Eseménytört. kiegészítés": 0.12
          }}
        }}

        PARAGRAPH TO SCORE:
        ---
        {paragraph}
        ---

        LIST OF ALL TOPIC COMBINATIONS TO SCORE AGAINST:
        ---
        {topic_options_string}
        ---

        YOUR JSON RESPONSE:
        """
        
        genai.configure()
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned_response)
        return result

    except Exception as e:
        print(f"An error occurred during scoring: {e}")
        return {"scores": {}}

if __name__ == "__main__":
    json_path = "data/main_topics_subtopics.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        all_topics = json.load(f)['topics']
    
    all_subtopic_combinations = []
    for topic in all_topics:
        for sub_topic in topic['subTopics']:
            all_subtopic_combinations.append(f"{topic['mainTopic']} -> {sub_topic}")

    pdf_file_path = "data/3.Anglia és Franciaország 10–15. században .docx.pdf"
    raw_text = get_raw_text_from_pdf(pdf_file_path)
    paragraphs = split_text_into_paragraphs_with_gemini(raw_text)

    scored_data_cache = load_cache(TRAINING_DATA_UNIVERSAL_CACHE_PATH)
    
    print(f"Starting universal paragraph scoring. Found {len(scored_data_cache)} items in cache.")
    
    for i, p in enumerate(paragraphs):
        print(f"Processing paragraph {i+1}/{len(paragraphs)}...")
        
        if p in scored_data_cache:
            print("  -> Cache HIT. Skipping.")
            continue
        
        print("  -> Cache MISS. Scoring with Gemini API...")
        scored_data = score_paragraph_universally(p, all_subtopic_combinations)
        
        if scored_data and scored_data['scores']:
            scored_data_cache[p] = {
                "text": p,
                "scores": scored_data['scores']
            }
            save_cache(TRAINING_DATA_UNIVERSAL_CACHE_PATH, scored_data_cache)
            print("  -> Successfully scored and saved to cache.")
        else:
            print("  -> Failed to score this paragraph.")
            
    print(f"\nScoring complete. The full dataset is at '{TRAINING_DATA_UNIVERSAL_CACHE_PATH}'")