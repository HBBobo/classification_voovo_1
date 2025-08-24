import json
import google.generativeai as genai
from typing import List, Dict
import os
import asyncio
from dotenv import load_dotenv

# A többi import változatlan...
from utils.paragraph_splitter import split_text_into_paragraphs_with_gemini_async
from config import constants

load_dotenv()

# ... a load_cache és save_cache függvények változatlanok ...
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

# --- ÚJ, ROBUZTUSABB FÜGGVÉNY ---
async def score_paragraph_universally_async(paragraph: str, all_subtopic_combinations: List[str]) -> Dict:
    """
    Uses Gemini to generate a relevance score for a paragraph against subtopic combinations.
    Includes robust error handling with retry logic for API stability issues.
    """
    max_retries = 3
    delay = 2  # Kezdeti várakozás másodpercben

    for attempt in range(max_retries):
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
            5. If the paragraph is nonsensical, irrelevant to all topics, or you cannot analyze it for safety reasons, return an empty scores object like this: {{"scores": {{}}}}. DO NOT return an error message.

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
            
            model = genai.GenerativeModel(constants.ORACLE_MODEL)
            # A biztonsági beállításokat enyhébbre vesszük, hogy csökkentsük a téves blokkolásokat.
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = await model.generate_content_async(prompt, safety_settings=safety_settings)
            
            # Előfeldolgozás a JSON hibák csökkentésére
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            
            if not cleaned_response:
                print(f"    -> Warning: Received empty response for paragraph '{paragraph[:50]}...'. Skipping.")
                return {"text": paragraph, "result": {"scores": {}}}
                
            result = json.loads(cleaned_response)
            return {"text": paragraph, "result": result}

        except json.JSONDecodeError:
            print(f"    -> Error: Failed to decode JSON for paragraph '{paragraph[:50]}...'. Response was: '{cleaned_response}'")
            # Ne próbálkozzunk újra JSON hiba esetén, mert valószínűleg a modell válasza a hibás.
            return {"text": paragraph, "result": {"scores": {}}}
            
        except Exception as e:
            print(f"    -> API Error on attempt {attempt + 1}/{max_retries} for paragraph '{paragraph[:50]}...': {e}")
            if attempt < max_retries - 1:
                print(f"    -> Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"    -> Max retries reached. Skipping this paragraph.")
                return {"text": paragraph, "result": {"scores": {}}}

    # Ez a rész elvileg sosem érhető el, de a biztonság kedvéért itt van
    return {"text": paragraph, "result": {"scores": {}}}


# ... az `async def main():` és `if __name__ == "__main__":` blokkok változatlanok maradnak.
async def main():
    """Main async function to generate training data."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    with open(constants.TOPICS_FILE_PATH, 'r', encoding='utf-8') as f:
        all_topics = json.load(f)['topics']
    
    all_subtopic_combinations = []
    for topic in all_topics:
        for sub_topic in topic['subTopics']:
            all_subtopic_combinations.append(f"{topic['mainTopic']} -> {sub_topic}")

    from utils.paragraph_splitter import get_raw_text_from_pdf
    pdf_file_path = "data/3.Anglia és Franciaország 10–15. században .docx.pdf"
    raw_text = get_raw_text_from_pdf(pdf_file_path)
    paragraphs = await split_text_into_paragraphs_with_gemini_async(raw_text)

    scored_data_cache = load_cache(constants.TRAINING_DATASET_PATH)
    print(f"Starting universal paragraph scoring. Found {len(scored_data_cache)} items in cache.")
    
    paragraphs_to_score = [p for p in paragraphs if p not in scored_data_cache]
    
    if not paragraphs_to_score:
        print("All paragraphs are already in the cache. Nothing to do.")
    else:
        print(f"Cache MISS for {len(paragraphs_to_score)} paragraphs. Scoring with Gemini API in parallel...")
        tasks = [score_paragraph_universally_async(p, all_subtopic_combinations) for p in paragraphs_to_score]
        results = await asyncio.gather(*tasks)

        newly_scored_count = 0
        for res in results:
            paragraph_text = res['text']
            scored_data = res['result']
            if scored_data and scored_data.get('scores'):
                scored_data_cache[paragraph_text] = {
                    "text": paragraph_text,
                    "scores": scored_data['scores']
                }
                newly_scored_count += 1
        
        if newly_scored_count > 0:
            save_cache(constants.TRAINING_DATASET_PATH, scored_data_cache)
            print(f"  -> Successfully scored and saved {newly_scored_count} new items to cache.")
            
    print(f"\nScoring complete. The full dataset is at '{constants.TRAINING_DATASET_PATH}'")

if __name__ == "__main__":
    asyncio.run(main())