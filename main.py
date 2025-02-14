import os
import json
import argparse
import google.generativeai as genai
from config import GEMINI_API_KEY
from typing import List, Dict

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
if not GEMINI_API_KEY:
    print("Error: Gemini API key not found. Set the GEMINI_API_KEY environment variable.")
    exit()

model = genai.GenerativeModel('gemini-pro')

def search_with_gemini(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """ Queries Gemini API to generate and refine search queries, then retrieves relevant results. """
    try:
        # Stronger prompt to ensure relevance
        prompt = f"""
        Given the search query: "{query}", generate up to 3 precise sub-queries that are strictly related.
        Only return queries directly relevant to "{query}".
        Example:
        - If the input is 'google', return queries about Google Search, Google products, or Google history.
        - If the input is 'New York', return queries about its history, geography, or attractions.
        
        Return only the queries, one per line.
        """
        response = model.generate_content(prompt)
        search_queries = response.text.strip().split('\n')
        search_queries = [q.strip() for q in search_queries if q.strip()]

        # Fetch and combine results from refined queries
        all_results = []
        for search_query in search_queries:
            search_results = generate_results_with_gemini(search_query, num_results=2)
            all_results.extend(search_results)

        return all_results[:max_results]
    except Exception as e:
        print(f"Error: {e}")
        return []

def generate_results_with_gemini(query: str, num_results: int = 2):
    """ Generates search results using Gemini AI and ensures valid JSON. """
    try:
        prompt = f"""
        Generate {num_results} search results for query: {query}.
        Response format: JSON array of objects with 'title', 'description', 'link'.
        Example:
        [
            {{"title": "Best Smartphones 2024", "description": "A review of the top smartphones.", "link": "https://example.com"}},
            {{"title": "Budget Laptops Guide", "description": "Affordable laptops for students.", "link": "https://example.com"}}
        ]
        """
        response = model.generate_content(prompt)

        # Ensure response is valid
        if not response or not response.text:
            print(f"Error: Empty response from Gemini API for query: {query}")
            return []

        raw_text = response.text.strip()

        # Fix: Remove Markdown formatting (` ```json ... ``` `)
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]  # Remove ```json (first 7 characters)
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]  # Remove closing ```

        try:
            results = json.loads(raw_text.strip())  # Convert to JSON
            if not isinstance(results, list):
                print(f"Error: Expected a JSON list but got: {raw_text}")
                return []
            return results
        except json.JSONDecodeError:
            print(f"Error: Gemini API returned invalid JSON for query: {query}")
            print(f"Raw response: {raw_text}")  # Debug print
            return []

    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, nargs="?", help="Search query")
    args = parser.parse_args()

    # Ask for query if not provided
    if not args.query:
        args.query = input("Enter your search query: ").strip()

    results = search_with_gemini(args.query)

    if not results:
        print("No results found.")
        return

    print("\n🔍 Search Results:\n" + "="*40)
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] {res['title']}\n   📖 {res['description']}\n   🔗 {res['link']}\n" + "-"*40)

if __name__ == "__main__":
    main()
