import pandas as pd
import google.generativeai as genai  # Fixed import
import os
from dotenv import load_dotenv
from prompts import GENERAL_PROMPT, DYNAMIC_PROMPT
from functools import lru_cache

load_dotenv()  # Load environment variables from .env file

class ChatbotHelper:
    """
    LLM-based chatbot using Gemini API for dynamic data analysis and preprocessing suggestions.
    Provides simple one-line responses based on user queries and data sample.
    """
    def __init__(self, data: pd.DataFrame, task_type: str = 'regression'):
        self.data = data  # Use full data passed
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Adjusted to valid model; change if needed
        self.task_type = task_type
        self.precomputed_suggestions = None  # To store precomputed results

    def _get_data_sample_text(self) -> str:
        """
        Generate a text representation of a sample of the data (30% or max 2000 rows).
        """
        sample_size = min(2000, max(1, int(len(self.data) * 0.3)))  # Ensure at least 1 if data is small
        sample_df = self.data.sample(n=sample_size, random_state=42)
        return sample_df.to_csv(index=False)

    @lru_cache(maxsize=32)  # Cache based on user_query
    def _query_llm(self, prompt: str) -> str:
        """
        Query Gemini with a given prompt (cached for repeated calls).
        """
        try:
            response = self.model.generate_content(prompt)
            if response is None or response.text is None:
                return "Error: No response received from the AI model."
            return response.text.strip()
        except Exception as e:
            return f"Error: Failed to get AI guidance - {str(e)}"

    def precompute_suggestions(self) -> dict:
        """
        Precompute all common suggestions in one LLM call after data upload.
        """
        if self.precomputed_suggestions:
            return self.precomputed_suggestions  # Return cached if already computed

        data_sample = self._get_data_sample_text()
        prompt = GENERAL_PROMPT.format(task_type=self.task_type, data_sample=data_sample)
        response_text = self._query_llm(prompt)

        # Parse the response (assuming separated by '||')
        parts = response_text.split('||')
        suggestions = {
            'target': parts[0].strip() if len(parts) > 0 else "No suggestion available.",
            'drops': parts[1].strip() if len(parts) > 1 else "No suggestion available.",
            'encoding': parts[2].strip() if len(parts) > 2 else "No suggestion available.",
            'other': parts[3].strip() if len(parts) > 3 else "No other tips."
        }
        self.precomputed_suggestions = suggestions
        return suggestions

    def get_guidance(self, user_query: str) -> str:
        """
        Get simple guidance or analysis based on the user's query using LLM (for dynamic sidebar queries).
        """
        data_sample = self._get_data_sample_text()
        prompt = DYNAMIC_PROMPT.format(user_query=user_query, data_sample=data_sample)
        return self._query_llm(prompt)