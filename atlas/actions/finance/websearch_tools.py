import requests
from typing import Dict, Any

class WebSearchTool:
    GOOGLE_API_KEY = "your_google_api_key"
    GOOGLE_CX = "your_google_cx"
    PERPLEXITY_API_KEY = "your_perplexity_api_key"
    YOU_API_KEY = "your_you_api_key"

    def __init__(self):
        self._validate_api_keys()

    def _validate_api_keys(self):
        if not all([self.GOOGLE_API_KEY, self.GOOGLE_CX, self.PERPLEXITY_API_KEY, self.YOU_API_KEY]):
            raise ValueError("One or more API keys are invalid or missing.")

    def google_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a Google search.

        :param query: The search query string.
        :return: A dictionary containing the search results.
        """
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={self.GOOGLE_API_KEY}&cx={self.GOOGLE_CX}"
        response = requests.get(url)
        return response.json()

    def google_news_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a Google News search.

        :param query: The search query string.
        :return: A dictionary containing the search results.
        """
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.GOOGLE_API_KEY}"
        response = requests.get(url)
        return response.json()

    def wikipedia_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a Wikipedia search.

        :param query: The search query string.
        :return: A dictionary containing the search results.
        """
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
        response = requests.get(url)
        return response.json()

    def perplexity_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a Perplexity search.

        :param query: The search query string.
        :return: A dictionary containing the search results.
        """
        url = f"https://api.perplexity.ai/search?q={query}&key={self.PERPLEXITY_API_KEY}"
        response = requests.get(url)
        return response.json()

    def you_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a You.com search.

        :param query: The search query string.
        :return: A dictionary containing the search results.
        """
        url = f"https://api.you.com/search?q={query}&key={self.YOU_API_KEY}"
        response = requests.get(url)
        return response.json()

def main():
    tool = WebSearchTool()

    # Smoke tests
    print("Google Search Test:", tool.google_search("OpenAI"))
    print("Google News Search Test:", tool.google_news_search("OpenAI"))
    print("Wikipedia Search Test:", tool.wikipedia_search("OpenAI"))
    print("Perplexity Search Test:", tool.perplexity_search("OpenAI"))
    print("You.com Search Test:", tool.you_search("OpenAI"))

if __name__ == "__main__":
    main()
