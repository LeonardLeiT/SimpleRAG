import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

def get_rag_key(env_key='DASHSCOPE_API_KEY'):
    """Get API key from environment variables.
    
    Args:
        env_key: The environment variable name for the API key
        
    Returns:
        str: The API key
        
    Raises:
        ValueError: If the API key is not found in environment variables
    """
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv(env_key)
    if api_key is None:
        raise ValueError(f"{env_key} not found in environment variables.")
    return api_key

def get_embedding(input, dimension=1024, api_key=None, base_url=None):
    """Get embeddings using specified API key and base URL.
    
    Args:
        input: The input text to embed
        dimension: The dimension of the embeddings
        api_key: The API key to use (defaults to DASHSCOPE_API_KEY)
        base_url: The base URL for the API (defaults to DashScope URL)
    """
    if api_key is None:
        api_key = get_rag_key()
    if base_url is None:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
    client = OpenAI(
        api_key=api_key,  
        base_url=base_url
    )
    completion = client.embeddings.create(
        model="text-embedding-v3",
        input=input,
        dimensions=dimension,
        encoding_format="float"
    )
    return completion

from typing import List
from langchain_core.embeddings import Embeddings

class MyEmbeddings(Embeddings):
    def __init__(
        self, 
        dimension: int = 1024,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str = None,
        env_key: str = 'DASHSCOPE_API_KEY'
    ):
        """Initialize the embeddings class.
        
        Args:
            dimension: The dimension of the embeddings
            base_url: The base URL for the API
            api_key: The API key to use (if None, will get from environment)
            env_key: The environment variable name for the API key
        """
        self.dimension = dimension
        self.base_url = base_url
        if api_key is None:
            api_key = get_rag_key(env_key)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        results = []
        for text in texts:
            completion = self.client.embeddings.create(
                model="text-embedding-v3",
                input=text,
                dimensions=self.dimension,
                encoding_format="float"
            )
            results.append(completion.data[0].embedding)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        completion = self.client.embeddings.create(
            model="text-embedding-v3",
            input=text,
            dimensions=self.dimension,
            encoding_format="float"
        )
        return completion.data[0].embedding

