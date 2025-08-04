import os
from typing import List
from V8_api import UltraFastRAGParser, UltraFastRAGRetriever
import asyncio
import net_utils  # ADD THIS IMPORT

async def answer_questions(document_urls: List[str], questions: List[str]) -> List[str]:
    """
    Main pipeline function that processes documents and answers questions.
    Returns a list of answers corresponding to the input questions.
    """
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize components
    parser = UltraFastRAGParser(chunk_size=1000, overlap=200, api_key=api_key)
    retriever = UltraFastRAGRetriever(parser)
    
    # Index documents asynchronously (with caching)
    cache_path = f"cache_multi_docs_v6_llm_enhanced"
    chunks = await retriever.index_documents_async(document_urls)
    
    # Process queries in parallel - PASS THE SESSION
    tasks = [retriever.retrieve_and_answer_async(query, k=5, session=net_utils.shared_session) for query in questions]
    all_results = await asyncio.gather(*tasks)
    
    # Extract just the answers
    answers = []
    for results, answer in all_results:
        answers.append(answer)
    
    return answers