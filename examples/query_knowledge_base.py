"""Example script to query the AWS Bedrock Knowledge Base agent."""

import asyncio
from typing import Dict, Any

from src.agent.graph import graph
from src.agent.config import get_default_config


async def query_knowledge_base(query: str, config: Dict[str, Any] = None):
    """Query the knowledge base with a question.

    Args:
        query: The question to ask
        config: Optional configuration overrides

    Returns:
        The agent's response
    """
    # Use default config if none provided
    if config is None:
        config = get_default_config()
    
    # Prepare the input state
    input_state = {
        "query": query,
    }
    
    # Prepare the configuration
    runnable_config = {
        "configurable": config
    }
    
    # Invoke the graph
    result = await graph.ainvoke(input_state, runnable_config)
    
    return result


async def main():
    """Example usage of the Knowledge Base agent."""
    # Example queries
    queries = [
        "What is AWS Bedrock?",
        "How do I create a knowledge base?",
        "What models are available in Bedrock?",
    ]
    
    # Get default configuration (from environment)
    config = get_default_config()
    
    # You can override specific settings if needed
    # config["temperature"] = 0.5
    # config["max_results"] = 10
    
    print("AWS Bedrock Knowledge Base Q&A Agent")
    print("=" * 50)
    
    for query in queries:
        print(f"\nQ: {query}")
        try:
            result = await query_knowledge_base(query, config)
            print(f"A: {result.get('answer', 'No answer generated')}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())