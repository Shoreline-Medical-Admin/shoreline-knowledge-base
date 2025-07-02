"""Example demonstrating reasoning transparency feature."""

import asyncio
import os
from agent.graph import graph, State

async def main():
    """Run example query with reasoning transparency."""
    
    # Example query
    query = "What are the guidelines for Medicare billing?"
    
    print(f"Query: {query}\n")
    print("=" * 80)
    
    # Configure the agent
    config = {
        "configurable": {
            "knowledge_bases": "both",  # Query both medical and CMS knowledge bases
            "max_results": 5,
            "temperature": 0.3,
        }
    }
    
    # Initialize state with reasoning enabled (default)
    initial_state = State(
        query=query,
        show_reasoning=True  # This is True by default
    )
    
    # Run the query
    result = await graph.ainvoke(initial_state, config)
    
    # Display the result
    print(result["answer"])
    
    print("\n" + "=" * 80)
    print("\nReasoningTransparency Features Demonstrated:")
    print("- Confidence score with visual indicator (🟢🟡🟠🔴)")
    print("- Uncertainty warnings when applicable")
    print("- Detailed reasoning process in collapsible section")
    print("- Source evaluation with relevance, recency, and authority scores")
    print("- Detection of outdated information, limited sources, or conflicts")
    
    # Example with reasoning disabled
    print("\n" + "=" * 80)
    print("\nSame query with reasoning disabled:")
    print("=" * 80 + "\n")
    
    initial_state_no_reasoning = State(
        query=query,
        show_reasoning=False
    )
    
    result_no_reasoning = await graph.ainvoke(initial_state_no_reasoning, config)
    print(result_no_reasoning["answer"])

if __name__ == "__main__":
    # Ensure environment variables are loaded
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please configure your .env file with AWS credentials.")
        exit(1)
    
    # Run the example
    asyncio.run(main())