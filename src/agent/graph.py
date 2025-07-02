"""AWS Bedrock Knowledge Base Q&A Agent.

This agent retrieves information from AWS Bedrock Knowledge Base and generates answers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from agent.bedrock_retriever import BedrockKnowledgeBaseRetriever

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Validate AWS credentials at module load time
if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
    logger.error(
        "AWS credentials not found. Please configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
        "in your .env file or environment variables."
    )


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    """

    knowledge_base_id: str  # Legacy single KB support
    medical_guidelines_kb_id: str
    cms_coding_kb_id: str
    aws_region: str
    model_id: str
    max_results: int
    temperature: float
    knowledge_bases: str  # Which KBs to query: "medical", "cms", "both"


@dataclass
class State:
    """State for the Knowledge Base Q&A agent.

    Tracks the query, retrieved documents, and generated answer.
    """

    query: str = ""
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


async def retrieve_documents(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve relevant documents from AWS Bedrock Knowledge Base."""
    configuration = config.get("configurable", {})
    
    # Get configuration with environment fallbacks
    aws_region = configuration.get("aws_region") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    max_results = configuration.get("max_results") or int(os.getenv("BEDROCK_MAX_RESULTS", "5"))
    knowledge_bases = configuration.get("knowledge_bases") or os.getenv("KNOWLEDGE_BASES", "both")
    
    # Get KB IDs with environment fallbacks
    medical_kb_id = configuration.get("medical_guidelines_kb_id") or os.getenv("MEDICAL_GUIDELINES_KB_ID", "VXMUOUXXCF")
    cms_kb_id = configuration.get("cms_coding_kb_id") or os.getenv("CMS_CODING_KB_ID", "X1DCXMHW9T")
    legacy_kb_id = configuration.get("knowledge_base_id") or os.getenv("BEDROCK_KNOWLEDGE_BASE_ID")
    
    # Debug logging
    logger.info(f"Query: '{state.query}'")
    logger.info(f"Configuration received: {configuration}")
    logger.info(f"Environment - Medical KB: {os.getenv('MEDICAL_GUIDELINES_KB_ID')}")
    logger.info(f"Environment - CMS KB: {os.getenv('CMS_CODING_KB_ID')}")
    logger.info(f"Using - Medical KB: {medical_kb_id}, CMS KB: {cms_kb_id}, Knowledge bases: {knowledge_bases}")
    logger.info(f"AWS Region: {aws_region}, Max Results: {max_results}")
    
    # Determine which knowledge bases to query
    kb_ids = []
    
    # Check for legacy single KB configuration
    if legacy_kb_id:
        kb_ids.append((legacy_kb_id, "general"))
    else:
        # Use multiple KB configuration
        if knowledge_bases in ["medical", "both"] and medical_kb_id:
            kb_ids.append((medical_kb_id, "medical_guidelines"))
        
        if knowledge_bases in ["cms", "both"] and cms_kb_id:
            kb_ids.append((cms_kb_id, "cms_coding"))
    
    if not kb_ids:
        logger.error(f"No Knowledge Base IDs found. Legacy KB: {legacy_kb_id}, Medical: {medical_kb_id}, CMS: {cms_kb_id}")
        return {
            "error": "No Knowledge Base IDs configured",
            "retrieved_documents": [],
            "context": "",
        }
    
    logger.info(f"Will query {len(kb_ids)} knowledge base(s): {kb_ids}")
    
    try:
        all_documents = []
        all_sources = []
        context_parts = []
        
        # Query each knowledge base
        for kb_id, kb_type in kb_ids:
            logger.info(f"Querying {kb_type} knowledge base with ID: {kb_id}")
            
            # Initialize retriever
            retriever = BedrockKnowledgeBaseRetriever(
                knowledge_base_id=kb_id,
                region_name=aws_region,
                max_results=max_results,
            )
            
            # Retrieve documents (using async version)
            retrieval_result = await retriever.aretrieve(state.query)
            
            if retrieval_result.get("error"):
                logger.warning(f"Error retrieving from {kb_type} KB: {retrieval_result['error']}")
                continue
            
            documents = retrieval_result.get("documents", [])
            logger.info(f"Retrieved {len(documents)} documents from {kb_type} KB")
            
            # Add KB source type to documents
            for i, doc in enumerate(documents):
                doc["kb_type"] = kb_type
                all_documents.append(doc)
                if i < 2:  # Log first 2 documents for debugging
                    logger.info(f"Document {i+1} score: {doc.get('score', 0.0):.3f}")
                    logger.info(f"Document {i+1} content preview: {doc.get('content', '')[:200]}...")
            
            # Format documents for context with KB type prefix
            if documents:
                kb_context = f"[From {kb_type.upper()} Knowledge Base]\n"
                kb_context += retriever.format_documents_for_context(documents)
                context_parts.append(kb_context)
            
            # Extract sources with KB type
            for doc in documents:
                source_info = {
                    "score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {}),
                    "location": doc.get("location", {}),
                    "kb_type": kb_type,
                }
                all_sources.append(source_info)
        
        # Combine contexts
        combined_context = "\n\n===\n\n".join(context_parts) if context_parts else "No relevant documents found."
        
        # Sort all documents by score
        all_documents.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        all_sources.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        return {
            "retrieved_documents": all_documents,
            "context": combined_context,
            "sources": all_sources,
            "error": None,
        }
        
    except Exception as e:
        logger.error(f"Error during document retrieval: {str(e)}")
        return {
            "error": f"Retrieval error: {str(e)}",
            "retrieved_documents": [],
            "context": "",
        }


async def generate_answer(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate answer using retrieved context and Bedrock LLM."""
    configuration = config.get("configurable", {})
    
    # Check if we have context
    if not state.context or state.context == "No relevant documents found.":
        return {
            "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about something else.",
        }
    
    # Get LLM configuration with environment fallbacks
    model_id = configuration.get("model_id") or os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
    aws_region = configuration.get("aws_region") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    temperature = configuration.get("temperature") or float(os.getenv("BEDROCK_MODEL_TEMPERATURE", "0.3"))
    
    try:
        # Initialize Bedrock LLM - wrap in asyncio.to_thread to avoid blocking
        def create_llm():
            return ChatBedrock(
                model_id=model_id,
                region_name=aws_region,
                model_kwargs={"temperature": temperature},
            )
        
        llm = await asyncio.to_thread(create_llm)
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer questions. 
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise but comprehensive in your answers."""),
            ("human", """Context:
{context}

Question: {query}

Please provide a clear and accurate answer based on the context above."""),
        ])
        
        # Generate answer - use asyncio.to_thread for the invoke
        chain = prompt | llm
        
        # Wrap the synchronous invoke in asyncio.to_thread
        response = await asyncio.to_thread(
            chain.invoke,
            {
                "context": state.context,
                "query": state.query,
            }
        )
        
        return {
            "answer": response.content,
        }
        
    except Exception as e:
        logger.error(f"Error during answer generation: {str(e)}")
        return {
            "answer": f"I encountered an error while generating the answer: {str(e)}",
            "error": f"Generation error: {str(e)}",
        }


async def format_response(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Format the final response with answer and sources."""
    # If there's an error, return it
    if state.error:
        return {
            "answer": f"Error: {state.error}",
        }
    
    # Format response with sources if available
    if state.sources:
        sources_text = "\n\nSources:"
        
        # Group sources by KB type
        sources_by_kb = {}
        for source in state.sources[:6]:  # Limit to top 6 sources total
            kb_type = source.get("kb_type", "unknown")
            if kb_type not in sources_by_kb:
                sources_by_kb[kb_type] = []
            sources_by_kb[kb_type].append(source)
        
        # Format sources grouped by KB
        for kb_type, kb_sources in sources_by_kb.items():
            if kb_sources:
                kb_name = kb_type.replace("_", " ").title()
                sources_text += f"\n\nFrom {kb_name}:"
                for i, source in enumerate(kb_sources[:3], 1):  # Max 3 per KB
                    score = source.get("score", 0.0)
                    metadata = source.get("metadata", {})
                    if metadata:
                        sources_text += f"\n- Source {i} (Relevance: {score:.2f})"
        
        return {
            "answer": state.answer + sources_text,
        }
    
    return {"answer": state.answer}


# Define the graph
graph = (
    StateGraph(State, config_schema=Configuration)
    .add_node("retrieve_documents", retrieve_documents)
    .add_node("generate_answer", generate_answer)
    .add_node("format_response", format_response)
    .add_edge("__start__", "retrieve_documents")
    .add_edge("retrieve_documents", "generate_answer")
    .add_edge("generate_answer", "format_response")
    .compile(name="Bedrock Knowledge Base Q&A")
)
