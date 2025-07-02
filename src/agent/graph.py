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
from agent.reasoning import ReasoningEngine, UncertaintyType

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

    Tracks the query, retrieved documents, generated answer, and reasoning.
    """

    query: str = ""
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    reasoning_engine: Optional[ReasoningEngine] = None
    confidence_score: float = 0.0
    show_reasoning: bool = True


async def retrieve_documents(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve relevant documents from AWS Bedrock Knowledge Base."""
    configuration = config.get("configurable", {})
    
    # Initialize reasoning engine
    reasoning_engine = ReasoningEngine()
    
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
        reasoning_engine.uncertainty_flags.append(UncertaintyType.NO_SOURCES)
        return {
            "error": "No Knowledge Base IDs configured",
            "retrieved_documents": [],
            "context": "",
            "reasoning_engine": reasoning_engine,
            "confidence_score": 0.0,
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
                kb_name = kb_type.replace("_", " ").title()
                kb_icon = "🏥" if "medical" in kb_type.lower() else "📋" if "cms" in kb_type.lower() else "📄"
                kb_context = f"{kb_icon} **{kb_name} Knowledge Base**\n"
                kb_context += "═" * 40 + "\n"
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
        combined_context = "\n\n\n🔗 " + "━" * 50 + " 🔗\n\n\n".join(context_parts) if context_parts else "No relevant documents found."
        
        # Sort all documents by score
        all_documents.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        all_sources.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Evaluate sources and calculate confidence
        source_confidence, evaluations = reasoning_engine.evaluate_sources(all_documents, state.query)
        confidence_score = reasoning_engine.calculate_confidence_score(source_confidence)
        
        return {
            "retrieved_documents": all_documents,
            "context": combined_context,
            "sources": all_sources,
            "error": None,
            "reasoning_engine": reasoning_engine,
            "confidence_score": confidence_score,
        }
        
    except Exception as e:
        logger.error(f"Error during document retrieval: {str(e)}")
        return {
            "error": f"Retrieval error: {str(e)}",
            "retrieved_documents": [],
            "context": "",
            "reasoning_engine": reasoning_engine,
            "confidence_score": 0.0,
        }


async def generate_answer(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate answer using retrieved context and Bedrock LLM."""
    configuration = config.get("configurable", {})
    
    # Check if we have context
    if not state.context or state.context == "No relevant documents found.":
        no_context_answer = "🔍 **No relevant information found**\n\nI couldn't find any documents in the knowledge base that match your query.\n\n**Suggestions:**\n• Try rephrasing your question\n• Use different keywords\n• Check if the topic is covered in the available knowledge bases"
        
        # Add reasoning if available
        if state.reasoning_engine and state.show_reasoning:
            no_context_answer += state.reasoning_engine.format_reasoning(state.confidence_score, show_details=True)
        
        return {
            "answer": no_context_answer,
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
        
        # Create confidence-aware prompt
        confidence_level = "high" if state.confidence_score >= 0.8 else "medium" if state.confidence_score >= 0.6 else "low"
        
        # Add uncertainty considerations to system prompt if needed
        uncertainty_notes = ""
        if state.reasoning_engine and state.reasoning_engine.uncertainty_flags:
            if UncertaintyType.OUTDATED_INFORMATION in state.reasoning_engine.uncertainty_flags:
                uncertainty_notes += "\n- Some sources may be outdated - mention if information might have changed"
            if UncertaintyType.CONFLICTING_SOURCES in state.reasoning_engine.uncertainty_flags:
                uncertainty_notes += "\n- Sources contain conflicting information - present different viewpoints if relevant"
            if UncertaintyType.LIMITED_SOURCES in state.reasoning_engine.uncertainty_flags:
                uncertainty_notes += "\n- Limited sources available - acknowledge gaps in information"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful medical and coding assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer questions. 
If the context doesn't contain enough information to answer the question, say so clearly.

Current confidence level: {confidence_level} ({state.confidence_score:.0%})
{uncertainty_notes}

When answering:
- Be concise but comprehensive
- Use bullet points for lists
- Highlight key information with **bold** text
- For codes or specific values, format them clearly (e.g., Code: **12**)
- If multiple relevant pieces of information exist, organize them clearly
- If confidence is low or medium, acknowledge limitations in the available information"""),
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
            "answer": f"❌ Error: {state.error}",
        }
    
    # Start with the answer
    formatted_response = f"💡 **Answer:**\n{state.answer}"
    
    # Add sources if available
    if state.sources:
        formatted_response += "\n\n" + "─" * 50 + "\n"
        formatted_response += "📚 **Sources:**"
        
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
                # Use different icons for different KB types
                kb_icon = "🏥" if "medical" in kb_type.lower() else "📋" if "cms" in kb_type.lower() else "📄"
                
                formatted_response += f"\n\n{kb_icon} **{kb_name}:**"
                for i, source in enumerate(kb_sources[:3], 1):  # Max 3 per KB
                    score = source.get("score", 0.0)
                    metadata = source.get("metadata", {})
                    
                    # Format relevance score with visual indicator
                    if score >= 0.9:
                        relevance_indicator = "🟢"
                    elif score >= 0.7:
                        relevance_indicator = "🟡"
                    else:
                        relevance_indicator = "🔴"
                    
                    formatted_response += f"\n  {relevance_indicator} Source {i} (Relevance: {score:.2%})"
                    
                    # Add metadata if available
                    if metadata:
                        for key, value in metadata.items():
                            if value and key not in ['chunkId', 'x-amz-bedrock-kb-chunk-id']:
                                formatted_response += f"\n     • {key}: {value}"
        
        formatted_response += "\n\n" + "─" * 50
    
    # Add a footer with retrieval statistics
    formatted_response += f"\n\n📊 Retrieved {len(state.retrieved_documents)} documents from {len(set(doc.get('kb_type', 'unknown') for doc in state.retrieved_documents))} knowledge base(s)"
    
    # Add reasoning explanation if available
    if state.reasoning_engine and state.show_reasoning:
        formatted_response += state.reasoning_engine.format_reasoning(state.confidence_score, show_details=True)
    
    return {"answer": formatted_response}


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
