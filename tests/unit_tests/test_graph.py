"""Unit tests for the graph components."""

import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from agent.graph import (
    State,
    Configuration,
    retrieve_documents,
    generate_answer,
    format_response,
    graph
)


class TestState:
    """Test cases for the State dataclass."""

    def test_state_initialization(self):
        """Test State initialization with default values."""
        state = State()
        assert state.query == ""
        assert state.retrieved_documents == []
        assert state.context == ""
        assert state.answer == ""
        assert state.sources == []
        assert state.error is None
        assert state.reasoning_engine is None
        assert state.confidence_score == 0.0
        assert state.show_reasoning is True

    def test_state_with_values(self):
        """Test State initialization with custom values."""
        docs = [{"content": "test doc"}]
        sources = [{"score": 0.9}]
        state = State(
            query="test query",
            retrieved_documents=docs,
            context="test context",
            answer="test answer",
            sources=sources,
            error="test error"
        )
        assert state.query == "test query"
        assert state.retrieved_documents == docs
        assert state.context == "test context"
        assert state.answer == "test answer"
        assert state.sources == sources
        assert state.error == "test error"


class TestGraphNodes:
    """Test cases for graph node functions."""

    @pytest.mark.asyncio
    async def test_retrieve_documents_success(self):
        """Test successful document retrieval."""
        state = State(query="test query")
        config = {
            "configurable": {
                "knowledge_base_id": "test-kb-id",
                "aws_region": "us-east-1",
                "max_results": 5
            }
        }
        
        with patch("agent.graph.BedrockKnowledgeBaseRetriever") as mock_retriever_class:
            mock_retriever = Mock()
            mock_retriever_class.return_value = mock_retriever
            
            # Create async mock for aretrieve
            async def mock_aretrieve(query):
                return {
                    "documents": [
                        {
                            "content": "Test content",
                            "score": 0.95,
                            "metadata": {"source": "test.pdf"},
                            "location": {"s3": "s3://bucket/test.pdf"},
                            "kb_type": "general"
                        }
                    ],
                    "error": None
                }
            
            mock_retriever.aretrieve = mock_aretrieve
            mock_retriever.format_documents_for_context.return_value = "Formatted context"
            
            result = await retrieve_documents(state, config)
            
            assert len(result["retrieved_documents"]) == 1
            assert "Formatted context" in result["context"]
            assert len(result["sources"]) == 1
            assert result["sources"][0]["score"] == 0.95
            assert result["error"] is None
            assert result["reasoning_engine"] is not None
            assert result["confidence_score"] > 0

    @pytest.mark.asyncio
    async def test_retrieve_documents_no_kb_id(self):
        """Test retrieval without knowledge base ID."""
        state = State(query="test query")
        config = {"configurable": {}}
        
        # Mock environment variables to be empty
        with patch.dict(os.environ, {
            "MEDICAL_GUIDELINES_KB_ID": "",
            "CMS_CODING_KB_ID": "",
            "BEDROCK_KNOWLEDGE_BASE_ID": ""
        }, clear=False):
            result = await retrieve_documents(state, config)
            
            assert result["error"] == "No Knowledge Base IDs configured"
            assert result["retrieved_documents"] == []
            assert result["context"] == ""
            assert result["reasoning_engine"] is not None
            assert result["confidence_score"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_answer_with_context(self):
        """Test answer generation with context."""
        state = State(
            query="What is AWS?",
            context="AWS is Amazon Web Services, a cloud computing platform."
        )
        config = {
            "configurable": {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "aws_region": "us-east-1",
                "temperature": 0.3
            }
        }
        
        with patch("agent.graph.ChatBedrock") as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm
            
            # Mock the chain response
            mock_response = Mock()
            mock_response.content = "AWS is Amazon Web Services."
            
            # Create a mock chain that returns the response
            mock_chain = Mock()
            mock_chain.invoke = Mock(return_value=mock_response)
            
            with patch("agent.graph.ChatPromptTemplate.from_messages") as mock_prompt:
                mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
                
                # Add asyncio.to_thread patch since we use it in the actual function
                with patch("asyncio.to_thread") as mock_to_thread:
                    # Make asyncio.to_thread just call the function directly
                    mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
                    
                    result = await generate_answer(state, config)
                    
                    assert result["answer"] == "AWS is Amazon Web Services."

    @pytest.mark.asyncio
    async def test_generate_answer_no_context(self):
        """Test answer generation without context."""
        state = State(query="What is AWS?", context="")
        config = {"configurable": {}}
        
        result = await generate_answer(state, config)
        
        assert "No relevant information found" in result["answer"]

    @pytest.mark.asyncio
    async def test_format_response_with_sources(self):
        """Test response formatting with sources."""
        state = State(
            answer="This is the answer.",
            sources=[
                {"score": 0.95, "metadata": {"source": "doc1"}, "kb_type": "medical_guidelines"},
                {"score": 0.87, "metadata": {"source": "doc2"}, "kb_type": "cms_coding"}
            ],
            retrieved_documents=[
                {"kb_type": "medical_guidelines"},
                {"kb_type": "cms_coding"}
            ]
        )
        config = {"configurable": {}}
        
        result = await format_response(state, config)
        
        assert "This is the answer." in result["answer"]
        assert "Sources:" in result["answer"]
        assert "95.00%" in result["answer"]  # Score formatting changed
        assert "87.00%" in result["answer"]
        assert "🟢" in result["answer"]  # High relevance indicator
        assert "🟡" in result["answer"]  # Medium relevance indicator

    @pytest.mark.asyncio
    async def test_format_response_with_error(self):
        """Test response formatting with error."""
        state = State(
            answer="Some answer",
            error="Test error message"
        )
        config = {"configurable": {}}
        
        result = await format_response(state, config)
        
        assert result["answer"] == "❌ Error: Test error message"


class TestGraph:
    """Test cases for the compiled graph."""

    def test_graph_structure(self):
        """Test that the graph is properly structured."""
        # Check nodes
        assert "retrieve_documents" in graph.nodes
        assert "generate_answer" in graph.nodes
        assert "format_response" in graph.nodes
        
        # Check graph name
        assert graph.name == "Bedrock Knowledge Base Q&A"
        
        # Check configuration schema
        assert graph.config_schema() is not None