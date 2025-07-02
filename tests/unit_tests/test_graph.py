"""Unit tests for the graph components."""

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
            
            mock_retriever.retrieve.return_value = {
                "documents": [
                    {
                        "content": "Test content",
                        "score": 0.95,
                        "metadata": {"source": "test.pdf"},
                        "location": {"s3": "s3://bucket/test.pdf"}
                    }
                ],
                "error": None
            }
            mock_retriever.format_documents_for_context.return_value = "Formatted context"
            
            result = await retrieve_documents(state, config)
            
            assert len(result["retrieved_documents"]) == 1
            assert result["context"] == "Formatted context"
            assert len(result["sources"]) == 1
            assert result["sources"][0]["score"] == 0.95
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_retrieve_documents_no_kb_id(self):
        """Test retrieval without knowledge base ID."""
        state = State(query="test query")
        config = {"configurable": {}}
        
        result = await retrieve_documents(state, config)
        
        assert result["error"] == "Knowledge Base ID not configured"
        assert result["retrieved_documents"] == []
        assert result["context"] == ""

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
            mock_llm = AsyncMock()
            mock_llm_class.return_value = mock_llm
            
            # Mock the chain response
            mock_response = Mock()
            mock_response.content = "AWS is Amazon Web Services."
            
            # Create a mock chain that returns the response
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = mock_response
            
            with patch("agent.graph.ChatPromptTemplate.from_messages") as mock_prompt:
                mock_prompt.return_value.__or__ = Mock(return_value=mock_chain)
                
                result = await generate_answer(state, config)
                
                assert result["answer"] == "AWS is Amazon Web Services."

    @pytest.mark.asyncio
    async def test_generate_answer_no_context(self):
        """Test answer generation without context."""
        state = State(query="What is AWS?", context="")
        config = {"configurable": {}}
        
        result = await generate_answer(state, config)
        
        assert "couldn't find relevant information" in result["answer"]

    @pytest.mark.asyncio
    async def test_format_response_with_sources(self):
        """Test response formatting with sources."""
        state = State(
            answer="This is the answer.",
            sources=[
                {"score": 0.95, "metadata": {"source": "doc1"}},
                {"score": 0.87, "metadata": {"source": "doc2"}}
            ]
        )
        config = {"configurable": {}}
        
        result = await format_response(state, config)
        
        assert "This is the answer." in result["answer"]
        assert "Sources:" in result["answer"]
        assert "Source 1 (Relevance: 0.95)" in result["answer"]
        assert "Source 2 (Relevance: 0.87)" in result["answer"]

    @pytest.mark.asyncio
    async def test_format_response_with_error(self):
        """Test response formatting with error."""
        state = State(
            answer="Some answer",
            error="Test error message"
        )
        config = {"configurable": {}}
        
        result = await format_response(state, config)
        
        assert result["answer"] == "Error: Test error message"


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