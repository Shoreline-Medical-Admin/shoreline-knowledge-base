"""Unit tests for the Bedrock Knowledge Base retriever."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from agent.bedrock_retriever import BedrockKnowledgeBaseRetriever


class TestBedrockKnowledgeBaseRetriever:
    """Test cases for BedrockKnowledgeBaseRetriever."""

    @pytest.fixture
    def retriever(self):
        """Create a retriever instance for testing."""
        with patch("boto3.client") as mock_client:
            retriever = BedrockKnowledgeBaseRetriever(
                knowledge_base_id="test-kb-id",
                region_name="us-east-1",
                max_results=5
            )
            return retriever

    def test_initialization(self):
        """Test retriever initialization."""
        with patch("boto3.client") as mock_client:
            retriever = BedrockKnowledgeBaseRetriever(
                knowledge_base_id="test-kb-id",
                region_name="us-west-2",
                model_arn="arn:aws:bedrock:us-west-2::foundation-model/test",
                max_results=10
            )
            
            assert retriever.knowledge_base_id == "test-kb-id"
            assert retriever.region_name == "us-west-2"
            assert retriever.model_arn == "arn:aws:bedrock:us-west-2::foundation-model/test"
            assert retriever.max_results == 10
            mock_client.assert_called_once_with("bedrock-agent-runtime", region_name="us-west-2")

    def test_retrieve_success(self, retriever):
        """Test successful document retrieval."""
        # Mock response from AWS
        mock_response = {
            "retrievalResults": [
                {
                    "content": {"text": "Document 1 content"},
                    "metadata": {"source": "doc1.pdf"},
                    "score": 0.95,
                    "location": {"s3Location": {"uri": "s3://bucket/doc1.pdf"}}
                },
                {
                    "content": {"text": "Document 2 content"},
                    "metadata": {"source": "doc2.pdf"},
                    "score": 0.87,
                    "location": {"s3Location": {"uri": "s3://bucket/doc2.pdf"}}
                }
            ],
            "nextToken": "next-page-token"
        }
        
        retriever.bedrock_agent_runtime.retrieve = Mock(return_value=mock_response)
        
        # Test retrieval
        result = retriever.retrieve("test query")
        
        # Verify results
        assert len(result["documents"]) == 2
        assert result["documents"][0]["content"] == "Document 1 content"
        assert result["documents"][0]["score"] == 0.95
        assert result["documents"][1]["content"] == "Document 2 content"
        assert result["next_token"] == "next-page-token"
        assert result["query"] == "test query"
        
        # Verify API call
        retriever.bedrock_agent_runtime.retrieve.assert_called_once()

    def test_retrieve_with_filter(self, retriever):
        """Test retrieval with metadata filter."""
        retriever.bedrock_agent_runtime.retrieve = Mock(return_value={"retrievalResults": []})
        
        filter_dict = {"andAll": [{"equals": {"key": "category", "value": "technical"}}]}
        retriever.retrieve("test query", filter=filter_dict)
        
        # Verify filter was included in the request
        call_args = retriever.bedrock_agent_runtime.retrieve.call_args[1]
        assert call_args["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] == filter_dict

    def test_retrieve_client_error(self, retriever):
        """Test handling of AWS client errors."""
        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "Knowledge base not found"
            }
        }
        retriever.bedrock_agent_runtime.retrieve = Mock(
            side_effect=ClientError(error_response, "Retrieve")
        )
        
        result = retriever.retrieve("test query")
        
        assert result["documents"] == []
        assert "ResourceNotFoundException" in result["error"]
        assert "Knowledge base not found" in result["error"]

    def test_retrieve_unexpected_error(self, retriever):
        """Test handling of unexpected errors."""
        retriever.bedrock_agent_runtime.retrieve = Mock(
            side_effect=Exception("Unexpected error")
        )
        
        result = retriever.retrieve("test query")
        
        assert result["documents"] == []
        assert "Unexpected error" in result["error"]

    def test_format_documents_for_context(self, retriever):
        """Test document formatting for context."""
        documents = [
            {"content": "First document", "score": 0.9},
            {"content": "Second document", "score": 0.8},
            {"content": "Third document", "score": 0.7}
        ]
        
        context = retriever.format_documents_for_context(documents)
        
        assert "[Document 1] (Relevance: 0.90)" in context
        assert "First document" in context
        assert "[Document 2] (Relevance: 0.80)" in context
        assert "Second document" in context
        assert "[Document 3] (Relevance: 0.70)" in context
        assert "Third document" in context

    def test_format_documents_empty_list(self, retriever):
        """Test formatting with no documents."""
        context = retriever.format_documents_for_context([])
        assert context == "No relevant documents found."