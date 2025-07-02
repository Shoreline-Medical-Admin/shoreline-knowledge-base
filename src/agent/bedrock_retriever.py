"""AWS Bedrock Knowledge Base retriever component."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class BedrockKnowledgeBaseRetriever:
    """Retriever for AWS Bedrock Knowledge Base."""

    def __init__(
        self,
        knowledge_base_id: str,
        region_name: str = "us-east-1",
        model_arn: Optional[str] = None,
        max_results: int = 5,
    ):
        """Initialize the Bedrock Knowledge Base retriever.

        Args:
            knowledge_base_id: The ID of the Bedrock Knowledge Base
            region_name: AWS region name
            model_arn: Optional model ARN for retrieval
            max_results: Maximum number of results to retrieve
        """
        self.knowledge_base_id = knowledge_base_id
        self.region_name = region_name
        self.model_arn = model_arn
        self.max_results = max_results
        
        # Initialize Bedrock client lazily to avoid blocking at import time
        self._client = None
        
    def _get_client(self):
        """Get or create the Bedrock client."""
        if self._client is None:
            self._client = boto3.client(
                "bedrock-agent-runtime",
                region_name=self.region_name
            )
        return self._client
    
    @property
    def bedrock_agent_runtime(self):
        """Property to access the client."""
        return self._get_client()

    def retrieve(
        self, 
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve relevant documents from the knowledge base.

        Args:
            query: The query to search for
            filter: Optional metadata filter
            next_token: Token for pagination

        Returns:
            Dictionary containing retrieved documents and metadata
        """
        try:
            # Prepare the retrieval request
            retrieval_config = {
                "vectorSearchConfiguration": {
                    "numberOfResults": self.max_results,
                }
            }
            
            if filter:
                retrieval_config["vectorSearchConfiguration"]["filter"] = filter
            
            # Build the request parameters
            request_params = {
                "knowledgeBaseId": self.knowledge_base_id,
                "retrievalQuery": {"text": query},
                "retrievalConfiguration": retrieval_config,
            }
            
            if self.model_arn:
                request_params["modelArn"] = self.model_arn
                
            if next_token:
                request_params["nextToken"] = next_token
            
            # Perform the retrieval
            response = self.bedrock_agent_runtime.retrieve(**request_params)
            
            # Process the results
            retrieved_documents = []
            for result in response.get("retrievalResults", []):
                doc = {
                    "content": result.get("content", {}).get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0),
                    "location": result.get("location", {}),
                }
                retrieved_documents.append(doc)
            
            return {
                "documents": retrieved_documents,
                "next_token": response.get("nextToken"),
                "query": query,
            }
            
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]
            logger.error(f"AWS Bedrock error: {error_code} - {error_message}")
            
            # Return empty results on error
            return {
                "documents": [],
                "error": f"{error_code}: {error_message}",
                "query": query,
            }
        except Exception as e:
            logger.error(f"Unexpected error during retrieval: {str(e)}")
            return {
                "documents": [],
                "error": str(e),
                "query": query,
            }

    async def aretrieve(
        self, 
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of retrieve using asyncio.to_thread."""
        return await asyncio.to_thread(
            self.retrieve, 
            query,
            filter,
            next_token
        )

    def format_documents_for_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string for the LLM.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "").strip()
            score = doc.get("score", 0.0)
            
            if content:
                context_parts.append(f"[Document {i}] (Relevance: {score:.2f})\n{content}")
        
        return "\n\n---\n\n".join(context_parts)