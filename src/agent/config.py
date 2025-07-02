"""Configuration loader for AWS Bedrock Knowledge Base agent."""

import os
from typing import Dict, Any

from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables.

    Returns:
        Dictionary containing configuration values
    """
    # Load .env file if it exists
    load_dotenv()
    
    # AWS Configuration
    aws_config = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }
    
    # Validate AWS credentials are provided
    if not aws_config["aws_access_key_id"] or not aws_config["aws_secret_access_key"]:
        raise ValueError(
            "AWS credentials are required but not configured.\n"
            "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file or environment.\n"
            "Example:\n"
            "  AWS_ACCESS_KEY_ID=your_access_key_here\n"
            "  AWS_SECRET_ACCESS_KEY=your_secret_key_here"
        )
    
    # Bedrock Configuration
    bedrock_config = {
        "knowledge_base_id": os.getenv("BEDROCK_KNOWLEDGE_BASE_ID"),  # Legacy support
        "medical_guidelines_kb_id": os.getenv("MEDICAL_GUIDELINES_KB_ID", "VXMUOUXXCF"),
        "cms_coding_kb_id": os.getenv("CMS_CODING_KB_ID", "X1DCXMHW9T"),
        "model_id": os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        "temperature": float(os.getenv("BEDROCK_MODEL_TEMPERATURE", "0.3")),
        "max_results": int(os.getenv("BEDROCK_MAX_RESULTS", "5")),
        "knowledge_bases": os.getenv("KNOWLEDGE_BASES", "both"),  # "medical", "cms", or "both"
    }
    
    # Validate required configuration - at least one KB must be configured
    if not any([
        bedrock_config["knowledge_base_id"],
        bedrock_config["medical_guidelines_kb_id"],
        bedrock_config["cms_coding_kb_id"]
    ]):
        raise ValueError(
            "At least one Knowledge Base ID must be configured. "
            "Please set BEDROCK_KNOWLEDGE_BASE_ID, MEDICAL_GUIDELINES_KB_ID, "
            "or CMS_CODING_KB_ID in your .env file or environment."
        )
    
    # Set AWS credentials if provided
    if aws_config["aws_access_key_id"] and aws_config["aws_secret_access_key"]:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_config["aws_access_key_id"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_config["aws_secret_access_key"]
        os.environ["AWS_DEFAULT_REGION"] = aws_config["aws_region"]
    
    return {
        **bedrock_config,
        "aws_region": aws_config["aws_region"],
    }


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for the agent.

    This can be overridden when invoking the graph.
    
    Returns:
        Default configuration dictionary
    """
    try:
        config = load_config()
        return {
            "knowledge_base_id": config.get("knowledge_base_id", ""),
            "medical_guidelines_kb_id": config["medical_guidelines_kb_id"],
            "cms_coding_kb_id": config["cms_coding_kb_id"],
            "aws_region": config["aws_region"],
            "model_id": config["model_id"],
            "max_results": config["max_results"],
            "temperature": config["temperature"],
            "knowledge_bases": config["knowledge_bases"],
        }
    except ValueError:
        # Return minimal config if environment is not set up
        return {
            "knowledge_base_id": "",
            "medical_guidelines_kb_id": "VXMUOUXXCF",
            "cms_coding_kb_id": "X1DCXMHW9T",
            "aws_region": "us-east-1",
            "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "max_results": 5,
            "temperature": 0.3,
            "knowledge_bases": "both",
        }