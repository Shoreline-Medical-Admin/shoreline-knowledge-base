"""Unit tests for configuration management."""

import os
from unittest.mock import patch
import pytest

from agent.config import load_config, get_default_config


class TestConfiguration:
    """Test cases for configuration loading."""

    def test_load_config_with_env_vars(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_DEFAULT_REGION": "us-west-2",
            "BEDROCK_KNOWLEDGE_BASE_ID": "test-kb-id",
            "BEDROCK_MODEL_ID": "test-model",
            "BEDROCK_MODEL_TEMPERATURE": "0.5",
            "BEDROCK_MAX_RESULTS": "10"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_config()
            
            assert config["knowledge_base_id"] == "test-kb-id"
            assert config["model_id"] == "test-model"
            assert config["temperature"] == 0.5
            assert config["max_results"] == 10
            assert config["aws_region"] == "us-west-2"

    def test_load_config_missing_kb_id(self):
        """Test configuration validation when KB ID is missing."""
        env_vars = {
            "AWS_DEFAULT_REGION": "us-east-1"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError) as exc_info:
                load_config()
            
            assert "BEDROCK_KNOWLEDGE_BASE_ID environment variable is required" in str(exc_info.value)

    def test_load_config_defaults(self):
        """Test configuration with default values."""
        env_vars = {
            "BEDROCK_KNOWLEDGE_BASE_ID": "test-kb-id"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_config()
            
            assert config["knowledge_base_id"] == "test-kb-id"
            assert config["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert config["temperature"] == 0.3
            assert config["max_results"] == 5
            assert config["aws_region"] == "us-east-1"

    def test_get_default_config_success(self):
        """Test getting default configuration."""
        env_vars = {
            "BEDROCK_KNOWLEDGE_BASE_ID": "test-kb-id"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = get_default_config()
            
            assert config["knowledge_base_id"] == "test-kb-id"
            assert "aws_region" in config
            assert "model_id" in config
            assert "max_results" in config
            assert "temperature" in config

    def test_get_default_config_no_env(self):
        """Test getting default configuration without environment setup."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_default_config()
            
            # Should return minimal config without raising error
            assert config["knowledge_base_id"] == ""
            assert config["aws_region"] == "us-east-1"
            assert config["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
            assert config["max_results"] == 5
            assert config["temperature"] == 0.3
