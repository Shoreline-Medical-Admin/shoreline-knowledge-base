"""Unit tests for the reasoning module."""

import pytest
from datetime import datetime, timedelta, timezone
from agent.reasoning import (
    ReasoningEngine,
    ConfidenceLevel,
    UncertaintyType,
    SourceEvaluation,
    ReasoningStep
)


class TestReasoningEngine:
    """Test cases for the ReasoningEngine class."""
    
    def test_initialization(self):
        """Test ReasoningEngine initialization."""
        engine = ReasoningEngine()
        assert engine.reasoning_steps == []
        assert engine.uncertainty_flags == []
        assert engine.source_evaluations == []
    
    def test_evaluate_sources_no_documents(self):
        """Test source evaluation with no documents."""
        engine = ReasoningEngine()
        confidence, evaluations = engine.evaluate_sources([], "test query")
        
        assert confidence == 0.0
        assert evaluations == []
        assert UncertaintyType.NO_SOURCES in engine.uncertainty_flags
        assert len(engine.reasoning_steps) == 1
        assert engine.reasoning_steps[0].description == "No documents were found matching the query"
    
    def test_evaluate_sources_with_documents(self):
        """Test source evaluation with multiple documents."""
        engine = ReasoningEngine()
        documents = [
            {
                "content": "Test content 1",
                "score": 0.95,
                "metadata": {"source": "doc1.pdf"},
                "kb_type": "medical_guidelines"
            },
            {
                "content": "Test content 2",
                "score": 0.85,
                "metadata": {"source": "doc2.pdf"},
                "kb_type": "cms_coding"
            }
        ]
        
        confidence, evaluations = engine.evaluate_sources(documents, "test query")
        
        assert len(evaluations) == 2
        assert evaluations[0].relevance_score == 0.95
        assert evaluations[1].relevance_score == 0.85
        assert confidence > 0  # Should have positive confidence
        assert len(engine.reasoning_steps) >= 1
    
    def test_limited_sources_detection(self):
        """Test detection of limited sources."""
        engine = ReasoningEngine()
        documents = [
            {"content": "Test", "score": 0.9, "metadata": {}}
        ]
        
        confidence, evaluations = engine.evaluate_sources(documents, "test query")
        
        assert UncertaintyType.LIMITED_SOURCES in engine.uncertainty_flags
        assert any(step.description.startswith("Only 1 source(s) found") 
                  for step in engine.reasoning_steps)
    
    def test_outdated_information_detection(self):
        """Test detection of outdated information."""
        engine = ReasoningEngine()
        old_date = (datetime.now(timezone.utc) - timedelta(days=800)).isoformat()
        documents = [
            {
                "content": "Old content",
                "score": 0.9,
                "metadata": {"publicationDate": old_date}
            }
        ]
        
        confidence, evaluations = engine.evaluate_sources(documents, "test query")
        
        assert UncertaintyType.OUTDATED_INFORMATION in engine.uncertainty_flags
        assert "outdated" in evaluations[0].flags
    
    def test_low_relevance_detection(self):
        """Test detection of low relevance sources."""
        engine = ReasoningEngine()
        documents = [
            {"content": "Test 1", "score": 0.5, "metadata": {}},
            {"content": "Test 2", "score": 0.6, "metadata": {}},
            {"content": "Test 3", "score": 0.7, "metadata": {}}
        ]
        
        confidence, evaluations = engine.evaluate_sources(documents, "test query")
        
        assert UncertaintyType.LOW_RELEVANCE in engine.uncertainty_flags
        assert any(step.description.startswith("No highly relevant sources found") 
                  for step in engine.reasoning_steps)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        engine = ReasoningEngine()
        
        # Add some reasoning steps with impacts
        engine.reasoning_steps.append(ReasoningStep(
            step_type="test",
            description="Positive impact",
            confidence_impact=0.2
        ))
        engine.reasoning_steps.append(ReasoningStep(
            step_type="test",
            description="Negative impact",
            confidence_impact=-0.1
        ))
        
        # Add uncertainty flags
        engine.uncertainty_flags = [UncertaintyType.LIMITED_SOURCES]
        
        confidence = engine.calculate_confidence_score(0.5)
        
        # Base 0.5 + 0.2 - 0.1 - 0.1 (one uncertainty flag) = 0.5
        assert confidence == 0.5
    
    def test_confidence_level_mapping(self):
        """Test confidence score to level mapping."""
        engine = ReasoningEngine()
        
        assert engine.get_confidence_level(0.9) == ConfidenceLevel.HIGH
        assert engine.get_confidence_level(0.7) == ConfidenceLevel.MEDIUM
        assert engine.get_confidence_level(0.5) == ConfidenceLevel.LOW
        assert engine.get_confidence_level(0.2) == ConfidenceLevel.VERY_LOW
    
    def test_authority_score_calculation(self):
        """Test authority score calculation for different sources."""
        engine = ReasoningEngine()
        
        # Medical KB should have high authority
        medical_doc = {"kb_type": "medical_guidelines", "metadata": {}}
        eval1 = engine._evaluate_single_source(medical_doc, 0)
        assert eval1.authority_score == 0.9
        
        # Official source should get boost
        official_doc = {
            "kb_type": "general",
            "metadata": {"source": "cms.gov/guidelines.pdf"}
        }
        eval2 = engine._evaluate_single_source(official_doc, 0)
        assert eval2.authority_score >= 0.79  # Allow for floating point precision
    
    def test_format_reasoning(self):
        """Test reasoning formatting output."""
        engine = ReasoningEngine()
        
        # Add some test data
        engine.reasoning_steps.append(ReasoningStep(
            step_type="retrieval",
            description="Retrieved 5 documents",
            confidence_impact=0.1
        ))
        engine.uncertainty_flags.append(UncertaintyType.LIMITED_SOURCES)
        
        output = engine.format_reasoning(0.7, show_details=True)
        
        assert "🟡" in output  # Medium confidence indicator
        assert "70%" in output
        assert "Limited sources available" in output
        assert "<details>" in output
        assert "Reasoning Steps:" in output
    
    def test_format_reasoning_no_details(self):
        """Test reasoning formatting without details."""
        engine = ReasoningEngine()
        engine.uncertainty_flags.append(UncertaintyType.OUTDATED_INFORMATION)
        
        output = engine.format_reasoning(0.85, show_details=False)
        
        assert "🟢" in output  # High confidence indicator
        assert "85%" in output
        assert "outdated information" in output
        assert "<details>" not in output
    
    def test_recency_score_calculation(self):
        """Test recency score calculation for different dates."""
        engine = ReasoningEngine()
        
        # Recent document (< 6 months)
        recent_date = datetime.now(timezone.utc).isoformat()
        recent_doc = {"metadata": {"publicationDate": recent_date}}
        score1, outdated1 = engine._calculate_recency_score(recent_doc)
        assert score1 == 1.0
        assert not outdated1
        
        # Old document (> 2 years)
        old_date = (datetime.now(timezone.utc) - timedelta(days=900)).isoformat()
        old_doc = {"metadata": {"publicationDate": old_date}}
        score2, outdated2 = engine._calculate_recency_score(old_doc)
        assert score2 == 0.3
        assert outdated2
        
        # No date
        no_date_doc = {"metadata": {}}
        score3, outdated3 = engine._calculate_recency_score(no_date_doc)
        assert score3 == 0.5
        assert not outdated3
    
    def test_conflict_detection(self):
        """Test conflict detection between sources."""
        engine = ReasoningEngine()
        
        # Large score difference should indicate conflict
        conflicting_docs = [
            {"content": "Test 1", "score": 0.95},
            {"content": "Test 2", "score": 0.35}
        ]
        
        assert engine._detect_conflicts(conflicting_docs) == True
        
        # Similar scores should not indicate conflict
        similar_docs = [
            {"content": "Test 1", "score": 0.85},
            {"content": "Test 2", "score": 0.80}
        ]
        
        assert engine._detect_conflicts(similar_docs) == False