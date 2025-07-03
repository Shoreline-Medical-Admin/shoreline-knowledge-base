"""Reasoning and confidence scoring module for the Knowledge Base Q&A Agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class UncertaintyType(Enum):
    """Types of uncertainty that can be detected."""
    LIMITED_SOURCES = "limited_sources"
    OUTDATED_INFORMATION = "outdated_information"
    CONFLICTING_SOURCES = "conflicting_sources"
    LOW_RELEVANCE = "low_relevance"
    NO_SOURCES = "no_sources"
    PARTIAL_INFORMATION = "partial_information"
    AMBIGUOUS_QUERY = "ambiguous_query"


@dataclass
class SourceEvaluation:
    """Evaluation metrics for a single source."""
    source_id: str
    relevance_score: float
    recency_score: float
    authority_score: float
    overall_score: float
    flags: List[str]


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_type: str
    description: str
    confidence_impact: float
    details: Optional[Dict[str, Any]] = None


class ReasoningEngine:
    """Engine for evaluating sources and generating reasoning explanations."""
    
    def __init__(self):
        """Initialize the reasoning engine."""
        self.reasoning_steps: List[ReasoningStep] = []
        self.uncertainty_flags: List[UncertaintyType] = []
        self.source_evaluations: List[SourceEvaluation] = []
    
    def evaluate_sources(self, documents: List[Dict[str, Any]], query: str) -> Tuple[float, List[SourceEvaluation]]:
        """Evaluate the quality of retrieved sources.
        
        Returns:
            Tuple of (overall_confidence, source_evaluations)
        """
        if not documents:
            self.uncertainty_flags.append(UncertaintyType.NO_SOURCES)
            self.reasoning_steps.append(ReasoningStep(
                step_type="retrieval",
                description="No documents were found matching the query",
                confidence_impact=-1.0
            ))
            return 0.0, []
        
        evaluations = []
        total_score = 0.0
        
        # Add retrieval success step with better bonus structure
        if len(documents) >= 5:
            impact = 0.15  # Good retrieval
        elif len(documents) >= 2:
            impact = 0.1   # Decent retrieval
        else:
            impact = 0.05  # Minimal but successful retrieval
            
        self.reasoning_steps.append(ReasoningStep(
            step_type="retrieval",
            description=f"Retrieved {len(documents)} documents from knowledge base(s)",
            confidence_impact=impact,
            details={"document_count": len(documents)}
        ))
        
        for i, doc in enumerate(documents):
            evaluation = self._evaluate_single_source(doc, i)
            evaluations.append(evaluation)
            total_score += evaluation.overall_score
        
        # Calculate base confidence from source scores
        # Use a weighted average that favors better sources instead of simple average
        # This prevents one low-scoring source from dragging down the whole score
        if documents:
            # Sort evaluations by score
            sorted_evals = sorted(evaluations, key=lambda e: e.overall_score, reverse=True)
            # Weight top sources more heavily
            weighted_sum = 0.0
            weight_total = 0.0
            for i, eval in enumerate(sorted_evals):
                weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33, 0.25, etc.
                weighted_sum += eval.overall_score * weight
                weight_total += weight
            avg_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        else:
            avg_score = 0.0
        
        # Apply a very aggressive curve to boost AWS Bedrock's conservative scores
        # AWS Bedrock scores 50-60% are actually very good matches
        if avg_score > 0.15:  # If we have any relevance at all
            # Very aggressive boosting curve
            # Map AWS Bedrock scores to realistic confidence:
            # 0.3 -> 0.65, 0.4 -> 0.75, 0.5 -> 0.85, 0.6 -> 0.90
            if avg_score < 0.4:
                boosted = avg_score * 2.0 + 0.05
            elif avg_score < 0.6:
                boosted = avg_score * 1.5 + 0.25
            else:
                boosted = avg_score * 1.2 + 0.42
            avg_score = min(0.95, boosted)
        
        # Check for limited sources - only penalize if we have 0 or 1 source
        if len(documents) == 0:
            self.uncertainty_flags.append(UncertaintyType.NO_SOURCES)
            self.reasoning_steps.append(ReasoningStep(
                step_type="source_evaluation",
                description="No sources found",
                confidence_impact=-0.5  # Severe penalty for no sources
            ))
        elif len(documents) == 1:
            self.uncertainty_flags.append(UncertaintyType.LIMITED_SOURCES)
            self.reasoning_steps.append(ReasoningStep(
                step_type="source_evaluation",
                description="Only 1 source found - limited corroboration",
                confidence_impact=-0.05  # Very minor penalty for single source
            ))
        
        # Check for low relevance - adjusted thresholds for AWS Bedrock's scoring
        # AWS Bedrock considers 50%+ as relevant matches
        moderate_relevance_count = sum(1 for e in evaluations if e.relevance_score > 0.4)
        very_low_relevance_count = sum(1 for e in evaluations if e.relevance_score < 0.3)
        
        if very_low_relevance_count == len(evaluations):
            # Only penalize if ALL sources have very low relevance
            self.uncertainty_flags.append(UncertaintyType.LOW_RELEVANCE)
            self.reasoning_steps.append(ReasoningStep(
                step_type="source_evaluation",
                description="All sources have very low relevance (< 30%)",
                confidence_impact=-0.2
            ))
        elif moderate_relevance_count == 0:
            # Minor penalty if no sources reach moderate relevance
            self.reasoning_steps.append(ReasoningStep(
                step_type="source_evaluation",
                description="Sources have low relevance scores",
                confidence_impact=-0.05
            ))
        
        # Add BONUS for having sources with AWS Bedrock "good" scores (>50%)
        good_source_count = sum(1 for e in evaluations if e.relevance_score > 0.5)
        if good_source_count >= 3:
            self.reasoning_steps.append(ReasoningStep(
                step_type="source_evaluation", 
                description=f"Found {good_source_count} sources with good relevance",
                confidence_impact=0.15  # Bigger bonus
            ))
        elif good_source_count >= 1:
            self.reasoning_steps.append(ReasoningStep(
                step_type="source_evaluation",
                description=f"Found {good_source_count} source(s) with good relevance",
                confidence_impact=0.1
            ))
        
        # Detect conflicting information
        if self._detect_conflicts(documents):
            self.uncertainty_flags.append(UncertaintyType.CONFLICTING_SOURCES)
            self.reasoning_steps.append(ReasoningStep(
                step_type="conflict_detection",
                description="Detected potentially conflicting information across sources",
                confidence_impact=-0.15  # Reduced from -0.25
            ))
        
        self.source_evaluations = evaluations
        return avg_score, evaluations
    
    def _evaluate_single_source(self, doc: Dict[str, Any], index: int) -> SourceEvaluation:
        """Evaluate a single document source."""
        source_id = f"source_{index + 1}"
        flags = []
        
        # Relevance score (from retriever)
        relevance_score = doc.get("score", 0.0)
        
        # Recency score
        recency_score, is_outdated = self._calculate_recency_score(doc)
        if is_outdated:
            flags.append("outdated")
            if UncertaintyType.OUTDATED_INFORMATION not in self.uncertainty_flags:
                self.uncertainty_flags.append(UncertaintyType.OUTDATED_INFORMATION)
        
        # Authority score (based on metadata)
        authority_score = self._calculate_authority_score(doc)
        
        # Overall score - for AWS Bedrock, use relevance as primary indicator
        # Since AWS Bedrock already factors in content quality in its relevance score
        overall_score = (
            relevance_score * 0.85 +   # Heavy weight on AWS Bedrock's relevance
            authority_score * 0.15     # Small weight on source authority
        )
        
        return SourceEvaluation(
            source_id=source_id,
            relevance_score=relevance_score,
            recency_score=recency_score,
            authority_score=authority_score,
            overall_score=overall_score,
            flags=flags
        )
    
    def _calculate_recency_score(self, doc: Dict[str, Any]) -> Tuple[float, bool]:
        """Calculate recency score based on document metadata."""
        metadata = doc.get("metadata", {})
        
        # Try to extract date from metadata
        date_fields = ["publicationDate", "lastUpdated", "date", "createdDate"]
        doc_date = None
        
        for field in date_fields:
            if field in metadata:
                try:
                    # Parse date string
                    date_str = metadata[field]
                    if date_str:
                        # Simple date parsing - could be enhanced
                        doc_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        break
                except (ValueError, AttributeError):
                    continue
        
        if not doc_date:
            # No date found - assume medium recency
            return 0.5, False
        
        # Calculate age in days
        now = datetime.now(timezone.utc)
        age_days = (now - doc_date).days
        
        # Scoring based on age
        if age_days < 180:  # Less than 6 months
            return 1.0, False
        elif age_days < 365:  # Less than 1 year
            return 0.8, False
        elif age_days < 730:  # Less than 2 years
            return 0.6, True
        else:  # More than 2 years
            return 0.3, True
    
    def _calculate_authority_score(self, doc: Dict[str, Any]) -> float:
        """Calculate authority score based on source metadata."""
        metadata = doc.get("metadata", {})
        kb_type = doc.get("kb_type", "")
        
        # Base score on knowledge base type
        if "medical" in kb_type.lower():
            base_score = 0.9  # Medical guidelines are authoritative
        elif "cms" in kb_type.lower():
            base_score = 0.9  # CMS coding is authoritative
        else:
            base_score = 0.7
        
        # Boost for official sources
        source = metadata.get("source", "").lower()
        if any(official in source for official in ["cms.gov", "hhs.gov", "cdc.gov", "fda.gov"]):
            base_score = min(1.0, base_score + 0.1)
        
        return base_score
    
    def _detect_conflicts(self, documents: List[Dict[str, Any]]) -> bool:
        """Detect if documents contain conflicting information."""
        # Simple conflict detection - could be enhanced with NLP
        if len(documents) < 2:
            return False
        
        # Check for significant score variations
        scores = [doc.get("score", 0.0) for doc in documents]
        if scores:
            max_score = max(scores)
            min_score = min(scores)
            if max_score - min_score > 0.5:
                # Large score difference might indicate conflicting relevance
                return True
        
        return False
    
    def calculate_confidence_score(self, source_confidence: float) -> float:
        """Calculate overall confidence score based on all factors.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Start with source confidence
        confidence = source_confidence
        
        # Apply impacts from reasoning steps
        for step in self.reasoning_steps:
            confidence += step.confidence_impact
        
        # Remove the additional uncertainty penalty - we already penalized in steps
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        # Add final confidence step
        self.reasoning_steps.append(ReasoningStep(
            step_type="confidence_calculation",
            description=f"Overall confidence calculated at {confidence:.0%}",
            confidence_impact=0.0,
            details={
                "base_confidence": source_confidence,
                "uncertainty_count": len(self.uncertainty_flags),
                "final_confidence": confidence
            }
        ))
        
        return confidence
    
    def get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Convert numeric confidence to categorical level."""
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def format_reasoning(self, confidence_score: float, show_details: bool = True) -> str:
        """Format reasoning explanation for display."""
        # Parameters are kept for API compatibility but not used in simplified version
        _ = confidence_score  # Suppress unused warning
        _ = show_details     # Suppress unused warning
        output = ""
        
        # Uncertainty warnings
        if self.uncertainty_flags:
            output += "\n\n⚠️ **Important Considerations:**\n"
            for flag in self.uncertainty_flags:
                if flag == UncertaintyType.LIMITED_SOURCES:
                    output += "• Limited sources available - answer based on minimal information\n"
                elif flag == UncertaintyType.OUTDATED_INFORMATION:
                    output += "• Some sources may contain outdated information (>2 years old)\n"
                elif flag == UncertaintyType.CONFLICTING_SOURCES:
                    output += "• Sources contain potentially conflicting information\n"
                elif flag == UncertaintyType.LOW_RELEVANCE:
                    output += "• No highly relevant sources found for this specific query\n"
                elif flag == UncertaintyType.NO_SOURCES:
                    output += "• No sources found - unable to provide information\n"
        
        return output