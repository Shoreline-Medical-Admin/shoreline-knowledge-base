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
        
        # Add retrieval success step
        self.reasoning_steps.append(ReasoningStep(
            step_type="retrieval",
            description=f"Retrieved {len(documents)} documents from knowledge base(s)",
            confidence_impact=0.1,
            details={"document_count": len(documents)}
        ))
        
        for i, doc in enumerate(documents):
            evaluation = self._evaluate_single_source(doc, i)
            evaluations.append(evaluation)
            total_score += evaluation.overall_score
        
        # Calculate base confidence from source scores
        avg_score = total_score / len(documents) if documents else 0.0
        
        # Check for limited sources
        if len(documents) < 3:
            self.uncertainty_flags.append(UncertaintyType.LIMITED_SOURCES)
            self.reasoning_steps.append(ReasoningStep(
                step_type="source_evaluation",
                description=f"Only {len(documents)} source(s) found - limited information available",
                confidence_impact=-0.2
            ))
        
        # Check for low relevance
        high_relevance_count = sum(1 for e in evaluations if e.relevance_score > 0.8)
        if high_relevance_count == 0:
            self.uncertainty_flags.append(UncertaintyType.LOW_RELEVANCE)
            self.reasoning_steps.append(ReasoningStep(
                step_type="source_evaluation",
                description="No highly relevant sources found (all relevance scores < 80%)",
                confidence_impact=-0.3
            ))
        
        # Detect conflicting information
        if self._detect_conflicts(documents):
            self.uncertainty_flags.append(UncertaintyType.CONFLICTING_SOURCES)
            self.reasoning_steps.append(ReasoningStep(
                step_type="conflict_detection",
                description="Detected potentially conflicting information across sources",
                confidence_impact=-0.25
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
        
        # Overall score (weighted average)
        overall_score = (
            relevance_score * 0.5 +
            recency_score * 0.3 +
            authority_score * 0.2
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
        
        # Apply uncertainty penalties
        uncertainty_penalty = len(self.uncertainty_flags) * 0.1
        confidence -= uncertainty_penalty
        
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
        confidence_level = self.get_confidence_level(confidence_score)
        
        # Confidence indicator
        if confidence_level == ConfidenceLevel.HIGH:
            indicator = "🟢"
        elif confidence_level == ConfidenceLevel.MEDIUM:
            indicator = "🟡"
        elif confidence_level == ConfidenceLevel.LOW:
            indicator = "🟠"
        else:
            indicator = "🔴"
        
        output = f"\n\n{indicator} **Confidence: {confidence_score:.0%}** ({confidence_level.value})\n"
        
        # Uncertainty warnings
        if self.uncertainty_flags:
            output += "\n⚠️ **Important Considerations:**\n"
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
        
        if show_details:
            output += "\n<details>\n<summary>🔍 <b>Show Reasoning Process</b></summary>\n\n"
            output += "### Reasoning Steps:\n"
            
            for i, step in enumerate(self.reasoning_steps, 1):
                output += f"\n{i}. **{step.step_type.replace('_', ' ').title()}**: {step.description}"
                if step.confidence_impact != 0:
                    impact_sign = "+" if step.confidence_impact > 0 else ""
                    output += f" ({impact_sign}{step.confidence_impact:.0%} confidence)"
                output += "\n"
            
            if self.source_evaluations:
                output += "\n### Source Evaluation:\n"
                for eval in self.source_evaluations[:3]:  # Top 3 sources
                    output += f"\n**{eval.source_id}**:\n"
                    output += f"• Relevance: {eval.relevance_score:.0%}\n"
                    output += f"• Recency: {eval.recency_score:.0%}"
                    if "outdated" in eval.flags:
                        output += " ⚠️"
                    output += f"\n• Authority: {eval.authority_score:.0%}\n"
                    output += f"• Overall: {eval.overall_score:.0%}\n"
            
            output += "\n</details>"
        
        return output