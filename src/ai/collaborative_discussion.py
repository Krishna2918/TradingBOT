"""
Collaborative Discussion Framework

This module implements the 5-round collaborative discussion system where
multiple AI models discuss, critique, and reach consensus on trading decisions.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

from src.ai.multi_model import get_multi_model_manager, ModelOpinion, ModelRole
from src.config.mode_manager import get_current_mode

logger = logging.getLogger(__name__)

class DiscussionRound(Enum):
    """Discussion round enumeration."""
    INITIAL_ANALYSIS = "INITIAL_ANALYSIS"
    PEER_REVIEW = "PEER_REVIEW"
    DEFENSE_REFINEMENT = "DEFENSE_REFINEMENT"
    CONSENSUS_BUILDING = "CONSENSUS_BUILDING"
    FINAL_DECISION = "FINAL_DECISION"

@dataclass
class Critique:
    """Critique from one model to another."""
    critic_model: str
    target_model: str
    critique_points: List[str]
    agreement_level: float  # 0.0 to 1.0
    suggestions: List[str]
    timestamp: datetime

@dataclass
class Defense:
    """Defense and refinement from a model."""
    defending_model: str
    original_opinion: ModelOpinion
    defense_points: List[str]
    refinements: List[str]
    updated_confidence: float
    timestamp: datetime

@dataclass
class Consensus:
    """Consensus reached by the models."""
    consensus_recommendation: str  # BUY/SELL/HOLD
    consensus_confidence: float
    agreement_level: float
    supporting_models: List[str]
    dissenting_models: List[str]
    common_ground: List[str]
    remaining_disagreements: List[str]
    timestamp: datetime

@dataclass
class FinalDecision:
    """Final collaborative decision."""
    recommendation: str  # BUY/SELL/HOLD
    confidence: float
    reasoning: str
    risk_assessment: str
    supporting_evidence: List[str]
    concerns: List[str]
    model_contributions: Dict[str, float]
    discussion_summary: str
    timestamp: datetime

@dataclass
class DiscussionSession:
    """Complete discussion session."""
    session_id: str
    topic: str
    market_data: Dict[str, Any]
    initial_opinions: List[ModelOpinion]
    critiques: List[Critique]
    defenses: List[Defense]
    consensus: Optional[Consensus]
    final_decision: Optional[FinalDecision]
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]

class CollaborativeDiscussion:
    """Manages collaborative discussions between AI models."""
    
    def __init__(self, mode: str = "DEMO"):
        self.mode = mode
        self.multi_model_manager = get_multi_model_manager(mode)
        self.discussion_sessions = []
        
        # Discussion configuration
        self.max_rounds = 5
        self.consensus_threshold = 0.7  # 70% agreement required
        self.min_confidence_threshold = 0.6  # Minimum confidence for final decision
        
        logger.info(f"Collaborative Discussion initialized for {mode} mode")
    
    async def conduct_discussion(self, topic: str, market_data: Dict[str, Any]) -> DiscussionSession:
        """Conduct a complete 5-round collaborative discussion."""
        try:
            session_id = f"discussion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            start_time = datetime.now()
            
            logger.info(f"Starting collaborative discussion: {topic}")
            
            # Initialize discussion session
            session = DiscussionSession(
                session_id=session_id,
                topic=topic,
                market_data=market_data,
                initial_opinions=[],
                critiques=[],
                defenses=[],
                consensus=None,
                final_decision=None,
                start_time=start_time,
                end_time=None,
                duration_seconds=None
            )
            
            # Round 1: Initial Analysis
            logger.info("Round 1: Initial Analysis")
            initial_opinions = await self._round_1_initial_analysis(topic, market_data)
            session.initial_opinions = initial_opinions
            
            if not initial_opinions:
                logger.error("No initial opinions received")
                return session
            
            # Round 2: Peer Review
            logger.info("Round 2: Peer Review")
            critiques = await self._round_2_peer_review(initial_opinions, topic, market_data)
            session.critiques = critiques
            
            # Round 3: Defense & Refinement
            logger.info("Round 3: Defense & Refinement")
            defenses = await self._round_3_defense_refinement(initial_opinions, critiques, topic, market_data)
            session.defenses = defenses
            
            # Round 4: Consensus Building
            logger.info("Round 4: Consensus Building")
            consensus = await self._round_4_consensus_building(initial_opinions, critiques, defenses)
            session.consensus = consensus
            
            # Round 5: Final Decision
            logger.info("Round 5: Final Decision")
            final_decision = await self._round_5_final_decision(initial_opinions, critiques, defenses, consensus)
            session.final_decision = final_decision
            
            # Complete session
            session.end_time = datetime.now()
            session.duration_seconds = (session.end_time - session.start_time).total_seconds()
            
            # Store session
            self.discussion_sessions.append(session)
            
            logger.info(f"Discussion completed in {session.duration_seconds:.2f} seconds")
            return session
            
        except Exception as e:
            logger.error(f"Error conducting discussion: {e}")
            return session
    
    async def _round_1_initial_analysis(self, topic: str, market_data: Dict[str, Any]) -> List[ModelOpinion]:
        """Round 1: Each model gives initial opinion."""
        try:
            analysis_prompt = f"""
Please provide your initial analysis on the following topic:

{topic}

Consider your specialized expertise and provide a detailed opinion with:
- Your recommendation (BUY/SELL/HOLD)
- Confidence level (0.0 to 1.0)
- Supporting evidence
- Any concerns or risks
- Specific reasoning based on your expertise
"""
            
            opinions = await self.multi_model_manager.get_all_model_opinions(analysis_prompt, market_data)
            
            logger.info(f"Round 1: Received {len(opinions)} initial opinions")
            return opinions
            
        except Exception as e:
            logger.error(f"Error in Round 1: {e}")
            return []
    
    async def _round_2_peer_review(self, initial_opinions: List[ModelOpinion], 
                                 topic: str, market_data: Dict[str, Any]) -> List[Critique]:
        """Round 2: Models critique each other's analysis."""
        try:
            critiques = []
            
            for opinion in initial_opinions:
                # Get critiques from other models
                other_opinions = [op for op in initial_opinions if op.model_name != opinion.model_name]
                
                for other_opinion in other_opinions:
                    critique_prompt = f"""
Please review the analysis from {other_opinion.model_name} ({other_opinion.role.value}):

ORIGINAL ANALYSIS:
{other_opinion.opinion}

RECOMMENDATION: {other_opinion.recommendation}
CONFIDENCE: {other_opinion.confidence}

Please provide a critique focusing on:
- What do you agree with?
- What do you disagree with?
- What concerns do you have?
- What suggestions do you have for improvement?
- Rate your agreement level (0.0 to 1.0)

Topic: {topic}
"""
                    
                    # Get critique from current model
                    critique_opinion = await self.multi_model_manager.get_model_opinion(
                        opinion.model_name, critique_prompt, market_data
                    )
                    
                    if critique_opinion:
                        # Parse critique
                        critique = self._parse_critique(critique_opinion, opinion.model_name, other_opinion.model_name)
                        critiques.append(critique)
            
            logger.info(f"Round 2: Generated {len(critiques)} critiques")
            return critiques
            
        except Exception as e:
            logger.error(f"Error in Round 2: {e}")
            return []
    
    async def _round_3_defense_refinement(self, initial_opinions: List[ModelOpinion],
                                        critiques: List[Critique], topic: str,
                                        market_data: Dict[str, Any]) -> List[Defense]:
        """Round 3: Models defend and refine their positions."""
        try:
            defenses = []
            
            for opinion in initial_opinions:
                # Get critiques for this model
                model_critiques = [c for c in critiques if c.target_model == opinion.model_name]
                
                if model_critiques:
                    defense_prompt = f"""
You received the following critiques on your analysis:

{self._format_critiques_for_model(model_critiques)}

Your original analysis was:
{opinion.opinion}

Please provide:
- Defense of your position
- Any refinements or updates to your analysis
- Updated confidence level
- Response to the critiques

Topic: {topic}
"""
                    
                    # Get defense from the model
                    defense_opinion = await self.multi_model_manager.get_model_opinion(
                        opinion.model_name, defense_prompt, market_data
                    )
                    
                    if defense_opinion:
                        # Parse defense
                        defense = self._parse_defense(defense_opinion, opinion, model_critiques)
                        defenses.append(defense)
            
            logger.info(f"Round 3: Generated {len(defenses)} defenses")
            return defenses
            
        except Exception as e:
            logger.error(f"Error in Round 3: {e}")
            return []
    
    async def _round_4_consensus_building(self, initial_opinions: List[ModelOpinion],
                                        critiques: List[Critique],
                                        defenses: List[Defense]) -> Optional[Consensus]:
        """Round 4: Build consensus among models."""
        try:
            # Analyze agreement levels
            recommendations = {}
            confidences = {}
            
            # Collect final recommendations and confidences
            for opinion in initial_opinions:
                model_name = opinion.model_name
                recommendations[model_name] = opinion.recommendation
                confidences[model_name] = opinion.confidence
            
            # Update with refinements from defenses
            for defense in defenses:
                model_name = defense.defending_model
                if model_name in confidences:
                    confidences[model_name] = defense.updated_confidence
            
            # Calculate consensus
            consensus = self._calculate_consensus(recommendations, confidences, initial_opinions, critiques)
            
            logger.info(f"Round 4: Consensus reached - {consensus.consensus_recommendation} ({consensus.consensus_confidence:.2f})")
            return consensus
            
        except Exception as e:
            logger.error(f"Error in Round 4: {e}")
            return None
    
    async def _round_5_final_decision(self, initial_opinions: List[ModelOpinion],
                                    critiques: List[Critique], defenses: List[Defense],
                                    consensus: Optional[Consensus]) -> Optional[FinalDecision]:
        """Round 5: Make final collaborative decision."""
        try:
            if not consensus:
                logger.error("No consensus reached, cannot make final decision")
                return None
            
            # Create final decision based on consensus and all discussion elements
            final_decision = self._create_final_decision(initial_opinions, critiques, defenses, consensus)
            
            logger.info(f"Round 5: Final decision - {final_decision.recommendation} ({final_decision.confidence:.2f})")
            return final_decision
            
        except Exception as e:
            logger.error(f"Error in Round 5: {e}")
            return None
    
    def _parse_critique(self, critique_opinion: ModelOpinion, critic_model: str, 
                       target_model: str) -> Critique:
        """Parse critique from model opinion."""
        try:
            # Extract critique points from opinion
            critique_points = []
            suggestions = []
            agreement_level = 0.5  # Default
            
            # Parse the opinion text for critique elements
            opinion_text = critique_opinion.opinion.lower()
            
            # Look for agreement indicators
            if "agree" in opinion_text:
                agreement_level += 0.2
            if "disagree" in opinion_text:
                agreement_level -= 0.2
            if "concern" in opinion_text:
                agreement_level -= 0.1
            
            # Extract critique points from supporting evidence
            critique_points.extend(critique_opinion.supporting_evidence)
            
            # Extract suggestions from concerns (inverted)
            suggestions.extend(critique_opinion.concerns)
            
            return Critique(
                critic_model=critic_model,
                target_model=target_model,
                critique_points=critique_points,
                agreement_level=max(0.0, min(1.0, agreement_level)),
                suggestions=suggestions,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing critique: {e}")
            return Critique(
                critic_model=critic_model,
                target_model=target_model,
                critique_points=[],
                agreement_level=0.5,
                suggestions=[],
                timestamp=datetime.now()
            )
    
    def _parse_defense(self, defense_opinion: ModelOpinion, original_opinion: ModelOpinion,
                      model_critiques: List[Critique]) -> Defense:
        """Parse defense from model opinion."""
        try:
            # Extract defense points
            defense_points = defense_opinion.supporting_evidence.copy()
            
            # Extract refinements
            refinements = defense_opinion.concerns.copy()  # Concerns become refinements
            
            # Calculate updated confidence
            updated_confidence = defense_opinion.confidence
            
            # Adjust confidence based on critiques
            if model_critiques:
                avg_agreement = sum(c.agreement_level for c in model_critiques) / len(model_critiques)
                # If agreement is low, reduce confidence slightly
                if avg_agreement < 0.5:
                    updated_confidence *= 0.9
            
            return Defense(
                defending_model=original_opinion.model_name,
                original_opinion=original_opinion,
                defense_points=defense_points,
                refinements=refinements,
                updated_confidence=max(0.0, min(1.0, updated_confidence)),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing defense: {e}")
            return Defense(
                defending_model=original_opinion.model_name,
                original_opinion=original_opinion,
                defense_points=[],
                refinements=[],
                updated_confidence=original_opinion.confidence,
                timestamp=datetime.now()
            )
    
    def _format_critiques_for_model(self, model_critiques: List[Critique]) -> str:
        """Format critiques for a specific model."""
        formatted = []
        for critique in model_critiques:
            formatted.append(f"""
From {critique.critic_model}:
- Agreement Level: {critique.agreement_level:.2f}
- Critique Points: {', '.join(critique.critique_points)}
- Suggestions: {', '.join(critique.suggestions)}
""")
        return '\n'.join(formatted)
    
    def _calculate_consensus(self, recommendations: Dict[str, str], 
                           confidences: Dict[str, float],
                           initial_opinions: List[ModelOpinion],
                           critiques: List[Critique]) -> Consensus:
        """Calculate consensus from all opinions and critiques."""
        try:
            # Count recommendations
            rec_counts = {}
            for rec in recommendations.values():
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            
            # Find most common recommendation
            consensus_recommendation = max(rec_counts, key=rec_counts.get)
            agreement_count = rec_counts[consensus_recommendation]
            total_models = len(recommendations)
            agreement_level = agreement_count / total_models
            
            # Calculate consensus confidence
            supporting_models = [name for name, rec in recommendations.items() 
                               if rec == consensus_recommendation]
            consensus_confidence = sum(confidences[name] for name in supporting_models) / len(supporting_models)
            
            # Identify supporting and dissenting models
            dissenting_models = [name for name, rec in recommendations.items() 
                               if rec != consensus_recommendation]
            
            # Find common ground
            common_ground = self._find_common_ground(initial_opinions, consensus_recommendation)
            
            # Find remaining disagreements
            remaining_disagreements = self._find_remaining_disagreements(initial_opinions, consensus_recommendation)
            
            return Consensus(
                consensus_recommendation=consensus_recommendation,
                consensus_confidence=consensus_confidence,
                agreement_level=agreement_level,
                supporting_models=supporting_models,
                dissenting_models=dissenting_models,
                common_ground=common_ground,
                remaining_disagreements=remaining_disagreements,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            return Consensus(
                consensus_recommendation="HOLD",
                consensus_confidence=0.5,
                agreement_level=0.5,
                supporting_models=[],
                dissenting_models=[],
                common_ground=[],
                remaining_disagreements=[],
                timestamp=datetime.now()
            )
    
    def _find_common_ground(self, opinions: List[ModelOpinion], consensus_rec: str) -> List[str]:
        """Find common ground among opinions."""
        common_ground = []
        
        # Find common supporting evidence
        all_evidence = []
        for opinion in opinions:
            if opinion.recommendation == consensus_rec:
                all_evidence.extend(opinion.supporting_evidence)
        
        # Find frequently mentioned evidence
        evidence_counts = {}
        for evidence in all_evidence:
            evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
        
        # Add evidence mentioned by multiple models
        for evidence, count in evidence_counts.items():
            if count > 1:
                common_ground.append(evidence)
        
        return common_ground
    
    def _find_remaining_disagreements(self, opinions: List[ModelOpinion], consensus_rec: str) -> List[str]:
        """Find remaining disagreements."""
        disagreements = []
        
        for opinion in opinions:
            if opinion.recommendation != consensus_rec:
                disagreements.extend(opinion.concerns)
        
        return disagreements
    
    def _create_final_decision(self, initial_opinions: List[ModelOpinion],
                             critiques: List[Critique], defenses: List[Defense],
                             consensus: Consensus) -> FinalDecision:
        """Create final collaborative decision."""
        try:
            # Combine all supporting evidence
            supporting_evidence = consensus.common_ground.copy()
            
            # Add evidence from supporting models
            for opinion in initial_opinions:
                if opinion.model_name in consensus.supporting_models:
                    supporting_evidence.extend(opinion.supporting_evidence)
            
            # Combine all concerns
            concerns = consensus.remaining_disagreements.copy()
            
            # Add concerns from dissenting models
            for opinion in initial_opinions:
                if opinion.model_name in consensus.dissenting_models:
                    concerns.extend(opinion.concerns)
            
            # Create reasoning
            reasoning = self._create_reasoning(consensus, initial_opinions, critiques)
            
            # Create risk assessment
            risk_assessment = self._create_risk_assessment(consensus, concerns)
            
            # Calculate model contributions
            model_contributions = self._calculate_model_contributions(initial_opinions, consensus)
            
            # Create discussion summary
            discussion_summary = self._create_discussion_summary(initial_opinions, critiques, defenses, consensus)
            
            return FinalDecision(
                recommendation=consensus.consensus_recommendation,
                confidence=consensus.consensus_confidence,
                reasoning=reasoning,
                risk_assessment=risk_assessment,
                supporting_evidence=supporting_evidence,
                concerns=concerns,
                model_contributions=model_contributions,
                discussion_summary=discussion_summary,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating final decision: {e}")
            return FinalDecision(
                recommendation="HOLD",
                confidence=0.5,
                reasoning="Error in decision creation",
                risk_assessment="Unknown risk",
                supporting_evidence=[],
                concerns=["Error in analysis"],
                model_contributions={},
                discussion_summary="Error in discussion",
                timestamp=datetime.now()
            )
    
    def _create_reasoning(self, consensus: Consensus, opinions: List[ModelOpinion],
                         critiques: List[Critique]) -> str:
        """Create comprehensive reasoning for the final decision."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Consensus reached: {consensus.consensus_recommendation}")
        reasoning_parts.append(f"Agreement level: {consensus.agreement_level:.1%}")
        reasoning_parts.append(f"Supporting models: {', '.join(consensus.supporting_models)}")
        
        if consensus.dissenting_models:
            reasoning_parts.append(f"Dissenting models: {', '.join(consensus.dissenting_models)}")
        
        if consensus.common_ground:
            reasoning_parts.append(f"Common ground: {', '.join(consensus.common_ground[:3])}")
        
        return ". ".join(reasoning_parts)
    
    def _create_risk_assessment(self, consensus: Consensus, concerns: List[str]) -> str:
        """Create risk assessment for the final decision."""
        if consensus.consensus_confidence > 0.8:
            risk_level = "LOW"
        elif consensus.consensus_confidence > 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        risk_assessment = f"Risk Level: {risk_level}"
        
        if concerns:
            risk_assessment += f". Key concerns: {', '.join(concerns[:2])}"
        
        return risk_assessment
    
    def _calculate_model_contributions(self, opinions: List[ModelOpinion],
                                     consensus: Consensus) -> Dict[str, float]:
        """Calculate contribution weights for each model."""
        contributions = {}
        
        for opinion in opinions:
            model_name = opinion.model_name
            
            # Base contribution from confidence
            base_contribution = opinion.confidence
            
            # Bonus for supporting consensus
            if model_name in consensus.supporting_models:
                base_contribution *= 1.2
            
            # Penalty for dissenting
            if model_name in consensus.dissenting_models:
                base_contribution *= 0.8
            
            contributions[model_name] = base_contribution
        
        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            contributions = {name: weight/total_contribution 
                           for name, weight in contributions.items()}
        
        return contributions
    
    def _create_discussion_summary(self, opinions: List[ModelOpinion], critiques: List[Critique],
                                 defenses: List[Defense], consensus: Consensus) -> str:
        """Create summary of the discussion."""
        summary_parts = []
        
        summary_parts.append(f"Discussion involved {len(opinions)} AI models")
        summary_parts.append(f"Generated {len(critiques)} critiques and {len(defenses)} defenses")
        summary_parts.append(f"Reached {consensus.agreement_level:.1%} agreement on {consensus.consensus_recommendation}")
        
        if consensus.dissenting_models:
            summary_parts.append(f"{len(consensus.dissenting_models)} models had different views")
        
        return ". ".join(summary_parts)
    
    def get_discussion_history(self, limit: int = 10) -> List[DiscussionSession]:
        """Get recent discussion sessions."""
        return self.discussion_sessions[-limit:] if self.discussion_sessions else []
    
    def get_discussion_statistics(self) -> Dict[str, Any]:
        """Get statistics about discussions."""
        if not self.discussion_sessions:
            return {}
        
        total_sessions = len(self.discussion_sessions)
        successful_sessions = len([s for s in self.discussion_sessions if s.final_decision])
        
        # Calculate average duration
        durations = [s.duration_seconds for s in self.discussion_sessions if s.duration_seconds]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate recommendation distribution
        recommendations = {}
        for session in self.discussion_sessions:
            if session.final_decision:
                rec = session.final_decision.recommendation
                recommendations[rec] = recommendations.get(rec, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": successful_sessions / total_sessions if total_sessions > 0 else 0,
            "average_duration_seconds": avg_duration,
            "recommendation_distribution": recommendations
        }
    
    def validate_discussion(self) -> bool:
        """Validate discussion system."""
        try:
            # Check if multi-model manager is available
            if not self.multi_model_manager:
                logger.error("Multi-model manager not available")
                return False
            
            # Check if models are configured
            if not self.multi_model_manager.validate_models():
                logger.error("Model validation failed")
                return False
            
            logger.info("Discussion system validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Discussion system validation error: {e}")
            return False

# Global collaborative discussion instance
_collaborative_discussion: Optional[CollaborativeDiscussion] = None

def get_collaborative_discussion(mode: str = None) -> CollaborativeDiscussion:
    """Get the global collaborative discussion instance."""
    global _collaborative_discussion
    if _collaborative_discussion is None:
        if mode is None:
            mode = get_current_mode()
        _collaborative_discussion = CollaborativeDiscussion(mode)
    return _collaborative_discussion

async def conduct_discussion(topic: str, market_data: Dict[str, Any], mode: str = None) -> DiscussionSession:
    """Conduct a collaborative discussion."""
    return await get_collaborative_discussion(mode).conduct_discussion(topic, market_data)

def get_discussion_history(limit: int = 10, mode: str = None) -> List[DiscussionSession]:
    """Get discussion history."""
    return get_collaborative_discussion(mode).get_discussion_history(limit)

def get_discussion_statistics(mode: str = None) -> Dict[str, Any]:
    """Get discussion statistics."""
    return get_collaborative_discussion(mode).get_discussion_statistics()

def validate_discussion_system(mode: str = None) -> bool:
    """Validate discussion system."""
    return get_collaborative_discussion(mode).validate_discussion()
