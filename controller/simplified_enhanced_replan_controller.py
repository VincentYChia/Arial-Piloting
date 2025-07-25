from typing import Optional, Tuple, Dict, Any
import time
import json
from dataclasses import dataclass
from enum import Enum

from .utils import print_t


class ReplanDecision(Enum):
    """Simplified replan decisions - only two types with manual stop option available"""
    COMPLETE = "COMPLETE"                    # Task finished successfully
    REPLAN_CONTINUE = "REPLAN_CONTINUE"      # Continue with current approach, make adjustments


@dataclass
class PipelineState:
    """Complete state of the agentic pipeline for replan assessment"""
    # Task information
    original_task: str
    current_iteration: int
    total_execution_time: float
    
    # Planning stage data
    past_reasoning: str
    generated_minispec: str
    planning_time: float
    
    # Execution stage data
    executed_commands: str
    execution_result: Any
    execution_time: float
    execution_success: bool
    
    # Environment data
    initial_scene: str
    current_scene: str
    scene_changes: str
    
    # Auto-correction data
    corrections_applied: bool
    correction_details: str
    
    # Progress tracking
    estimated_progress: int
    success_indicators: str
    failure_indicators: str


@dataclass
class ReplanResponse:
    """Simplified response from the enhanced replan controller"""
    decision: ReplanDecision
    confidence: float              # 0.0-1.0 confidence in decision
    progress_estimate: int         # 0-100 progress percentage
    reasoning: str                 # Detailed reasoning for decision
    next_guidance: Optional[str]   # Guidance for next iteration if continuing
    safety_notes: str             # Safety considerations
    execution_limit_reached: bool  # If max iterations/time exceeded


class SimplifiedEnhancedReplanController:
    """
    Simplified enhanced replan controller with only two decision types
    COMPLETE or REPLAN_CONTINUE (manual stop available for user intervention)
    """
    
    def __init__(self, llm_wrapper):
        self.llm_wrapper = llm_wrapper
        
        # Safety limits (configurable)
        self.max_iterations = 5           # Maximum planning iterations per task
        self.max_execution_time = 300.0   # Maximum total execution time (5 minutes)
        self.max_planning_failures = 2    # Maximum consecutive planning failures
        
        # Progress tracking
        self.current_task_start_time = None
        self.consecutive_failures = 0
        self.iteration_history = []
        
        # Model configuration
        self.replan_model = "qwen3:4b"
        self.replan_temperature = 0.2
        
        print_t("[SERC] Simplified Enhanced Replan Controller initialized (COMPLETE/CONTINUE only)")

    def assess_pipeline_state(self, pipeline_state: PipelineState) -> ReplanResponse:
        """
        Simplified assessment method - analyzes pipeline state for COMPLETE vs CONTINUE
        
        Args:
            pipeline_state: Complete state of the current pipeline execution
            
        Returns:
            ReplanResponse with simplified decision (COMPLETE or REPLAN_CONTINUE)
        """
        print_t(f"[SERC] ========== SIMPLIFIED PIPELINE ASSESSMENT ==========")
        print_t(f"[SERC] Task: {pipeline_state.original_task}")
        print_t(f"[SERC] Iteration: {pipeline_state.current_iteration}")
        print_t(f"[SERC] Execution time: {pipeline_state.total_execution_time:.1f}s")
        
        # Safety checks first (override all other decisions)
        safety_check = self._check_safety_limits(pipeline_state)
        if safety_check is not None:
            print_t(f"[SERC] Safety limit triggered: {safety_check.decision}")
            return safety_check
        
        # Prepare comprehensive input for replan model
        assessment_input = self._prepare_assessment_input(pipeline_state)
        
        # Get model assessment
        try:
            model_response = self._query_replan_model(assessment_input)
            
            # Parse and validate response
            parsed_response = self._parse_model_response(model_response, pipeline_state)
            
            # Apply safety defaults if model response is unclear
            final_response = self._apply_safety_defaults(parsed_response, pipeline_state)
            
            print_t(f"[SERC] Final decision: {final_response.decision.value}")
            print_t(f"[SERC] Confidence: {final_response.confidence:.2f}")
            print_t(f"[SERC] Progress: {final_response.progress_estimate}%")
            
            return final_response
            
        except Exception as e:
            print_t(f"[SERC] Error in model assessment: {e}")
            return self._create_safety_fallback_response(pipeline_state, f"Assessment error: {e}")

    def _check_safety_limits(self, pipeline_state: PipelineState) -> Optional[ReplanResponse]:
        """
        Check safety limits and return COMPLETE decision if limits exceeded
        Safety is the top priority - overrides all other decisions
        """
        # Check maximum iterations
        if pipeline_state.current_iteration >= self.max_iterations:
            return ReplanResponse(
                decision=ReplanDecision.COMPLETE,
                confidence=1.0,
                progress_estimate=100,  # Mark as complete to stop execution
                reasoning=f"Maximum iterations ({self.max_iterations}) reached for safety",
                next_guidance=None,
                safety_notes="SAFETY LIMIT: Task marked complete to prevent infinite loops",
                execution_limit_reached=True
            )
        
        # Check maximum execution time
        if pipeline_state.total_execution_time >= self.max_execution_time:
            return ReplanResponse(
                decision=ReplanDecision.COMPLETE,
                confidence=1.0,
                progress_estimate=100,  # Mark as complete to stop execution
                reasoning=f"Maximum execution time ({self.max_execution_time}s) reached for safety",
                next_guidance=None,
                safety_notes="SAFETY LIMIT: Task marked complete to prevent excessive execution time",
                execution_limit_reached=True
            )
        
        # Check consecutive failures
        if not pipeline_state.execution_success:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_planning_failures:
                return ReplanResponse(
                    decision=ReplanDecision.COMPLETE,
                    confidence=0.8,
                    progress_estimate=100,  # Mark as complete to stop execution
                    reasoning=f"Maximum consecutive failures ({self.max_planning_failures}) reached",
                    next_guidance=None,
                    safety_notes="SAFETY LIMIT: Too many consecutive failures, marking task complete",
                    execution_limit_reached=True
                )
        else:
            self.consecutive_failures = 0  # Reset on success
        
        return None  # No safety limits triggered

    def _prepare_assessment_input(self, pipeline_state: PipelineState) -> str:
        """
        Prepare comprehensive, well-structured input for the simplified replan model
        """
        assessment_prompt = f"""You are a simplified replan controller managing an autonomous robot's task execution. Analyze the pipeline state and make a simple decision: COMPLETE or REPLAN_CONTINUE.

## TASK INFORMATION
**Original Task:** {pipeline_state.original_task}
**Current Iteration:** {pipeline_state.current_iteration}
**Total Execution Time:** {pipeline_state.total_execution_time:.1f} seconds

## PLANNING STAGE RESULTS
**Reasoning Generated:** {pipeline_state.past_reasoning}
**MiniSpec Code:** {pipeline_state.generated_minispec}
**Planning Time:** {pipeline_state.planning_time:.1f}s

## EXECUTION STAGE RESULTS  
**Commands Executed:** {pipeline_state.executed_commands}
**Execution Result:** {pipeline_state.execution_result}
**Execution Time:** {pipeline_state.execution_time:.1f}s
**Execution Success:** {pipeline_state.execution_success}

## ENVIRONMENT STATE
**Initial Scene:** {pipeline_state.initial_scene}
**Current Scene:** {pipeline_state.current_scene}
**Scene Changes:** {pipeline_state.scene_changes}

## AUTO-CORRECTION
**Corrections Applied:** {pipeline_state.corrections_applied}
**Correction Details:** {pipeline_state.correction_details}

## PROGRESS INDICATORS
**Estimated Progress:** {pipeline_state.estimated_progress}%
**Success Indicators:** {pipeline_state.success_indicators}
**Failure Indicators:** {pipeline_state.failure_indicators}

## SIMPLIFIED DECISION FRAMEWORK
You have only TWO options:

1. **COMPLETE:** The task is finished successfully or has reached a reasonable completion state
2. **REPLAN_CONTINUE:** More work is needed, continue with similar approach (possibly with minor adjustments)

Note: User has manual stop capability, so no abort/restart options needed.

## RESPONSE FORMAT
Provide your response in the following JSON format:

{{
    "decision": "COMPLETE|REPLAN_CONTINUE",
    "confidence": 0.0-1.0,
    "progress_estimate": 0-100,
    "reasoning": "Clear explanation of your decision",
    "next_guidance": "Specific guidance for next iteration if continuing (null if completing)",
    "safety_notes": "Any safety considerations"
}}

## DECISION GUIDELINES
- **COMPLETE:** Task appears finished, objective achieved, or reasonable stopping point reached
- **REPLAN_CONTINUE:** Task not complete, should try again with current or slightly modified approach

**SAFETY DEFAULT:** When uncertain, prefer COMPLETE over continuing execution.

Provide your assessment:"""

        return assessment_prompt

    def _query_replan_model(self, assessment_input: str) -> str:
        """
        Query the replan model with the prepared input
        """
        print_t(f"[SERC] Querying replan model: {self.replan_model}")
        print_t(f"[SERC] Input length: {len(assessment_input)} characters")
        
        response = self.llm_wrapper.request(
            prompt=assessment_input,
            model_name=self.replan_model,
            stream=False
        )
        
        print_t(f"[SERC] Model response length: {len(response)} characters")
        print_t(f"[SERC] Raw response preview: {response[:200]}...")
        
        return response

    def _parse_model_response(self, model_response: str, pipeline_state: PipelineState) -> ReplanResponse:
        """
        Parse the model response into a structured ReplanResponse
        Handles various response formats and provides fallbacks
        """
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', model_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                response_data = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                response_data = json.loads(model_response)
            
            # Extract and validate fields
            decision_str = response_data.get("decision", "COMPLETE")
            try:
                decision = ReplanDecision(decision_str)
            except ValueError:
                print_t(f"[SERC] Invalid decision '{decision_str}', defaulting to COMPLETE")
                decision = ReplanDecision.COMPLETE
            
            confidence = float(response_data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            progress = int(response_data.get("progress_estimate", 50))
            progress = max(0, min(100, progress))  # Clamp to [0,100]
            
            reasoning = response_data.get("reasoning", "No reasoning provided")
            next_guidance = response_data.get("next_guidance", None)
            if next_guidance == "null" or next_guidance == "":
                next_guidance = None
                
            safety_notes = response_data.get("safety_notes", "")
            
            return ReplanResponse(
                decision=decision,
                confidence=confidence,
                progress_estimate=progress,
                reasoning=reasoning,
                next_guidance=next_guidance,
                safety_notes=safety_notes,
                execution_limit_reached=False
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print_t(f"[SERC] Error parsing model response: {e}")
            print_t(f"[SERC] Response content: {model_response}")
            
            # Fallback: try to extract decision from text
            return self._extract_decision_from_text(model_response, pipeline_state)

    def _extract_decision_from_text(self, response_text: str, pipeline_state: PipelineState) -> ReplanResponse:
        """
        Fallback method to extract decision from unstructured text response
        """
        response_lower = response_text.lower()
        
        # Simple keyword-based decision extraction
        if any(word in response_lower for word in ["complete", "finished", "done", "success"]):
            decision = ReplanDecision.COMPLETE
            progress = 100
            confidence = 0.7
        else:
            # Default to continue if not clearly complete
            decision = ReplanDecision.REPLAN_CONTINUE
            progress = pipeline_state.estimated_progress
            confidence = 0.6
        
        return ReplanResponse(
            decision=decision,
            confidence=confidence,
            progress_estimate=progress,
            reasoning=f"Extracted from text: {response_text[:200]}...",
            next_guidance=None,
            safety_notes="Fallback text parsing used",
            execution_limit_reached=False
        )

    def _apply_safety_defaults(self, response: ReplanResponse, pipeline_state: PipelineState) -> ReplanResponse:
        """
        Apply safety defaults to ensure safe execution
        This is the final safety check before returning the decision
        """
        # If confidence is very low, default to completion for safety
        if response.confidence < 0.3:
            print_t(f"[SERC] Low confidence ({response.confidence:.2f}), applying safety default")
            response.decision = ReplanDecision.COMPLETE
            response.progress_estimate = 100
            response.safety_notes += " | LOW CONFIDENCE: Defaulted to completion for safety"
        
        # If too many iterations, force completion
        if pipeline_state.current_iteration >= self.max_iterations - 1:
            print_t(f"[SERC] Near iteration limit, forcing completion")
            response.decision = ReplanDecision.COMPLETE
            response.progress_estimate = 100
            response.safety_notes += " | ITERATION LIMIT: Forced completion"
        
        return response

    def _create_safety_fallback_response(self, pipeline_state: PipelineState, error_msg: str) -> ReplanResponse:
        """
        Create a safe fallback response when assessment fails
        Always defaults to completion for safety
        """
        return ReplanResponse(
            decision=ReplanDecision.COMPLETE,
            confidence=0.2,
            progress_estimate=100,  # Mark as complete to stop execution
            reasoning=f"Safety fallback due to assessment error: {error_msg}",
            next_guidance=None,
            safety_notes=f"SAFETY FALLBACK: {error_msg}",
            execution_limit_reached=False
        )

    def start_new_task(self, task_description: str):
        """Initialize state for a new task"""
        self.current_task_start_time = time.time()
        self.consecutive_failures = 0
        self.iteration_history = []
        print_t(f"[SERC] Started new task: {task_description}")

    def configure_safety_limits(self, max_iterations: int = None, max_time: float = None, 
                               max_failures: int = None):
        """Configure safety limits"""
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if max_time is not None:
            self.max_execution_time = max_time
        if max_failures is not None:
            self.max_planning_failures = max_failures
            
        print_t(f"[SERC] Safety limits updated - iterations:{self.max_iterations}, "
                f"time:{self.max_execution_time}s, failures:{self.max_planning_failures}")

    def set_replan_model(self, model_name: str, temperature: float = None):
        """Configure the replan model"""
        self.replan_model = model_name
        if temperature is not None:
            self.replan_temperature = temperature
        print_t(f"[SERC] Replan model updated: {model_name} (temp: {self.replan_temperature})")