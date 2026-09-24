"""
Planner component for the Astro-Flow-3D agent.
Responsible for decomposing high-level tasks into structured work units.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .types import TaskRequest
from .schemas import validate_task_request


class Planner:
    """Base planner class that can be extended with LLM or rule-based approaches."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load planner configuration from file if available."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
    
    def generate_plan(self, task_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a plan for the given task context.
        
        Args:
            task_context: Dictionary containing task details
            
        Returns:
            List of subtasks to execute
        """
        # Default rule-based planner - can be overridden with LLM implementation
        return self._rule_based_plan(task_context)
    
    def _rule_based_plan(self, task_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Default rule-based planning logic."""
        objective = task_context.get('objective', '')
        
        # Simple rule-based decomposition
        if 'build dataset' in objective.lower():
            return [
                {
                    'task_id': 'dataset_build_1',
                    'objective': 'Load FITS file',
                    'command': 'load_fits',
                    'inputs': {'fits_path': task_context.get('input_fits')}
                },
                {
                    'task_id': 'dataset_build_2', 
                    'objective': 'Normalize image',
                    'command': 'normalize',
                    'inputs': {'image': 'loaded_image', 'lower_percentile': 1.0, 'upper_percentile': 99.8}
                },
                {
                    'task_id': 'dataset_build_3',
                    'objective': 'Generate tiles',
                    'command': 'generate_tiles',
                    'inputs': {'image': 'normalized_image', 'tile_size': 256, 'stride': 256}
                }
            ]
        elif 'normalize' in objective.lower():
            return [
                {
                    'task_id': 'normalize_1',
                    'objective': 'Apply normalization',
                    'command': 'normalize',
                    'inputs': task_context.get('inputs', {})
                }
            ]
        else:
            # Default fallback - just one task
            return [
                {
                    'task_id': 'default_task',
                    'objective': objective,
                    'command': 'run_pipeline',
                    'inputs': task_context.get('inputs', {})
                }
            ]
    
    def summarize_result(self, run_result: Dict[str, Any]) -> str:
        """Summarize the result of a run."""
        return f"Task completed with exit code {run_result.get('exit_code', 0)}"


class LLMPlanner(Planner):
    """LLM-based planner that can use local models like qwen-code-3.5-30b."""
    
    def __init__(self, config_path: str = None, model_name: str = "qwen-code-3.5-30b"):
        super().__init__(config_path)
        self.model_name = model_name
        self._model_available = self._check_model_availability()
    
    def _check_model_availability(self) -> bool:
        """Check if the local LLM is available."""
        # This would check for model availability in a real implementation
        # For now, we'll assume it's available but can be disabled
        return True
    
    def generate_plan(self, task_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate plan using LLM if available, otherwise fall back to rule-based."""
        if self._model_available:
            # In a real implementation, this would call the local LLM
            # For now, we'll use the rule-based approach but with LLM prompt formatting
            return self._llm_plan(task_context)
        else:
            return super().generate_plan(task_context)
    
    def _llm_plan(self, task_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate plan using local LLM."""
        # This would be replaced with actual LLM call in a real implementation
        # For now, we'll return the rule-based plan but formatted for LLM consumption
        
        prompt = f"""
        You are an expert agent planner for astronomical data processing.
        
        Task context: {json.dumps(task_context, indent=2)}
        
        Please decompose this task into concrete subtasks that can be executed by a local system.
        Each subtask should include:
        - task_id: unique identifier
        - objective: clear description of what needs to be done
        - command: the command or function to execute
        - inputs: required input parameters
        
        Return a JSON list of these subtasks.
        """
        
        # In a real implementation, this would call the LLM with the prompt
        # For now, we'll return the rule-based plan as a fallback
        return self._rule_based_plan(task_context)


def create_planner(planner_type: str = "rule_based", config_path: str = None) -> Planner:
    """Factory function to create a planner of specified type."""
    if planner_type == "llm":
        return LLMPlanner(config_path=config_path)
    else:
        return Planner(config_path=config_path)
