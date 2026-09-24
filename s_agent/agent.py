"""
Main agent orchestration component for the Astro-Flow-3D agent.
Coordinates between planner, executor, validator, and memory components.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add the project root to Python path to resolve imports properly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from s_agent.planner import create_planner
from s_agent.executor import create_executor
from s_agent.validator import create_validator
from s_agent.memory import MemoryWriter
from s_agent.types import TaskRequest, TaskResult, ValidationReport


class AstroFlowAgent:
    """Main agent class that orchestrates the research-to-code workflow."""
    
    def __init__(self, 
                 config_path: str = None,
                 dry_run: bool = False,
                 memory_storage: str = "./agent_memory.db"):
        """
        Initialize the Astro-Flow agent.
        
        Args:
            config_path: Path to configuration file
            dry_run: If True, execute commands in dry-run mode
            memory_storage: Path to memory storage database
        """
        self.config_path = config_path
        self.dry_run = dry_run
        self.memory = MemoryWriter(memory_storage)
        
        # Initialize components
        self.planner = create_planner("rule_based", config_path)
        self.executor = create_executor(dry_run=dry_run)
        self.validator = create_validator()
        
        # Track current run
        self.current_run_id = None
    
    def start_run(self, run_id: str):
        """Start a new run with the given ID."""
        self.current_run_id = run_id
    
    def process_task(self, task_request: TaskRequest) -> TaskResult:
        """
        Process a single task through the agent loop.
        
        Args:
            task_request: The task to execute
            
        Returns:
            TaskResult with execution details
        """
        # Save the initial task request
        self.memory.save_task(task_request)
        
        # Generate plan for the task
        print(f"Planning task: {task_request.objective}")
        plan = self.planner.generate_plan({
            'objective': task_request.objective,
            'inputs': task_request.inputs
        })
        
        # Execute each step in the plan
        final_result = None
        all_outputs = {}
        
        for i, subtask in enumerate(plan):
            print(f"Executing subtask {i+1}: {subtask.get('objective', 'Unknown')}")
            
            # Execute the command
            result = self.executor.execute(
                command=subtask.get('command', ''),
                cwd=os.getcwd()
            )
            
            # Update outputs with results from this step
            all_outputs.update(result.outputs)
            
            if result.status == "failed":
                print(f"Subtask failed: {result.stderr}")
                final_result = result
                break
            
            # For demonstration, we'll just use the last successful result
            final_result = result
        
        if final_result is None:
            final_result = TaskResult(
                task_id=task_request.task_id,
                status="failed",
                outputs={},
                stdout="",
                stderr="No tasks executed",
                exit_code=1,
                runtime_seconds=0.0
            )
        
        # Add the final result to the task
        final_result.task_id = task_request.task_id
        
        # Validate the result
        print("Validating results...")
        validation_report = self.validator.validate({
            'task_id': task_request.task_id,
            'outputs': all_outputs,
            'exit_code': final_result.exit_code,
            'stdout': final_result.stdout,
            'stderr': final_result.stderr
        })
        
        # Update the result with validation report
        final_result.validation_report = validation_report
        
        # Save the result
        self.memory.save_task_result(final_result)
        
        # Update task status in memory
        self.memory.update_task_status(task_request.task_id, final_result.status.value)
        
        return final_result
    
    def run_pipeline(self, task_requests: List[TaskRequest]) -> List[TaskResult]:
        """
        Run a sequence of tasks through the agent.
        
        Args:
            task_requests: List of tasks to execute
            
        Returns:
            List of task results
        """
        results = []
        
        for task_request in task_requests:
            print(f"\n--- Processing Task: {task_request.objective} ---")
            result = self.process_task(task_request)
            results.append(result)
            
            if result.status == "failed":
                print(f"Task failed: {result.stderr}")
                break
        
        return results
    
    def get_run_summary(self, task_id: str) -> Dict[str, Any]:
        """
        Get a summary of a run.
        
        Args:
            task_id: The task ID to summarize
            
        Returns:
            Dictionary with run summary
        """
        task = self.memory.get_task(task_id)
        if not task:
            return {"error": "Task not found"}
        
        result = self.memory.get_task_result(task_id)
        artifacts = self.memory.get_artifacts_for_task(task_id)
        
        return {
            "task": task,
            "result": result,
            "artifacts": artifacts
        }


def create_agent(config_path: str = None, dry_run: bool = False) -> AstroFlowAgent:
    """Factory function to create an agent."""
    return AstroFlowAgent(config_path=config_path, dry_run=dry_run)


# Example usage
if __name__ == "__main__":
    # Create a simple example task
    from .types import TaskRequest
    
    agent = create_agent()
    
    # Create a sample task
    task = TaskRequest(
        task_id="test_task_1",
        objective="Build a small JWST dataset",
        inputs={"input_fits": "/path/to/test.fits"},
        expected_outputs=["output/index.csv", "output/manifest.json"],
        created_at=datetime.now()
    )
    
    # Process the task
    result = agent.process_task(task)
    
    print(f"Task completed with status: {result.status}")
    if result.validation_report:
        print(f"Validation status: {result.validation_report.status}")