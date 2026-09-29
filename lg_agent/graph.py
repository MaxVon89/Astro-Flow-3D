"""Main execution graph for Astro-Flow-3D agent system."""

from typing import Dict, Any, List, Optional
from .schemas.task import Task
from .planner.planner import Planner
from .executor.executor import Executor
from .validator.validator import Validator
from .memory.memory import Memory

class AstroFlowMultiAgent:
    """LangGraph-inspired multi-agent for astronomy research workflows."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.planner = Planner()
        self.executor = Executor()
        self.validator = Validator()
        self.memory = Memory()
        
    def run(self, task: Task) -> Dict[str, Any]:
        """Execute a single task through the agent loop."""
        # Planning phase
        plan = self.planner.plan(task)
        
        # Execution phase
        result = self.executor.execute(plan)
        
        # Validation phase
        validation_result = self.validator.validate(result)
        
        # Memory phase
        self.memory.save_results(task, result, validation_result)
        
        return {
            "task": task,
            "plan": plan,
            "result": result,
            "validation": validation_result,
            "memory": self.memory.get_session_data()
        }
        
    def run_batch(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Execute multiple tasks in sequence."""
        results = []
        for task in tasks:
            results.append(self.run(task))
        return results
        
    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get the last execution result."""
        return self.memory.get_last_result()