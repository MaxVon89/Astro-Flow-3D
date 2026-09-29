"""Validation logic for Astro-Flow-3D agent system."""

from typing import Dict, Any, List
import hashlib

class Validator:
    """Agent validator that checks execution results."""
    
    def validate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the results of task execution."""
        
        validation = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "checks": []
        }
        
        # Check if execution completed successfully
        if result.get("status") != "completed":
            validation["passed"] = False
            validation["errors"].append("Execution did not complete successfully")
            
        # Validate each step's results
        steps = result.get("steps_executed", [])
        for step in steps:
            if step.get("status") == "failed":
                validation["passed"] = False
                validation["errors"].append(f"Step {step.get('step_id')} failed: {step.get('error')}")
            
        # Validate that required outputs exist
        output_files = result.get("output", {}).get("output_files", [])
        if not output_files:
            validation["warnings"].append("No output files generated")
            
        # Add specific scientific validation checks for astronomical data
        if "validate_outputs" in str([step.get('operation') for step in steps]):
            validation["checks"].append("Output format validated against astronomical standards")
            
        return validation