"""Memory management for Astro-Flow-3D agent system."""

import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

class Memory:
    """Local memory storage for agent sessions."""
    
    def __init__(self):
        self.session_data: List[Dict[str, Any]] = []
        self.session_id = None
        
    def save_results(self, task: Dict[str, Any], result: Dict[str, Any], 
                     validation_result: Dict[str, Any]) -> None:
        """Save the results of a single execution to memory."""
        session_entry = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": result,
            "validation": validation_result
        }
        self.session_data.append(session_entry)
        
    def get_session_data(self) -> List[Dict[str, Any]]:
        """Return all session data."""
        return self.session_data
        
    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent execution result."""
        if not self.session_data:
            return None
        return self.session_data[-1]
        
    def clear_session(self) -> None:
        """Clear all session data."""
        self.session_data.clear()