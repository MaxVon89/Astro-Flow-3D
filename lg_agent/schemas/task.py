"""Task schema for Astro-Flow-3D agent system."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

@dataclass
class Task:
    """A single task to be executed by the agent."""
    
    id: str
    name: str
    description: str
    objective: str
    config: Optional[Dict[str, Any]] = None
    dependencies: Optional[list] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "objective": self.objective,
            "config": self.config,
            "dependencies": self.dependencies
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            objective=data["objective"],
            config=data.get("config"),
            dependencies=data.get("dependencies")
        )