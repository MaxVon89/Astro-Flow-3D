"""Registry of allowed tools for the Astro-Flow-3D agent system."""

import importlib
from typing import Dict, Any, Callable
from pathlib import Path

# Import actual tool functions from src/data modules
from src.data.pipeline import run_pipeline
from src.data.preprocess import normalize
from src.data.tile_generator import generate_tiles
from src.data.dataset_builder import build_dataset

# Registry mapping tool names to their implementations
TOOL_REGISTRY: Dict[str, Callable] = {
    "run_pipeline": run_pipeline,
    "normalize": normalize,
    "generate_tiles": generate_tiles,
    "build_dataset": build_dataset
}

def execute_registered_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool from the registry with provided arguments.
    
    Parameters
    ----------
    tool_name : str
        Name of the tool to execute
        
    args : dict
        Arguments to pass to the tool
        
    Returns
    -------
    dict
        Execution result with artifacts and metrics
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{tool_name}' is not registered")
        
    # Get the tool function from registry
    tool_func = TOOL_REGISTRY[tool_name]
    
    # Execute the tool
    try:
        result = tool_func(**args)
        
        # Prepare result for agent consumption
        return {
            "task_id": args.get("task_id", "unknown"),
            "tool": tool_name,
            "status": "success",
            "artifacts": [],
            "metrics": {},
            "result": result
        }
    except Exception as e:
        raise RuntimeError(f"Tool '{tool_name}' execution failed: {str(e)}")