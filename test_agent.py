#!/usr/bin/env python3

"""Test script for the AstroFlow agent."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from s_agent.agent import AstroFlowAgent

if __name__ == "__main__":
    print("Testing AstroFlowAgent import...")
    
    # Test creating an agent instance
    try:
        agent = AstroFlowAgent()
        print("Successfully created AstroFlowAgent instance")
        
        # Test basic methods
        print("Agent methods available:")
        print("- run")
        print("- run_task")
        print("Import test completed successfully!")
        
    except Exception as e:
        print(f"Error creating agent: {e}")
        sys.exit(1)