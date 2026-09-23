#!/usr/bin/env python3
"""
Simple script to run the Astro-Flow-3D agent from the project root.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from s_agent.agent import AstroFlowAgent
    
    # Create and run a simple test
    agent = AstroFlowAgent()
    print("Astro-Flow-3D Agent initialized successfully!")
    
except Exception as e:
    print(f"Error initializing agent: {e}")
    sys.exit(1)