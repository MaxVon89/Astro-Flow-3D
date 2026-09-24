"""
Executor component for the Astro-Flow-3D agent.
Responsible for running commands in a safe, controlled environment.
"""

import subprocess
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

from .types import TaskResult


class SafeExecutor:
    """Safely executes commands with controlled access and validation."""
    
    def __init__(self, allowlist: Optional[Dict[str, Any]] = None, dry_run: bool = False):
        self.allowlist = allowlist or self._default_allowlist()
        self.dry_run = dry_run
        self.executed_commands = []
    
    def _default_allowlist(self) -> Dict[str, Any]:
        """Define a default set of allowed commands and patterns."""
        return {
            "allowed_patterns": [
                r"python.*src/data/pipeline\.py",
                r"python.*src/data/dataset_builder\.py", 
                r"python.*src/data/preprocess\.py",
                r"python.*src/data/tile_generator\.py",
                r"ls.*",
                r"mkdir.*",
                r"cp.*",
                r"rm.*",
                r"cat.*",
                r"echo.*"
            ],
            "allowed_commands": [
                "python", 
                "ls", 
                "mkdir", 
                "cp", 
                "rm", 
                "cat", 
                "echo"
            ]
        }
    
    def execute(self, command: str, cwd: str = ".", timeout: int = 300) -> TaskResult:
        """
        Execute a command safely.
        
        Args:
            command: The shell command to execute
            cwd: Working directory for the command
            timeout: Timeout in seconds
            
        Returns:
            TaskResult with execution details
        """
        # Validate command against allowlist
        if not self._is_command_allowed(command):
            return TaskResult(
                task_id="invalid_command",
                status="failed",
                outputs={},
                stdout="",
                stderr=f"Command '{command}' is not in the allowed list",
                exit_code=1,
                runtime_seconds=0.0
            )
        
        # Record execution
        self.executed_commands.append({
            "command": command,
            "cwd": cwd,
            "timestamp": datetime.now().isoformat()
        })
        
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {command}")
            return TaskResult(
                task_id="dry_run",
                status="passed",
                outputs={"dry_run": True},
                stdout=f"Would execute: {command}",
                stderr="",
                exit_code=0,
                runtime_seconds=0.0
            )
        
        # Execute the command
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            end_time = datetime.now()
            runtime_seconds = (end_time - start_time).total_seconds()
            
            # Create a hash of the command for tracking
            command_hash = hashlib.sha256(command.encode()).hexdigest()[:16]
            
            return TaskResult(
                task_id="executed_command",
                status="passed" if result.returncode == 0 else "failed",
                outputs={"command": command},
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                runtime_seconds=runtime_seconds,
                command_hash=command_hash
            )
            
        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            runtime_seconds = (end_time - start_time).total_seconds()
            
            return TaskResult(
                task_id="timeout",
                status="failed",
                outputs={},
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                exit_code=1,
                runtime_seconds=runtime_seconds
            )
        except Exception as e:
            end_time = datetime.now()
            runtime_seconds = (end_time - start_time).total_seconds()
            
            return TaskResult(
                task_id="error",
                status="failed",
                outputs={},
                stdout="",
                stderr=str(e),
                exit_code=1,
                runtime_seconds=runtime_seconds
            )
    
    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command is allowed based on the allowlist."""
        # Check for exact matches first
        if command.strip() in self.allowlist.get("allowed_commands", []):
            return True
        
        # Check against patterns
        import re
        for pattern in self.allowlist.get("allowed_patterns", []):
            if re.search(pattern, command):
                return True
        
        # Allow commands that start with allowed prefixes
        allowed_prefixes = [cmd + " " for cmd in self.allowlist.get("allowed_commands", [])]
        for prefix in allowed_prefixes:
            if command.startswith(prefix):
                return True
        
        return False
    
    def get_executed_commands(self) -> list:
        """Return the list of executed commands."""
        return self.executed_commands


def create_executor(dry_run: bool = False) -> SafeExecutor:
    """Factory function to create a safe executor."""
    return SafeExecutor(dry_run=dry_run)
