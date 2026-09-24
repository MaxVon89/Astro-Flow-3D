"""
Memory system for the Astro-Flow-3D agent.
Stores task history, artifacts, and provenance information.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import sqlite3

from .types import TaskResult, ArtifactRecord, RunManifest


class MemoryWriter:
    """Writes agent state to persistent storage."""
    
    def __init__(self, storage_path: str = "./agent_memory.db"):
        self.storage_path = Path(storage_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database for storing agent state."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        # Create tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                parent_task_id TEXT,
                objective TEXT,
                inputs TEXT,
                expected_outputs TEXT,
                created_at TEXT,
                artifact_signature TEXT,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Create task_results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_results (
                task_id TEXT PRIMARY KEY,
                status TEXT,
                outputs TEXT,
                stdout TEXT,
                stderr TEXT,
                exit_code INTEGER,
                runtime_seconds REAL,
                command_hash TEXT,
                validation_report TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create artifacts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                task_id TEXT,
                path TEXT,
                hash TEXT,
                size_bytes INTEGER,
                created_at TEXT,
                metadata TEXT
            )
        ''')
        
        # Create run_manifests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS run_manifests (
                run_id TEXT PRIMARY KEY,
                task_id TEXT,
                created_at TEXT,
                git_sha TEXT,
                environment_snapshot TEXT,
                config_snapshot TEXT,
                command TEXT,
                result_path TEXT,
                validation_report_path TEXT,
                artifacts TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_task(self, task_request):
        """Save a new task request."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO tasks 
            (task_id, parent_task_id, objective, inputs, expected_outputs, created_at, artifact_signature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_request.task_id,
            getattr(task_request, 'parent_task_id', None),
            task_request.objective,
            json.dumps(task_request.inputs),
            json.dumps(task_request.expected_outputs),
            task_request.created_at.isoformat(),
            getattr(task_request, 'artifact_signature', None)
        ))
        
        conn.commit()
        conn.close()
    
    def update_task_status(self, task_id: str, status: str):
        """Update the status of a task."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE tasks SET status = ? WHERE task_id = ?
        ''', (status, task_id))
        
        conn.commit()
        conn.close()
    
    def save_task_result(self, task_result: TaskResult):
        """Save the result of a task execution."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO task_results 
            (task_id, status, outputs, stdout, stderr, exit_code, runtime_seconds, command_hash, validation_report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_result.task_id,
            task_result.status.value,
            json.dumps(task_result.outputs),
            task_result.stdout,
            task_result.stderr,
            task_result.exit_code,
            task_result.runtime_seconds,
            getattr(task_result, 'command_hash', None),
            json.dumps(task_result.validation_report.to_dict()) if task_result.validation_report else None
        ))
        
        conn.commit()
        conn.close()
    
    def save_artifact(self, artifact_record: ArtifactRecord):
        """Save an artifact record."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO artifacts 
            (artifact_id, task_id, path, hash, size_bytes, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            artifact_record.artifact_id,
            artifact_record.task_id,
            artifact_record.path,
            artifact_record.hash,
            artifact_record.size_bytes,
            artifact_record.created_at.isoformat(),
            json.dumps(artifact_record.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def save_run_manifest(self, run_manifest: RunManifest):
        """Save a run manifest."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO run_manifests 
            (run_id, task_id, created_at, git_sha, environment_snapshot, config_snapshot, command, result_path, validation_report_path, artifacts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_manifest.run_id,
            run_manifest.task_id,
            run_manifest.created_at.isoformat(),
            run_manifest.git_sha,
            json.dumps(run_manifest.environment_snapshot),
            json.dumps(run_manifest.config_snapshot),
            run_manifest.command,
            run_manifest.result_path,
            run_manifest.validation_report_path,
            json.dumps([art.to_dict() for art in run_manifest.artifacts])
        ))
        
        conn.commit()
        conn.close()
    
    def get_task(self, task_id: str):
        """Retrieve a task by ID."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return {
                'task_id': row[0],
                'parent_task_id': row[1],
                'objective': row[2],
                'inputs': json.loads(row[3]),
                'expected_outputs': json.loads(row[4]),
                'created_at': datetime.fromisoformat(row[5]),
                'artifact_signature': row[6],
                'status': row[7]
            }
        return None
    
    def get_task_result(self, task_id: str):
        """Retrieve a task result by ID."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM task_results WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return {
                'task_id': row[0],
                'status': row[1],
                'outputs': json.loads(row[2]),
                'stdout': row[3],
                'stderr': row[4],
                'exit_code': row[5],
                'runtime_seconds': row[6],
                'command_hash': row[7],
                'validation_report': json.loads(row[8]) if row[8] else None,
                'created_at': datetime.fromisoformat(row[9]) if row[9] else None
            }
        return None
    
    def get_artifacts_for_task(self, task_id: str):
        """Retrieve all artifacts for a specific task."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM artifacts WHERE task_id = ?', (task_id,))
        rows = cursor.fetchall()
        
        conn.close()
        
        artifacts = []
        for row in rows:
            artifacts.append({
                'artifact_id': row[0],
                'task_id': row[1],
                'path': row[2],
                'hash': row[3],
                'size_bytes': row[4],
                'created_at': datetime.fromisoformat(row[5]),
                'metadata': json.loads(row[6])
            })
        
        return artifacts
