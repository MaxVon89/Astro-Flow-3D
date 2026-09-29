#!/usr/bin/env python3
"""
Web server for Astro-Flow-3D agent system.
Handles task execution according to Day1 and Day2 objectives.
"""

import time
from flask import Flask, request, jsonify
import json
import os
from lg_agent import AstroFlowMultiAgent
from lg_agent.schemas.task import Task

app = Flask(__name__)
agent = AstroFlowMultiAgent()

def setup_logging():
    """Set up logging for the agent system"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

setup_logging()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'agent': 'AstroFlowMultiAgent',
        'version': '1.0.0'
    })

@app.route('/run_task', methods=['POST'])
def run_task():
    """Execute a single task through the agent system"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        # Create task from request
        task = Task(
            id=data.get('id', f'task_{int(time.time())}'),
            name=data.get('name', ''),
            description=data.get('description', ''),
            objective=data.get('objective', ''),
            config=data.get('config', {}),
            dependencies=data.get('dependencies', [])
        )
        
        # Execute the task through the agent
        result = agent.run(task)
        
        return jsonify({
            'status': 'success',
            'task_id': task.id,
            'result': result
        })
        
    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg}), 500

@app.route('/run_preprocessing', methods=['POST'])
def run_preprocessing():
    """Execute preprocessing pipeline tasks"""
    try:
        # Create preprocessing task
        task = Task(
            id=f'preproc_{int(time.time())}',
            name='JWST Data Preprocessing',
            description='Preprocess JWST FITS data for MBA model training',
            objective='preprocess JWST data for 3D reconstruction'
        )

        result = agent.run(task)

        return jsonify({
            'status': 'success',
            'task_id': task.id,
            'result': result
        })

    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg}), 500

@app.route('/build_dataset', methods=['POST'])
def build_dataset():
    """Execute dataset building tasks"""
    try:
        # Create dataset building task
        task = Task(
            id=f'dataset_{int(time.time())}',
            name='Dataset Building',
            description='Build dataset for MBA model training from processed data',
            objective='build dataset for MBA reconstruction model'
        )

        result = agent.run(task)

        return jsonify({
            'status': 'success',
            'task_id': task.id,
            'result': result
        })

    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg}), 500

@app.route('/validate_mba', methods=['POST'])
def validate_mba():
    """Execute MBA validation tasks"""
    try:
        # Create MBA validation task
        task = Task(
            id=f'mba_valid_{int(time.time())}',
            name='MBA Validation',
            description='Validate MBA reconstruction quality against expected standards',
            objective='validate MBA reconstruction quality using correlation metrics'
        )

        result = agent.run(task)

        return jsonify({
            'status': 'success',
            'task_id': task.id,
            'result': result
        })

    except Exception as e:
        error_msg = str(e)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    print("Starting Astro-Flow-3D Agent Web Server...")
    print("Endpoints available:")
    print("- GET   /health (Health check)")
    print("- POST  /run_task (Execute any task)")
    print("- POST  /run_preprocessing (Execute preprocessing)")
    print("- POST  /build_dataset (Build dataset)")
    print("- POST  /validate_mba (Validate MBA model)")
    print("\nServer starting on port 8080...")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)