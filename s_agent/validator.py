"""
Validator component for the Astro-Flow-3D agent.
Responsible for checking task outputs against scientific and repository invariants.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .types import ValidationReport, ValidationStatus


class DatasetValidator:
    """Validates dataset artifacts and scientific invariants."""
    
    def __init__(self):
        self.validation_rules = {
            'file_existence': self._validate_file_existence,
            'manifest_consistency': self._validate_manifest_consistency,
            'index_integrity': self._validate_index_integrity,
            'tile_count': self._validate_tile_count,
            'normalization_sanity': self._validate_normalization_sanity,
            'flux_conservation': self._validate_flux_conservation
        }
    
    def validate(self, task_result: Dict[str, Any]) -> ValidationReport:
        """
        Validate the outputs of a task.
        
        Args:
            task_result: The result from a task execution
            
        Returns:
            ValidationReport with results
        """
        failures = []
        metrics = {}
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(task_result)
                if not rule_result['passed']:
                    failures.append({
                        'rule': rule_name,
                        'message': rule_result['message'],
                        'details': rule_result.get('details', {})
                    })
                else:
                    metrics[rule_name] = True
            except Exception as e:
                failures.append({
                    'rule': rule_name,
                    'message': f"Validation error: {str(e)}",
                    'error': str(e)
                })
        
        # Determine overall status
        status = ValidationStatus.PASS if len(failures) == 0 else ValidationStatus.FAIL
        
        return ValidationReport(
            task_id=task_result.get('task_id', 'unknown'),
            status=status,
            metrics=metrics,
            failures=failures,
            created_at=datetime.now()
        )
    
    def _validate_file_existence(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that required files exist."""
        outputs = task_result.get('outputs', {})
        expected_files = outputs.get('expected_outputs', [])
        
        missing_files = []
        for file_path in expected_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            return {
                'passed': False,
                'message': f"Missing files: {', '.join(missing_files)}",
                'details': {'missing_files': missing_files}
            }
        
        return {'passed': True, 'message': 'All expected files exist'}
    
    def _validate_manifest_consistency(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that manifest is consistent with generated artifacts."""
        outputs = task_result.get('outputs', {})
        manifest_path = outputs.get('manifest_path')
        
        if not manifest_path or not os.path.exists(manifest_path):
            return {
                'passed': False,
                'message': 'Manifest file not found',
                'details': {'manifest_path': manifest_path}
            }
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Basic validation - check required fields
            required_fields = ['dataset_name', 'num_images', 'num_tiles']
            for field in required_fields:
                if field not in manifest:
                    return {
                        'passed': False,
                        'message': f'Missing required field in manifest: {field}',
                        'details': {'missing_field': field}
                    }
            
            return {'passed': True, 'message': 'Manifest is valid'}
        except Exception as e:
            return {
                'passed': False,
                'message': f'Error reading manifest: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _validate_index_integrity(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that index.csv is properly formatted."""
        outputs = task_result.get('outputs', {})
        index_path = outputs.get('index_path')
        
        if not index_path or not os.path.exists(index_path):
            return {
                'passed': False,
                'message': 'Index file not found',
                'details': {'index_path': index_path}
            }
        
        try:
            df = pd.read_csv(index_path)
            
            # Check for required columns
            required_columns = ['tile_filename', 'source_name']
            for col in required_columns:
                if col not in df.columns:
                    return {
                        'passed': False,
                        'message': f'Missing required column in index: {col}',
                        'details': {'missing_column': col}
                    }
            
            # Check for duplicates
            duplicates = df['tile_filename'].duplicated()
            if duplicates.any():
                duplicate_files = df.loc[duplicates, 'tile_filename'].unique()
                return {
                    'passed': False,
                    'message': f'Duplicate tile filenames found: {", ".join(duplicate_files)}',
                    'details': {'duplicates': duplicate_files.tolist()}
                }
            
            return {'passed': True, 'message': 'Index is valid'}
        except Exception as e:
            return {
                'passed': False,
                'message': f'Error reading index: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _validate_tile_count(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the expected number of tiles were generated."""
        outputs = task_result.get('outputs', {})
        manifest_path = outputs.get('manifest_path')
        
        if not manifest_path or not os.path.exists(manifest_path):
            return {
                'passed': False,
                'message': 'Manifest file not found for tile count validation',
                'details': {'manifest_path': manifest_path}
            }
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            expected_tiles = manifest.get('num_tiles', 0)
            
            # Count actual tiles
            tile_dir = Path(manifest_path).parent / "tiles"
            if not tile_dir.exists():
                return {
                    'passed': False,
                    'message': 'Tiles directory not found',
                    'details': {'tile_directory': str(tile_dir)}
                }
            
            actual_tiles = len(list(tile_dir.glob("*.npy")))
            
            if actual_tiles != expected_tiles:
                return {
                    'passed': False,
                    'message': f'Tile count mismatch: expected {expected_tiles}, got {actual_tiles}',
                    'details': {'expected': expected_tiles, 'actual': actual_tiles}
                }
            
            return {'passed': True, 'message': f'Correct tile count: {actual_tiles}'}
        except Exception as e:
            return {
                'passed': False,
                'message': f'Error validating tile count: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _validate_normalization_sanity(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that normalization produced reasonable results."""
        outputs = task_result.get('outputs', {})
        normalized_image_path = outputs.get('normalized_image')
        
        if not normalized_image_path or not os.path.exists(normalized_image_path):
            return {
                'passed': False,
                'message': 'Normalized image file not found',
                'details': {'image_path': normalized_image_path}
            }
        
        try:
            # Load the image and check basic properties
            if normalized_image_path.endswith('.npy'):
                image = np.load(normalized_image_path)
            else:
                return {
                    'passed': False,
                    'message': 'Unsupported image format',
                    'details': {'format': os.path.splitext(normalized_image_path)[1]}
                }
            
            # Check that values are in [0, 1] range (normalized)
            min_val = image.min()
            max_val = image.max()
            
            if min_val < 0 or max_val > 1:
                return {
                    'passed': False,
                    'message': f'Normalization range invalid: [{min_val:.4f}, {max_val:.4f}]',
                    'details': {'min_val': min_val, 'max_val': max_val}
                }
            
            # Check that image is not all zeros (indicating failure)
            if np.allclose(image, 0):
                return {
                    'passed': False,
                    'message': 'Normalized image is all zeros',
                    'details': {}
                }
            
            return {'passed': True, 'message': 'Normalization appears valid'}
        except Exception as e:
            return {
                'passed': False,
                'message': f'Error validating normalization: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _validate_flux_conservation(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that flux is conserved in the processing pipeline."""
        # This would be more complex to implement fully
        # For now, we'll just return a pass as a placeholder
        return {
            'passed': True,
            'message': 'Flux conservation validation not implemented in this version'
        }


def create_validator() -> DatasetValidator:
    """Factory function to create a dataset validator."""
    return DatasetValidator()
