#!/usr/bin/env python3
"""
Model Loader for BERT Asset/Deficit Classification Web App.

This script handles loading and copying the best trained model from the 
bert_finetuning directory to the webapp's local models directory.
"""

import os
import shutil
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles model loading and management for the webapp."""
    
    def __init__(self, 
                 source_models_dir="/N/slate/ankichau/projects/center/bert_finetuning/models",
                 webapp_models_dir="/N/slate/ankichau/projects/center/bert_webapp/models"):
        self.source_models_dir = Path(source_models_dir)
        self.webapp_models_dir = Path(webapp_models_dir)
        
        # Ensure the webapp models directory exists
        self.webapp_models_dir.mkdir(parents=True, exist_ok=True)
    
    def find_best_model(self):
        """
        Find the best model in the source directory.
        
        Returns:
            Path: Path to the best model directory, or None if not found
        """
        best_model_path = self.source_models_dir / "best_model"
        final_model_path = self.source_models_dir / "final_model"
        
        # Prefer best_model if it exists
        if best_model_path.exists() and best_model_path.is_dir():
            logger.info(f"Found best_model at: {best_model_path}")
            return best_model_path
        elif final_model_path.exists() and final_model_path.is_dir():
            logger.info(f"Found final_model at: {final_model_path}")
            return final_model_path
        else:
            logger.error("No suitable model found in source directory")
            return None
    
    def get_model_info(self, model_path):
        """
        Extract information about the model from its files.
        
        Args:
            model_path (Path): Path to the model directory
            
        Returns:
            dict: Model information
        """
        info = {
            "model_path": str(model_path),
            "model_name": model_path.name,
            "timestamp": datetime.now().isoformat(),
            "files": []
        }
        
        # List all files in the model directory
        if model_path.exists():
            info["files"] = [f.name for f in model_path.iterdir() if f.is_file()]
            
            # Try to read training stats if available
            stats_file = model_path / "training_stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        info["training_stats"] = stats
                except Exception as e:
                    logger.warning(f"Could not read training stats: {e}")
                    
            # Try to read final stats if available
            final_stats_file = model_path / "final_stats.json"
            if final_stats_file.exists():
                try:
                    with open(final_stats_file, 'r') as f:
                        final_stats = json.load(f)
                        info["final_stats"] = final_stats
                except Exception as e:
                    logger.warning(f"Could not read final stats: {e}")
        
        return info
    
    def copy_model_to_webapp(self, force_update=False):
        """
        Copy the best model from source to webapp directory.
        
        Args:
            force_update (bool): If True, overwrite existing model
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Find the best model
        source_model = self.find_best_model()
        if not source_model:
            return False
        
        # Define target path
        target_model = self.webapp_models_dir / "best_model"
        
        # Check if model already exists
        if target_model.exists() and not force_update:
            logger.info(f"Model already exists at {target_model}. Use force_update=True to overwrite.")
            return True
        
        try:
            # Remove existing model if force_update
            if target_model.exists() and force_update:
                logger.info(f"Removing existing model at {target_model}")
                shutil.rmtree(target_model)
            
            # Copy the model
            logger.info(f"Copying model from {source_model} to {target_model}")
            shutil.copytree(source_model, target_model)
            
            # Get and log model info
            model_info = self.get_model_info(target_model)
            logger.info(f"Successfully copied model: {model_info['model_name']}")
            logger.info(f"Model files: {', '.join(model_info['files'])}")
            
            # Save model info to webapp directory
            info_file = self.webapp_models_dir / "model_info.json"
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error copying model: {e}")
            return False
    
    def check_model_status(self):
        """
        Check the status of models in both source and webapp directories.
        
        Returns:
            dict: Status information
        """
        status = {
            "source_available": False,
            "webapp_available": False,
            "source_info": None,
            "webapp_info": None,
            "needs_update": False
        }
        
        # Check source model
        source_model = self.find_best_model()
        if source_model:
            status["source_available"] = True
            status["source_info"] = self.get_model_info(source_model)
        
        # Check webapp model
        webapp_model = self.webapp_models_dir / "best_model"
        if webapp_model.exists():
            status["webapp_available"] = True
            status["webapp_info"] = self.get_model_info(webapp_model)
        
        # Determine if update is needed (simple timestamp comparison)
        if status["source_available"] and status["webapp_available"]:
            # You could implement more sophisticated comparison here
            # For now, just check if files exist
            status["needs_update"] = False
        elif status["source_available"] and not status["webapp_available"]:
            status["needs_update"] = True
        
        return status
    
    def get_webapp_model_path(self):
        """
        Get the path to the webapp's local model.
        
        Returns:
            str: Path to the local model directory
        """
        return str(self.webapp_models_dir / "best_model")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and manage models for BERT webapp")
    parser.add_argument("--copy", action="store_true", help="Copy model to webapp directory")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing model")
    parser.add_argument("--status", action="store_true", help="Check model status")
    parser.add_argument("--info", action="store_true", help="Show model information")
    
    args = parser.parse_args()
    
    loader = ModelLoader()
    
    if args.status or args.info:
        status = loader.check_model_status()
        print("\n=== Model Status ===")
        print(f"Source model available: {status['source_available']}")
        print(f"Webapp model available: {status['webapp_available']}")
        print(f"Needs update: {status['needs_update']}")
        
        if args.info and status['webapp_available']:
            print(f"\nWebapp model path: {loader.get_webapp_model_path()}")
            if status['webapp_info']:
                print(f"Model files: {', '.join(status['webapp_info']['files'])}")
    
    if args.copy:
        print("\n=== Copying Model ===")
        success = loader.copy_model_to_webapp(force_update=args.force)
        if success:
            print("Model copied successfully!")
            print(f"Model available at: {loader.get_webapp_model_path()}")
        else:
            print("Failed to copy model!")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())