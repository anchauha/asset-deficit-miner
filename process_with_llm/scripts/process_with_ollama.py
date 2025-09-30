#!/usr/bin/env python3
"""
Ollama Processing Script for Asset vs Deficit Language Analysis

This script processes extracted content from JSON files using Ollama with prompts
to identify asset-based versus deficit-based language.

Output files are automatically named with model and prompt information to avoid overwrites:
- Format: {base_name}_{model}_{prompt}_{hash}.{extension}
- Example: output_llama4_17b-scout-16e-instruct-q4_K_M_standard_a1b2c3d4.json

Usage:
    python process_with_ollama.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--model MODEL_NAME] [--prompt PROMPT_NAME] [--skip-service-check]

Args:
    --input: Path to input JSON file (default: data/input/examples.json)
    --output: Path to output JSON file base name (default: data/output/ollama_results.json)
    --model: Ollama model to use (default: llama4:17b-scout-16e-instruct-fp16)
    --prompt: Prompt name from prompts.py to use (default: semantic_extraction2)
    --skip-service-check: Skip the Ollama service check (use if managing ollama serve manually)

Requirements:
    - ollama python package: pip install ollama
    - ollama service running (ollama serve)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import time
import datetime
import logging
import os
import hashlib

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available. Install with: pip install tqdm")
    # Fallback to identity function if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import ollama
except ImportError as e:
    print(f"Error importing ollama: {e}")
    print("Please install ollama: pip install ollama")
    sys.exit(1)

# Import prompts from local config
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from config.prompts import PROMPTS, MODEL_CONFIGS
except ImportError:
    print("Error: Could not import prompts.py. Make sure it's in the config/ directory.")
    sys.exit(1)


def setup_logging(model_name: str, prompt_name: str) -> str:
    """Setup logging configuration and return log filename."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Clean model name for filename safety
    clean_model = model_name.replace(":", "_").replace("/", "_").replace("\\", "_")
    
    # Create descriptive log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"ollama_analysis_{clean_model}_{prompt_name}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return str(log_filename)


def generate_output_filename(base_path: str, model_name: str, prompt_name: str, input_file: str) -> str:
    """Generate a descriptive filename using model, prompt, and hash for uniqueness."""
    path_obj = Path(base_path)
    
    # Clean model name (remove special characters for filename safety)
    clean_model = model_name.replace(":", "_").replace("/", "_").replace("\\", "_")
    
    # Create a unique hash based on current time and input file path
    hash_input = f"{datetime.datetime.now().isoformat()}_{input_file}_{model_name}_{prompt_name}"
    unique_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Create the new filename with model, prompt, and hash
    stem = path_obj.stem
    suffix = path_obj.suffix
    parent = path_obj.parent
    
    new_filename = f"{stem}_{clean_model}_{prompt_name}_{unique_hash}{suffix}"
    return str(parent / new_filename)


def check_ollama_service():
    """Check if Ollama service is running."""
    try:
        # Try to list models to check if service is running
        ollama.list()
        return True
    except Exception:
        return False


def call_ollama(prompt: str, model: str) -> str:
    """Call Ollama with the given prompt and model."""
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt
        )
        return response['response']
    except Exception as e:
        logging.error(f"Error calling Ollama: {e}")
        return ""


def process_content(content: str, model: str, prompt_name: str = "semantic_extraction2", debug_prompt: bool = False) -> Dict[str, Any]:
    """Process content using Ollama with the specified prompt."""
    # Get the specified prompt from prompts.py
    prompt_template = PROMPTS.get(prompt_name, "")
    if not prompt_template:
        raise ValueError(f"Prompt '{prompt_name}' not found in prompts.py")
    
    # Format the prompt with the content
    formatted_prompt = prompt_template.format(input_sentence=content)
    
    # Debug: Log the prompt being used
    if debug_prompt:
        logging.info(f"=== DEBUG: Using prompt '{prompt_name}' ===")
        logging.info(f"Prompt template preview: {prompt_template[:200]}...")
        logging.info(f"Formatted prompt preview: {formatted_prompt[:300]}...")
        logging.info("=== END DEBUG ===")
    
    # Call Ollama
    response = call_ollama(formatted_prompt, model)
    
    # Try to parse the response as JSON
    try:
        # Sometimes the response might have extra text, try to extract JSON
        if response:
            # Look for JSON-like content
            start_idx = response.find('[')
            if start_idx == -1:
                start_idx = response.find('{')
            
            if start_idx != -1:
                # Try to find the end of JSON
                if response[start_idx] == '[':
                    bracket_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(response[start_idx:], start_idx):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break
                else:
                    brace_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(response[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                
                json_str = response[start_idx:end_idx]
                parsed_response = json.loads(json_str)
                return {
                    "analysis": parsed_response,
                    "raw_response": response
                }
    except json.JSONDecodeError:
        pass
    
    # If parsing fails, return raw response
    return {
        "analysis": None,
        "raw_response": response,
        "parse_error": True
    }


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data from input JSON file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unexpected JSON structure")
    
    except FileNotFoundError:
        logging.error(f"Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in input file: {e}")
        sys.exit(1)


def save_output_data(output_file: str, processed_data: List[Dict[str, Any]], model_name: str, prompt_name: str, input_file: str) -> str:
    """Save processed data to output JSON file with descriptive filename."""
    try:
        # Generate descriptive filename
        descriptive_file = generate_output_filename(output_file, model_name, prompt_name, input_file)
        
        # Create output directory if it doesn't exist
        output_path = Path(descriptive_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        with open(descriptive_file, 'w', encoding='utf-8') as f:
            json.dump({"data": processed_data}, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results saved to: {descriptive_file}")
        return descriptive_file
    
    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        sys.exit(1)


def create_sample_input():
    """Create a sample input file for testing."""
    sample_data = {
        "data": [
            {
                "Sr": 1,
                "source": "https://example.com/community-article",
                "extracted_title": "Community Development Initiative",
                "extracted_content": "This underserved neighborhood lacks basic resources and faces significant challenges. However, local residents have shown remarkable resilience and have formed community gardens that are thriving. The area needs external intervention to address its deficits, but the community groups have demonstrated strong leadership in organizing weekly farmers markets."
            },
            {
                "Sr": 2,
                "source": "https://example.org/asset-based-approach",
                "extracted_title": "Building on Community Strengths",
                "extracted_content": "This vibrant community is rich in cultural heritage and has existing networks of mutual support. Local artists, elders, and youth leaders are collaborating to create a community center that will serve as a hub for skill-sharing and resource pooling. The residents are actively mapping their assets and building connections."
            }
        ]
    }
    
    input_dir = Path("data/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = input_dir / "sample_data.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    return str(sample_file)


def main():
    parser = argparse.ArgumentParser(description="Process content with Ollama for asset/deficit analysis")
    parser.add_argument(
        "--input", 
        default="data/input/sample_data.json",
        help="Path to input JSON file (default: data/input/sample_data.json)"
    )
    parser.add_argument(
        "--output", 
        default="data/output/ollama_results.json",
        help="Base path for output JSON file - actual filename will include model and prompt info (default: data/output/ollama_results.json)"
    )
    parser.add_argument(
        "--model", 
        default=MODEL_CONFIGS["ollama"]["default_model"],
        help=f"Ollama model to use (default: {MODEL_CONFIGS['ollama']['default_model']})"
    )
    parser.add_argument(
        "--prompt", 
        default="semantic_extraction2",
        help="Prompt name from prompts.py to use (default: semantic_extraction2)"
    )
    parser.add_argument(
        "--skip-service-check", 
        action="store_true",
        help="Skip the Ollama service check (use if managing ollama serve manually)"
    )
    parser.add_argument(
        "--debug-prompt", 
        action="store_true",
        help="Show debug information about the prompt being used"
    )
    
    args = parser.parse_args()
    
    # Create sample input if it doesn't exist
    if not Path(args.input).exists():
        logging.info(f"Input file {args.input} not found. Creating sample data...")
        args.input = create_sample_input()
        logging.info(f"Created sample input file: {args.input}")
    
    # Setup logging
    log_file = setup_logging(args.model, args.prompt)
    
    # Log run configuration
    logging.info("=== Ollama Analysis Run Started ===")
    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file pattern: {args.output}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Prompt: {args.prompt}")
    logging.info(f"Skip service check: {args.skip_service_check}")
    logging.info(f"Debug prompt: {args.debug_prompt}")
    logging.info(f"Log file: {log_file}")
    
    # Validate prompt name
    if args.prompt not in PROMPTS:
        logging.error(f"Prompt '{args.prompt}' not found in prompts.py")
        logging.error(f"Available prompts: {', '.join(PROMPTS.keys())}")
        sys.exit(1)
    
    # Check if Ollama is running (unless skipped)
    if not args.skip_service_check:
        logging.info("Checking Ollama service...")
        if not check_ollama_service():
            logging.error("Ollama service is not running or not accessible.")
            logging.error("Please start Ollama service with: ollama serve")
            logging.error("Or use --skip-service-check if you're managing ollama serve manually")
            sys.exit(1)
        logging.info("✓ Ollama service is running")
    else:
        logging.info("Skipping Ollama service check (assuming it's running)")
    
    # Load input data
    logging.info(f"Loading data from: {args.input}")
    input_data = load_input_data(args.input)
    logging.info(f"✓ Loaded {len(input_data)} records")
    
    # Process each record
    processed_data = []
    total_records = len(input_data)
    start_time = time.time()
    
    for i, record in enumerate(tqdm(input_data, desc="Processing records"), 1):
        # Extract required fields
        processed_record = {
            "Sr": record.get("Sr"),
            "source": record.get("source"),
            "extracted_title": record.get("extracted_title"),
            "extracted_content": record.get("extracted_content"),
        }
        
        # Process the extracted_content
        extracted_content = record.get("extracted_content", "")
        if extracted_content:
            try:
                analysis_result = process_content(extracted_content, args.model, args.prompt, args.debug_prompt)
                processed_record.update({
                    "linguistic_classification": analysis_result,
                    "processing_status": "success"
                })
                logging.debug(f"Successfully processed record {i}")
            except Exception as e:
                logging.error(f"Error processing record {i}: {e}")
                processed_record.update({
                    "linguistic_classification": {"error": str(e)},
                    "processing_status": "error"
                })
        else:
            logging.warning(f"Record {i} has no extracted_content")
            processed_record.update({
                "linguistic_classification": {"error": "No extracted_content found"},
                "processing_status": "skipped"
            })
        
        processed_data.append(processed_record)
        
        # Optional: Log every 10th record for debugging
        if i % 10 == 0:
            logging.debug(f"Processed {i}/{total_records} records")
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save results
    actual_output_file = save_output_data(args.output, processed_data, args.model, args.prompt, args.input)
    logging.info(f"✓ Processing complete! {len(processed_data)} records processed.")
    logging.info(f"Total processing time: {processing_time:.2f} seconds")
    
    # Print summary
    success_count = sum(1 for r in processed_data if r.get("processing_status") == "success")
    error_count = sum(1 for r in processed_data if r.get("processing_status") == "error")
    skipped_count = sum(1 for r in processed_data if r.get("processing_status") == "skipped")
    
    logging.info("\n=== Summary ===")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Errors: {error_count}")
    logging.info(f"Skipped: {skipped_count}")
    logging.info(f"Output file: {actual_output_file}")
    logging.info(f"Log file: {log_file}")
    logging.info("=== Run Completed ===")
    
    # Also print to console for immediate visibility
    print(f"\n✓ Analysis complete!")
    print(f"Results saved to: {actual_output_file}")
    print(f"Log saved to: {log_file}")
    print(f"Summary: {success_count} successful, {error_count} errors, {skipped_count} skipped")


if __name__ == "__main__":
    main()