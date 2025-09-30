#!/usr/bin/env python3
"""
HuggingFace Processing Script for Asset vs Deficit Language Analysis

This script processes extracted content from JSON files using HuggingFace transformers with prompts
to identify asset-based versus deficit-based language.

Features:
- Sequential processing (default) for reliable, predictable execution
- Batch processing for improved GPU utilization and faster inference
- Automatic optimal batch size detection based on GPU memory
- Chat template support for instruction-tuned models (auto-enabled)
- Real-time processing feedback with success/failure indicators
- Progress bars for monitoring

Output files are automatically named with model and prompt information to avoid overwrites:
- Format: {base_name}_{model}_{prompt}_{hash}.{extension}
- Example: results_meta-llama_Llama-2-7b-chat-hf_semantic_extraction2_a1b2c3d4.json

Usage:
    python process_with_huggingface.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--model MODEL_NAME] [--prompt PROMPT_NAME] [--device DEVICE] [--quantization QUANTIZATION] [--batch-size BATCH_SIZE] [--max-workers WORKERS]

Args:
    --input: Path to input JSON file (default: data/input/sample_data.json)
    --output: Path to output JSON file base name (default: data/output/huggingface_results.json)
    --model: HuggingFace model to use (default: meta-llama/Llama-2-7b-chat-hf)
    --prompt: Prompt name from prompts.py to use (default: semantic_extraction2)
    --device: Device to use (auto, cpu, cuda, cuda:0, etc.) (default: auto)
    --quantization: Model quantization method (none, 4bit, 8bit, auto) (default: auto)
    --max-length: Maximum generation length (default: 4096)
    --temperature: Generation temperature (default: 0.7)
    --top-p: Top-p sampling parameter (default: 0.9)
    --top-k: Top-k sampling parameter (default: None - disabled)
    --batch-size: Batch size for processing (1 = sequential, >1 = batch processing, 0 = auto-detect optimal batch size) (default: 1)
    --max-workers: Maximum worker threads for parallel processing (default: 1)

Requirements:
    - transformers: pip install transformers
    - torch: pip install torch
    - tqdm: pip install tqdm
    - HuggingFace cache and token environment variables
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import datetime
import logging
import os
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

# Suppress some common warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages:")
    print("pip install transformers>=4.45.0 torch accelerate bitsandbytes")
    sys.exit(1)

# Import prompts from local config
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from config.prompts import PROMPTS, MODEL_CONFIGS, ENV_CONFIG
except ImportError:
    print("Error: Could not import prompts.py. Make sure it's in the config/ directory.")
    sys.exit(1)


def setup_hf_environment():
    """Setup HuggingFace environment variables and cache directories."""
    cache_base = ENV_CONFIG["cache_base"]
    
    # Set environment variables
    env_vars = {
        "HF_HOME": cache_base,
        "HF_HUB_CACHE": f"{cache_base}",
        "HF_ASSETS_CACHE": f"{cache_base}/assets", 
        "HF_DATASETS_CACHE": f"{cache_base}/datasets",
        "TRANSFORMERS_HOME": cache_base,
        "TRANSFORMERS_CACHE": cache_base,
        "HF_TOKEN": ENV_CONFIG["hf_token"]
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Create cache directories
    for cache_dir in [cache_base, f"{cache_base}/hub", f"{cache_base}/assets", f"{cache_base}/datasets"]:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"✓ HuggingFace environment configured with cache at: {cache_base}")


def setup_logging(model_name: str, prompt_name: str) -> str:
    """Setup logging configuration and return log filename."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Clean model name for filename safety
    clean_model = model_name.replace("/", "_").replace(":", "_").replace("\\", "_")
    
    # Create descriptive log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"huggingface_analysis_{clean_model}_{prompt_name}_{timestamp}.log"
    
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
    clean_model = model_name.replace("/", "_").replace(":", "_").replace("\\", "_")
    
    # Create a unique hash based on current time and input file path
    hash_input = f"{datetime.datetime.now().isoformat()}_{input_file}_{model_name}_{prompt_name}"
    unique_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    # Create the new filename with model, prompt, and hash
    stem = path_obj.stem
    suffix = path_obj.suffix
    parent = path_obj.parent
    
    new_filename = f"{stem}_{clean_model}_{prompt_name}_{unique_hash}{suffix}"
    return str(parent / new_filename)


def detect_device() -> str:
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logging.info(f"✓ CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        logging.info(f"  CUDA version: {torch.version.cuda}")
        logging.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        logging.info("ℹ Using CPU (CUDA not available)")
    
    return device


def setup_quantization_config(quantization_type: str = "4bit") -> Optional[BitsAndBytesConfig]:
    """Setup quantization configuration for memory efficiency."""
    if quantization_type == "none":
        return None
    
    if quantization_type == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_type == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        return None
    
    return quantization_config


def load_model_and_tokenizer(model_name: str, device: str = "auto", quantization: str = "auto"):
    """Load HuggingFace model and tokenizer."""
    logging.info(f"Loading model: {model_name}")
    
    # Determine actual device
    if device == "auto":
        device = detect_device()
    
    # Determine quantization type
    if quantization == "auto":
        quantization_type = "4bit" if device.startswith("cuda") else "none"
    else:
        quantization_type = quantization
    
    # Setup quantization
    quantization_config = None
    if quantization_type != "none":
        quantization_config = setup_quantization_config(quantization_type)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=os.environ.get("HF_HOME")
    )
    
    # Load model with generic class
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": os.environ.get("HF_HOME"),
        "torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        if device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"
    elif device.startswith("cuda"):
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    if not quantization_config and not device.startswith("cuda"):
        model = model.to(device)
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = 'left'
    
    return model, tokenizer, device


def prepare_model_for_generation(model, tokenizer, device: str):
    """Prepare model and tokenizer for direct generation."""
    # Ensure model is in eval mode
    model.eval()
    
    # Configure tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    return model, tokenizer


def determine_optimal_batch_size(device: str) -> int:
    """Determine optimal batch size based on available GPU memory."""
    if not device.startswith("cuda"):
        return 1  # CPU processing typically better with batch_size=1
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_gb = total_memory / 1e9
        
        # Conservative estimates based on GPU memory
        if memory_gb >= 40:
            return 8
        elif memory_gb >= 24:
            return 4
        elif memory_gb >= 16:
            return 2
        else:
            return 1
    except:
        return 1


def generate_text(model, tokenizer, prompts, max_length: int = 2048, 
                 temperature: float = 0.7, top_p: float = 0.9, top_k: Optional[int] = None):
    """Generate text using model.generate directly."""
    # Handle both single prompt and batch of prompts
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Convert prompts to chat format and tokenize directly
    inputs_list = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs_list.append(inputs)
    
    # Combine all inputs into a batch
    if len(inputs_list) == 1:
        batch_inputs = inputs_list[0]
    else:
        # Pad to same length for batching
        max_len = max(inputs.shape[1] for inputs in inputs_list)
        padded_inputs = []
        for inputs in inputs_list:
            if inputs.shape[1] < max_len:
                padding = torch.full((inputs.shape[0], max_len - inputs.shape[1]), 
                                   tokenizer.pad_token_id, dtype=inputs.dtype)
                inputs = torch.cat([padding, inputs], dim=1)
            padded_inputs.append(inputs)
        batch_inputs = torch.cat(padded_inputs, dim=0)
    
    # Move to device
    batch_inputs = batch_inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if top_k is not None:
            generation_kwargs["top_k"] = top_k
        
        outputs = model.generate(batch_inputs, **generation_kwargs)
    
    # Decode outputs
    generated_texts = []
    for i, output in enumerate(outputs):
        # Remove input tokens to get only generated text
        input_length = batch_inputs[i].shape[0]
        generated_tokens = output[input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts


def process_content_batch(content_list: List[str], model, tokenizer, prompt_name: str, 
                         max_length: int = 2048, temperature: float = 0.7, top_p: float = 0.9, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Process a batch of content using the specified prompt."""
    # Get the specified prompt from prompts.py
    prompt_template = PROMPTS.get(prompt_name, "")
    if not prompt_template:
        raise ValueError(f"Prompt '{prompt_name}' not found in prompts.py")
    
    # Format prompts for each content item
    formatted_prompts = []
    for content in content_list:
        if content and content.strip():
            formatted_prompt = prompt_template.format(input_sentence=content)
            formatted_prompts.append(formatted_prompt)
        else:
            formatted_prompts.append("")  # Empty content
    
    # Generate responses
    try:
        if any(formatted_prompts):  # Only generate if there are non-empty prompts
            non_empty_indices = [i for i, p in enumerate(formatted_prompts) if p]
            non_empty_prompts = [formatted_prompts[i] for i in non_empty_indices]
            
            generated_texts = generate_text(model, tokenizer, non_empty_prompts, max_length, temperature, top_p, top_k)
            
            # Map results back to original indices
            all_results = [""] * len(formatted_prompts)
            for idx, result in zip(non_empty_indices, generated_texts):
                all_results[idx] = result
        else:
            all_results = [""] * len(formatted_prompts)
        
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        all_results = [""] * len(formatted_prompts)
    
    # Parse each response
    results = []
    for i, generated_text in enumerate(all_results):
        if not generated_text:
            parsed_result = {
                "analysis": None,
                "raw_response": "",
                "parse_error": "Empty or failed generation"
            }
        else:
            parsed_result = parse_json_response(generated_text)
        
        # Show success/failure status
        if parsed_result.get("parse_error"):
            print(f"  ❌ Parsing failed for item {i+1}")
        else:
            print(f"  ✅ Successfully parsed item {i+1}")
        
        results.append(parsed_result)
    
    return results


def parse_json_response(generated_text: str) -> Dict[str, Any]:
    """Parse JSON response from generated text with improved error handling."""
    if not generated_text or generated_text.strip() == "":
        return {
            "analysis": None,
            "raw_response": generated_text,
            "parse_error": "Empty response"
        }
    
    # Try to parse the response as JSON
    try:
        # Sometimes the response might have extra text, try to extract JSON
        if generated_text:
            # Look for JSON-like content
            start_idx = generated_text.find('[')
            if start_idx == -1:
                start_idx = generated_text.find('{')
            
            if start_idx != -1:
                # Try to find the end of JSON
                if generated_text[start_idx] == '[':
                    bracket_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(generated_text[start_idx:], start_idx):
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
                    for i, char in enumerate(generated_text[start_idx:], start_idx):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                
                json_str = generated_text[start_idx:end_idx]
                parsed_response = json.loads(json_str)
                
                # Validate the structure of parsed_response
                if isinstance(parsed_response, list):
                    # Check if all items in the list are dictionaries with required fields
                    valid_items = []
                    for item in parsed_response:
                        if isinstance(item, dict) and all(key in item for key in ['type', 'text_span']):
                            valid_items.append(item)
                        elif isinstance(item, str):
                            # Convert string items to warning in analysis
                            logging.warning(f"Found string item in analysis array: {item[:100]}... Skipping invalid format.")
                    
                    # Only return valid items if we have any
                    if valid_items:
                        return {
                            "analysis": valid_items,
                            "raw_response": generated_text
                        }
                    else:
                        return {
                            "analysis": [],
                            "raw_response": generated_text,
                            "parse_error": "No valid dictionary items found in analysis array"
                        }
                elif isinstance(parsed_response, dict):
                    # Single dictionary response
                    return {
                        "analysis": [parsed_response] if all(key in parsed_response for key in ['type', 'text_span']) else [],
                        "raw_response": generated_text
                    }
                else:
                    # Invalid format
                    return {
                        "analysis": [],
                        "raw_response": generated_text,
                        "parse_error": f"Invalid JSON structure: expected list or dict, got {type(parsed_response).__name__}"
                    }
                    
    except json.JSONDecodeError:
        pass
    
    # If parsing fails, return raw response
    return {
        "analysis": None,
        "raw_response": generated_text,
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
                "extracted_content": "This underserved neighborhood lacks basic resources and faces significant challenges. However, local residents have shown remarkable resilience and have formed community gardens that are thriving. The area needs external intervention to address its deficits, but the community groups have demonstrated strong leadership in organizing weekly farmers markets.",
                "content_length": 387
            },
            {
                "Sr": 2,
                "source": "https://example.org/asset-based-approach",
                "extracted_title": "Building on Community Strengths",
                "extracted_content": "This vibrant community is rich in cultural heritage and has existing networks of mutual support. Local artists, elders, and youth leaders are collaborating to create a community center that will serve as a hub for skill-sharing and resource pooling. The residents are actively mapping their assets and building connections.",
                "content_length": 342
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
    parser = argparse.ArgumentParser(description="Process content with HuggingFace for asset/deficit analysis")
    parser.add_argument(
        "--input", 
        default="data/input/sample_data.json",
        help="Path to input JSON file (default: data/input/sample_data.json)"
    )
    parser.add_argument(
        "--output", 
        default="data/output/huggingface_results.json",
        help="Base path for output JSON file - actual filename will include model and prompt info (default: data/output/huggingface_results.json)"
    )
    parser.add_argument(
        "--model", 
        default=MODEL_CONFIGS["huggingface"]["default_model"],
        help=f"HuggingFace model to use (default: {MODEL_CONFIGS['huggingface']['default_model']})"
    )
    parser.add_argument(
        "--prompt", 
        default="semantic_extraction2",
        help="Prompt name from prompts.py to use (default: semantic_extraction2)"
    )
    parser.add_argument(
        "--device", 
        default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0, etc.) (default: auto)"
    )
    parser.add_argument(
        "--max-length", 
        type=int,
        default=MODEL_CONFIGS["huggingface"]["default_max_length"],
        help=f"Maximum generation length (default: {MODEL_CONFIGS['huggingface']['default_max_length']})"
    )
    parser.add_argument(
        "--temperature", 
        type=float,
        default=MODEL_CONFIGS["huggingface"]["default_temperature"],
        help=f"Generation temperature (default: {MODEL_CONFIGS['huggingface']['default_temperature']})"
    )
    parser.add_argument(
        "--top-p", 
        type=float,
        default=MODEL_CONFIGS["huggingface"]["default_top_p"],
        help=f"Top-p sampling parameter (default: {MODEL_CONFIGS['huggingface']['default_top_p']})"
    )
    parser.add_argument(
        "--top-k", 
        type=int,
        default=None,
        help="Top-k sampling parameter (default: None - disabled)"
    )
    parser.add_argument(
        "--quantization", 
        choices=["none", "4bit", "8bit", "auto"],
        default=MODEL_CONFIGS["huggingface"]["default_quantization"],
        help=f"Model quantization method (none, 4bit, 8bit, auto). Auto uses 4bit on GPU, none on CPU (default: {MODEL_CONFIGS['huggingface']['default_quantization']})"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        default=1,
        help="Batch size for parallel processing (1 = sequential, >1 = batch processing, 0 = auto-detect optimal batch size) (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Create sample input if it doesn't exist
    if not Path(args.input).exists():
        logging.info(f"Input file {args.input} not found. Creating sample data...")
        args.input = create_sample_input()
        logging.info(f"Created sample input file: {args.input}")
    
    # Setup HuggingFace environment
    setup_hf_environment()
    
    # Setup logging
    log_file = setup_logging(args.model, args.prompt)
    
    # Log run configuration
    logging.info("=== HuggingFace Analysis Run Started ===")
    logging.info(f"Input file: {args.input}")
    logging.info(f"Output file pattern: {args.output}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Prompt: {args.prompt}")
    logging.info(f"Device: {args.device}")
    logging.info(f"Quantization: {args.quantization}")
    logging.info(f"Max length: {args.max_length}")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"Top-p: {args.top_p}")
    logging.info(f"Top-k: {args.top_k}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Log file: {log_file}")
    
    # Validate prompt name
    if args.prompt not in PROMPTS:
        logging.error(f"Prompt '{args.prompt}' not found in prompts.py")
        logging.error(f"Available prompts: {', '.join(PROMPTS.keys())}")
        sys.exit(1)
    
    # Load model and tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer(args.model, args.device, args.quantization)
        
        # Prepare model for generation
        model, tokenizer = prepare_model_for_generation(model, tokenizer, device)
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Determine batch size
    if args.batch_size == 0:
        batch_size = determine_optimal_batch_size(device)
        logging.info(f"✓ Auto-detected batch size: {batch_size}")
    else:
        batch_size = args.batch_size
        logging.info(f"✓ Using batch size: {batch_size}")
    
    # Load input data
    logging.info(f"Loading data from: {args.input}")
    input_data = load_input_data(args.input)
    logging.info(f"✓ Loaded {len(input_data)} records")
    
    # Process records
    processed_data = []
    total_records = len(input_data)
    start_time = time.time()
    
    # Process in batches
    logging.info(f"Starting processing with batch_size={batch_size}...")
    
    with tqdm(total=len(input_data), desc="Processing records", unit="record") as pbar:
        for i in range(0, len(input_data), batch_size):
            batch_records = input_data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(input_data) + batch_size - 1) // batch_size
            
            # Update progress bar description
            pbar.set_description(f"Batch {batch_num}/{total_batches}")
            
            # Extract content from batch
            content_list = [record.get("extracted_content", "") for record in batch_records]
            
            # Process content batch
            try:
                analysis_results = process_content_batch(
                    content_list, model, tokenizer, args.prompt,
                    args.max_length, args.temperature, args.top_p, args.top_k
                )
                
                # Combine results with record metadata
                for record, result in zip(batch_records, analysis_results):
                    processed_record = {
                        "Sr": record.get("Sr"),
                        "source": record.get("source"),
                        "extracted_title": record.get("extracted_title"),
                        "extracted_content": record.get("extracted_content"),
                        "content_length": record.get("content_length"),
                    }
                    
                    if record.get("extracted_content"):
                        if result.get("parse_error"):
                            processed_record.update({
                                "linguistic_classification": {"error": "JSON parsing failed"},
                                "processing_status": "error"
                            })
                        else:
                            processed_record.update({
                                "linguistic_classification": result,
                                "processing_status": "success"
                            })
                    else:
                        processed_record.update({
                            "linguistic_classification": {"error": "No extracted_content found"},
                            "processing_status": "skipped"
                        })
                    
                    processed_data.append(processed_record)
                    
                # Update progress bar
                pbar.update(len(batch_records))
                    
            except Exception as e:
                logging.error(f"Error processing batch {batch_num}: {e}")
                # Add error records for this batch
                for record in batch_records:
                    error_record = {
                        "Sr": record.get("Sr"),
                        "source": record.get("source"),
                        "extracted_title": record.get("extracted_title"),
                        "extracted_content": record.get("extracted_content"),
                        "content_length": record.get("content_length"),
                        "linguistic_classification": {"error": f"Batch processing failed: {str(e)}"},
                        "processing_status": "error"
                    }
                    processed_data.append(error_record)
                
                # Update progress bar even on error
                pbar.update(len(batch_records))
    
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