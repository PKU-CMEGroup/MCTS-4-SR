import argparse
import os
import yaml
from typing import Dict, Any

from benchmarks import run_benchmark

def deep_merge(base: dict, update: dict) -> dict:
    """Recursively deep merge dictionaries"""
    merged = base.copy()
    
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'base' in config:
        base_path = os.path.join(os.path.dirname(file_path), config['base'])
        base_config = load_config(base_path)
        config = deep_merge(base_config, config)
    
    return config

def main():
    parser = argparse.ArgumentParser(
        description="Symbolic Regression Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument("--benchmark", type=str, required=True,
                      help="Benchmark name (e.g.: Jin, Nguyen)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Parameter extraction and validation
    required_params = ['start_case', 'run_num', 'output_dir', 'model_params']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter in config: {param}")
    
    # Process operator list format
    if isinstance(config['model_params'].get('ops'), str):
        config['model_params']['ops'] = [op.strip() 
                                       for op in config['model_params']['ops'].split(',')]

    # Run benchmark (modification note: use get method to set end_case default)
    run_benchmark(
        benchmark=args.benchmark,
        model_params=config['model_params'],
        start_case=config['start_case'],
        end_case=config.get('end_case', None),  # Use get method to retrieve; returns None if not exists
        run_num=config['run_num'],
        output_dir=config['output_dir']
    )

if __name__ == "__main__":
    main()