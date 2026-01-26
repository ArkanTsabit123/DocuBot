# docubot/scripts/download_models.py

"""
Model download and management script for DocuBot.
Handles acquisition and validation of language and embedding models.
"""

import os
import sys
import subprocess
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


def check_ollama_installed() -> Tuple[bool, str]:
    """Verify Ollama installation status."""
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return True, f"Ollama version: {result.stdout.strip()}"
        else:
            return False, f"Ollama verification failed: {result.stderr}"
            
    except FileNotFoundError:
        return False, "Ollama not found. Install from https://ollama.ai/"
    except subprocess.TimeoutExpired:
        return False, "Ollama verification timeout"
    except Exception as error:
        return False, f"Ollama verification error: {error}"


def download_ollama_model(model_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """Acquire specified Ollama model."""
    if verbose:
        print(f"Initiating Ollama model download: {model_name}")
    
    try:
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if verbose:
            for line in iter(process.stdout.readline, ''):
                print(f"  {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            return True, f"Model acquired successfully: {model_name}"
        else:
            error_output = process.stderr.read()
            return False, f"Model acquisition failed for {model_name}: {error_output}"
            
    except FileNotFoundError:
        return False, "Ollama executable not found"
    except Exception as error:
        return False, f"Download error for {model_name}: {error}"


def download_ollama_model_binary(model_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """Alternative download method using binary protocol for Ollama models."""
    if verbose:
        print(f"Starting binary protocol download for: {model_name}")
    
    try:
        command = ['ollama', 'pull', model_name]
        
        if verbose:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1200
            )
            
            for line in process.stdout.splitlines():
                if 'downloading' in line.lower() or 'complete' in line.lower():
                    print(f"  {line.strip()}")
        else:
            process = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=1200
            )
        
        if process.returncode == 0:
            return True, f"Binary download completed for {model_name}"
        else:
            error_msg = process.stderr if hasattr(process, 'stderr') else "Unknown error"
            return False, f"Binary download failed: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, f"Download timeout for {model_name} (20 minutes)"
    except Exception as error:
        return False, f"Binary download exception: {error}"


def check_sentence_transformers_installed() -> Tuple[bool, str]:
    """Verify sentence-transformers package availability."""
    try:
        import importlib.util
        
        if importlib.util.find_spec('sentence_transformers'):
            return True, "sentence-transformers package available"
        else:
            return False, "sentence-transformers package not found"
            
    except ImportError:
        return False, "sentence-transformers not installed"
    except Exception as error:
        return False, f"sentence-transformers verification error: {error}"


def download_sentence_transformer_model(model_name: str) -> Tuple[bool, str]:
    """Download and load sentence-transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"Acquiring sentence-transformer model: {model_name}")
        
        start_time = time.time()
        model = SentenceTransformer(model_name)
        elapsed_time = time.time() - start_time
        
        if model:
            return True, f"Model loaded successfully in {elapsed_time:.1f} seconds"
        else:
            return False, f"Model loading failed: {model_name}"
            
    except ImportError:
        return False, "Install required: pip install sentence-transformers"
    except Exception as error:
        return False, f"Model acquisition error: {error}"


def verify_model_download(model_type: str, model_name: str) -> Tuple[bool, str]:
    """Validate successful model acquisition."""
    if model_type == 'llm':
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                if model_name in result.stdout:
                    return True, f"Model present in Ollama registry: {model_name}"
                else:
                    return False, f"Model missing from Ollama registry: {model_name}"
            else:
                return False, f"Ollama registry query failed: {result.stderr}"
                
        except Exception as error:
            return False, f"LLM verification error: {error}"
    
    elif model_type == 'embedding':
        try:
            from sentence_transformers import SentenceTransformer
            
            try:
                SentenceTransformer(model_name)
                return True, f"Embedding model validated: {model_name}"
            except Exception as error:
                return False, f"Embedding model validation failed: {error}"
                
        except ImportError:
            return False, "sentence-transformers package unavailable"
    
    return False, f"Unsupported model type: {model_type}"


def load_model_config(config_path: Optional[Path] = None) -> Dict:
    """Load model configuration from specified file."""
    default_configuration = {
        'llm_models': [
            {'name': 'llama2:7b', 'display_name': 'Llama 2 7B'},
            {'name': 'mistral:7b', 'display_name': 'Mistral 7B'},
            {'name': 'neural-chat:7b', 'display_name': 'Neural Chat 7B'}
        ],
        'embedding_models': [
            {'name': 'all-MiniLM-L6-v2', 'display_name': 'MiniLM L6 v2'},
            {'name': 'all-mpnet-base-v2', 'display_name': 'MPNet Base v2'}
        ]
    }
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as config_file:
                import yaml
                user_configuration = yaml.safe_load(config_file)
                
                if user_configuration:
                    default_configuration.update(user_configuration)
                    
        except Exception as error:
            print(f"Configuration file load warning: {error}")
    
    return default_configuration


def calculate_file_hash(file_path: Path) -> str:
    """Generate SHA256 hash for specified file."""
    hash_algorithm = hashlib.sha256()
    
    try:
        with open(file_path, 'rb') as target_file:
            for data_chunk in iter(lambda: target_file.read(4096), b''):
                hash_algorithm.update(data_chunk)
    except Exception:
        return ""
    
    return hash_algorithm.hexdigest()


def save_download_record(model_info: Dict, success: bool, message: str) -> None:
    """Record download attempt details."""
    record_directory = Path.home() / '.docubot' / 'download_logs'
    record_directory.mkdir(parents=True, exist_ok=True)
    
    record_filename = record_directory / f"download_{int(time.time())}.json"
    
    record_data = {
        'timestamp': time.time(),
        'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_info,
        'success': success,
        'message': message
    }
    
    try:
        with open(record_filename, 'w') as record_file:
            json.dump(record_data, record_file, indent=2)
    except Exception as error:
        print(f"Download record save warning: {error}")


def check_disk_space(required_gb: float) -> Tuple[bool, str]:
    """Validate available disk capacity."""
    try:
        import shutil
        
        home_directory = Path.home()
        usage_data = shutil.disk_usage(str(home_directory))
        
        free_gigabytes = usage_data.free / (1024 ** 3)
        
        if free_gigabytes >= required_gb:
            return True, f"Disk capacity sufficient: {free_gigabytes:.1f}GB available"
        else:
            return False, f"Insufficient disk space: {free_gigabytes:.1f}GB available, {required_gb}GB required"
            
    except Exception as error:
        return False, f"Disk space verification error: {error}"


def main():
    """Execute model acquisition workflow."""
    parser = argparse.ArgumentParser(description="DocuBot Model Acquisition System")
    parser.add_argument('--models', type=str, help='Comma-delimited model list')
    parser.add_argument('--type', choices=['llm', 'embedding', 'all'], default='all',
                       help='Model category specification')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--verify-only', action='store_true',
                       help='Validate existing models without acquisition')
    parser.add_argument('--skip-ollama-check', action='store_true',
                       help='Bypass Ollama installation verification')
    parser.add_argument('--skip-space-check', action='store_true',
                       help='Bypass disk capacity verification')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable detailed output')
    parser.add_argument('--use-binary', action='store_true',
                       help='Employ binary protocol for Ollama downloads')
    
    args = parser.parse_args()
    
    print("DocuBot Model Acquisition System")
    print("=" * 50)
    
    configuration_path = Path(args.config) if args.config else None
    configuration = load_model_config(configuration_path)
    
    if not args.skip_ollama_check:
        ollama_status, ollama_status_message = check_ollama_installed()
        print(f"Ollama Status: {ollama_status_message}")
    
    sentence_transformers_status, st_status_message = check_sentence_transformers_installed()
    print(f"Sentence-transformers Status: {st_status_message}")
    
    acquisition_list = []
    
    if args.models:
        user_specified_models = [model.strip() for model in args.models.split(',')]
        
        for model_identifier in user_specified_models:
            if ':7b' in model_identifier or model_identifier.startswith(('llama', 'mistral')):
                acquisition_list.append({
                    'type': 'llm',
                    'name': model_identifier,
                    'display_name': model_identifier
                })
            else:
                acquisition_list.append({
                    'type': 'embedding',
                    'name': model_identifier,
                    'display_name': model_identifier
                })
    else:
        if args.type in ['llm', 'all']:
            acquisition_list.extend([
                {**model, 'type': 'llm'} 
                for model in configuration['llm_models']
            ])
        
        if args.type in ['embedding', 'all']:
            acquisition_list.extend([
                {**model, 'type': 'embedding'} 
                for model in configuration['embedding_models']
            ])
    
    if not acquisition_list:
        print("No models specified for acquisition.")
        return
    
    if not args.skip_space_check:
        estimated_requirement = len(acquisition_list) * 4
        capacity_status, capacity_message = check_disk_space(estimated_requirement)
        print(f"Storage Capacity: {capacity_message}")
        
        if not capacity_status:
            print("Storage capacity insufficient. Process terminated.")
            return
    
    print(f"\nProcessing {len(acquisition_list)} model(s)...")
    
    successful_acquisitions = 0
    failed_acquisitions = 0
    
    for model_data in acquisition_list:
        model_category = model_data['type']
        model_identifier = model_data['name']
        model_display_name = model_data.get('display_name', model_identifier)
        
        print(f"\n{'='*60}")
        print(f"Model: {model_display_name}")
        print(f"Category: {model_category.upper()}")
        print(f"Identifier: {model_identifier}")
        
        if args.verify_only:
            verification_status, verification_message = verify_model_download(model_category, model_identifier)
            print(f"Verification: {verification_message}")
            
            if verification_status:
                successful_acquisitions += 1
            else:
                failed_acquisitions += 1
            
            save_download_record(model_data, verification_status, verification_message)
            continue
        
        if model_category == 'llm':
            if not ollama_status and not args.skip_ollama_check:
                print("LLM acquisition skipped: Ollama unavailable")
                failed_acquisitions += 1
                save_download_record(model_data, False, "Ollama unavailable")
                continue
            
            if args.use_binary:
                acquisition_status, acquisition_message = download_ollama_model_binary(model_identifier, args.verbose)
            else:
                acquisition_status, acquisition_message = download_ollama_model(model_identifier, args.verbose)
            
            print(f"Acquisition: {acquisition_message}")
            
            if acquisition_status:
                successful_acquisitions += 1
                
                verification_status, verification_message = verify_model_download(model_category, model_identifier)
                print(f"Verification: {verification_message}")
                acquisition_message = f"{acquisition_message} | {verification_message}"
            else:
                failed_acquisitions += 1
            
            save_download_record(model_data, acquisition_status, acquisition_message)
        
        elif model_category == 'embedding':
            if not sentence_transformers_status:
                print("Embedding acquisition skipped: sentence-transformers unavailable")
                failed_acquisitions += 1
                save_download_record(model_data, False, "sentence-transformers unavailable")
                continue
            
            acquisition_status, acquisition_message = download_sentence_transformer_model(model_identifier)
            print(f"Acquisition: {acquisition_message}")
            
            if acquisition_status:
                successful_acquisitions += 1
                
                verification_status, verification_message = verify_model_download(model_category, model_identifier)
                print(f"Verification: {verification_message}")
                acquisition_message = f"{acquisition_message} | {verification_message}"
            else:
                failed_acquisitions += 1
            
            save_download_record(model_data, acquisition_status, acquisition_message)
        
        else:
            print(f"Unsupported model category: {model_category}")
            failed_acquisitions += 1
            save_download_record(model_data, False, f"Unsupported model category: {model_category}")
    
    print(f"\n{'='*50}")
    print("ACQUISITION SUMMARY")
    print(f"{'='*50}")
    print(f"Total models processed: {len(acquisition_list)}")
    print(f"Successful: {successful_acquisitions}")
    print(f"Failed: {failed_acquisitions}")
    
    if successful_acquisitions == len(acquisition_list):
        print("All models acquired successfully.")
        sys.exit(0)
    elif successful_acquisitions > 0:
        print(f"Partial success: {successful_acquisitions}/{len(acquisition_list)} models acquired.")
        sys.exit(1)
    else:
        print("All model acquisitions failed.")
        sys.exit(2)


if __name__ == '__main__':
    main()