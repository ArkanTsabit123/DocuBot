# docubot/scripts/download_models.py

"""
DocuBot - Model Download CLI Tool

Command-line interface for downloading AI models with progress tracking,
batch operations, and comprehensive error handling.
"""

import sys
import os
import argparse
import json
import time
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_engine.model_manager import (
    ModelManager,
    DownloadStatus,
    DownloadProgress,
    ModelManagerError,
    NetworkError,
    DiskSpaceError
)


class DownloadCLI:
    """Command-line interface for model downloads."""
    
    def __init__(self):
        self.manager = ModelManager()
        self.running = True
        
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="DocuBot Model Download Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --model llama2:7b
  %(prog)s --all
  %(prog)s --list
  %(prog)s --model mistral:7b --force
  %(prog)s --model llama2:7b --model mistral:7b
            """
        )
        
        parser.add_argument(
            "--model", "-m",
            action="append",
            help="Model name to download (can be specified multiple times)"
        )
        
        parser.add_argument(
            "--all",
            action="store_true",
            help="Download all available models"
        )
        
        parser.add_argument(
            "--list", "-l",
            action="store_true",
            help="List available models"
        )
        
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force re-download of existing models"
        )
        
        parser.add_argument(
            "--output", "-o",
            type=Path,
            help="Output directory for downloaded models"
        )
        
        parser.add_argument(
            "--concurrent", "-c",
            type=int,
            default=1,
            help="Number of concurrent downloads (default: 1)"
        )
        
        parser.add_argument(
            "--timeout",
            type=int,
            default=3600,
            help="Download timeout in seconds (default: 3600)"
        )
        
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output results in JSON format"
        )
        
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress progress output"
        )
        
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Validate downloaded models"
        )
        
        return parser.parse_args()
    
    def show_progress_bar(self, progress: DownloadProgress, width: int = 40):
        """Display a progress bar for download progress."""
        if progress.total_bytes:
            percentage = progress.progress_percentage
            filled = int(width * percentage / 100)
            bar = "█" * filled + "░" * (width - filled)
            
            downloaded_mb = progress.downloaded_bytes / (1024 ** 2)
            total_mb = progress.total_bytes / (1024 ** 2)
            
            if progress.speed_bytes_per_sec > 0:
                speed_mb = progress.speed_bytes_per_sec / (1024 ** 2)
                if progress.estimated_seconds_remaining:
                    remaining = self._format_time(progress.estimated_seconds_remaining)
                    speed_info = f"{speed_mb:.1f}MB/s, {remaining} remaining"
                else:
                    speed_info = f"{speed_mb:.1f}MB/s"
            else:
                speed_info = "calculating..."
            
            status_map = {
                DownloadStatus.PENDING: "Pending",
                DownloadStatus.DOWNLOADING: "Downloading",
                DownloadStatus.VERIFYING: "Verifying",
                DownloadStatus.COMPLETED: "Completed",
                DownloadStatus.FAILED: "Failed",
                DownloadStatus.CANCELLED: "Cancelled",
                DownloadStatus.PARTIAL: "Partial"
            }
            
            status_text = status_map.get(progress.status, str(progress.status))
            
            print(f"\r{progress.model_name:20} [{bar}] {percentage:6.2f}% "
                  f"({downloaded_mb:.1f}/{total_mb:.1f} MB) {status_text:12} {speed_info}", end="")
        else:
            status_map = {
                DownloadStatus.PENDING: "Pending",
                DownloadStatus.DOWNLOADING: "Downloading...",
                DownloadStatus.VERIFYING: "Verifying...",
                DownloadStatus.COMPLETED: "Completed",
                DownloadStatus.FAILED: "Failed",
                DownloadStatus.CANCELLED: "Cancelled",
                DownloadStatus.PARTIAL: "Partial"
            }
            
            status_text = status_map.get(progress.status, str(progress.status))
            print(f"\r{progress.model_name:20} {status_text}", end="")
        
        sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def list_available_models(self, args: argparse.Namespace):
        """List available models."""
        try:
            models = self.manager.list_available_models()
            
            if args.json:
                print(json.dumps(models, indent=2))
                return
            
            print("\nAvailable Models:")
            print("=" * 80)
            print(f"{'Name':30} {'Size':10} {'Modified':20}")
            print("-" * 80)
            
            for model in models:
                name = model.get('name', '')
                size = model.get('size', '')
                modified = model.get('modified', '')
                print(f"{name:30} {size:10} {modified:20}")
            
            print("=" * 80)
            
            local_models = self.manager.list_available_models(local_only=True)
            if local_models:
                print(f"\nLocal Models ({len(local_models)}):")
                print(", ".join(m.get('name', '') for m in local_models))
            
        except ModelManagerError as e:
            print(f"Error listing models: {e}", file=sys.stderr)
            sys.exit(1)
    
    def download_single_model(self, model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
        """Download a single model."""
        start_time = time.time()
        result = {
            "model": model_name,
            "success": False,
            "error": None,
            "download_time": 0,
            "final_status": None
        }
        
        try:
            if not args.quiet:
                print(f"Starting download: {model_name}")
            
            last_progress = {}
            
            def progress_callback(progress: DownloadProgress):
                if not args.quiet and progress.model_name == model_name:
                    if args.json:
                        return
                    
                    current_progress = {
                        "percentage": progress.progress_percentage,
                        "status": progress.status.value,
                        "downloaded": progress.downloaded_bytes,
                        "speed": progress.speed_bytes_per_sec
                    }
                    
                    if current_progress != last_progress.get(model_name):
                        self.show_progress_bar(progress)
                        last_progress[model_name] = current_progress
            
            download_progress = self.manager.download_model(
                model_name,
                callback=progress_callback,
                force=args.force
            )
            
            while download_progress.status in [
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
                DownloadStatus.VERIFYING
            ]:
                time.sleep(0.5)
                
                current_progress = self.manager.get_download_status(model_name)
                if not current_progress:
                    break
                
                download_progress = current_progress
                
                if time.time() - start_time > args.timeout:
                    self.manager.cancel_download(model_name)
                    raise TimeoutError(f"Download timeout after {args.timeout} seconds")
            
            result["download_time"] = time.time() - start_time
            result["final_status"] = download_progress.status.value
            
            if download_progress.status == DownloadStatus.COMPLETED:
                result["success"] = True
                
                if args.validate:
                    if not args.quiet:
                        print(f"\nValidating model: {model_name}")
                    
                    if self.manager.verify_model(model_name):
                        result["validation"] = "passed"
                    else:
                        result["validation"] = "failed"
                        result["success"] = False
                        result["error"] = "Model validation failed"
            
            elif download_progress.status == DownloadStatus.FAILED:
                result["error"] = download_progress.error_message or "Download failed"
            
            elif download_progress.status == DownloadStatus.CANCELLED:
                result["error"] = "Download cancelled"
            
            if not args.quiet and not args.json:
                if download_progress.status == DownloadStatus.COMPLETED:
                    print(f"\n✓ Downloaded {model_name} in {result['download_time']:.1f}s")
                else:
                    status_text = download_progress.status.value.capitalize()
                    print(f"\n✗ {status_text}: {model_name}")
                    if download_progress.error_message:
                        print(f"  Error: {download_progress.error_message}")
            
            return result
            
        except (ModelManagerError, TimeoutError) as e:
            result["error"] = str(e)
            result["download_time"] = time.time() - start_time
            
            if not args.quiet:
                print(f"\n✗ Error downloading {model_name}: {e}")
            
            return result
    
    def download_multiple_models(self, model_names: List[str], args: argparse.Namespace) -> List[Dict[str, Any]]:
        """Download multiple models concurrently."""
        results = []
        
        if not args.quiet:
            print(f"\nStarting download of {len(model_names)} model(s)")
            if args.concurrent > 1:
                print(f"Concurrent downloads: {args.concurrent}")
            print()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            future_to_model = {
                executor.submit(self.download_single_model, model, args): model
                for model in model_names
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "model": model,
                        "success": False,
                        "error": str(e),
                        "download_time": 0,
                        "final_status": "exception"
                    })
        
        return results
    
    def get_models_to_download(self, args: argparse.Namespace) -> List[str]:
        """Determine which models to download based on arguments."""
        models_to_download = []
        
        if args.all:
            try:
                available_models = self.manager.list_available_models()
                models_to_download = [m['name'] for m in available_models]
            except ModelManagerError as e:
                print(f"Error getting available models: {e}", file=sys.stderr)
                sys.exit(1)
        
        elif args.model:
            models_to_download = args.model
        
        return models_to_download
    
    def print_summary(self, results: List[Dict[str, Any]], args: argparse.Namespace):
        """Print download summary."""
        if args.json:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total": len(results),
                "successful": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"]),
                "results": results
            }
            print(json.dumps(summary, indent=2))
            return
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total models: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print(f"\nSuccessful downloads:")
            for result in successful:
                time_str = self._format_time(result["download_time"])
                print(f"  ✓ {result['model']} ({time_str})")
        
        if failed:
            print(f"\nFailed downloads:")
            for result in failed:
                error = result.get("error", "Unknown error")
                print(f"  ✗ {result['model']}: {error}")
        
        total_time = sum(r.get("download_time", 0) for r in results)
        if total_time > 0:
            print(f"\nTotal download time: {self._format_time(total_time)}")
        
        print("="*60)
    
    def main(self) -> int:
        """Main entry point for the CLI tool."""
        args = self.parse_args()
        
        if args.list:
            self.list_available_models(args)
            return 0
        
        models_to_download = self.get_models_to_download(args)
        
        if not models_to_download:
            print("No models specified for download.", file=sys.stderr)
            print("Use --model <name> or --all to download models.", file=sys.stderr)
            return 1
        
        try:
            results = self.download_multiple_models(models_to_download, args)
            self.print_summary(results, args)
            
            successful = sum(1 for r in results if r["success"])
            if successful == len(results):
                return 0
            elif successful > 0:
                return 2
            else:
                return 1
                
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user.", file=sys.stderr)
            
            active_downloads = self.manager.get_active_downloads()
            if active_downloads:
                print(f"Cancelling {len(active_downloads)} active download(s)...")
                for model in active_downloads:
                    self.manager.cancel_download(model)
            
            return 130  # SIGINT exit code
        
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return 1


def main():
    """Command-line entry point."""
    cli = DownloadCLI()
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())