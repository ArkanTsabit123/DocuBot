"""
Test coverage utilities for DocuBot.
"""

import coverage
import sys
from pathlib import Path
from typing import Dict, Any, List


class CoverageManager:
    """Manage test coverage measurement."""
    
    def __init__(self, source_dir: str = "src", omit_patterns: List[str] = None):
        """
        Initialize coverage manager.
        
        Args:
            source_dir: Directory containing source code
            omit_patterns: Patterns to omit from coverage
        """
        self.source_dir = source_dir
        self.omit_patterns = omit_patterns or [
            "*/tests/*",
            "*/migrations/*",
            "*/__pycache__/*"
        ]
        
        self.cov = coverage.Coverage(
            source=[self.source_dir],
            omit=self.omit_patterns,
            branch=True,
            config_file=".coveragerc"
        )
    
    def start(self):
        """Start coverage measurement."""
        self.cov.start()
        print("Coverage measurement started")
    
    def stop(self):
        """Stop coverage measurement."""
        self.cov.stop()
        print("Coverage measurement stopped")
    
    def save(self):
        """Save coverage data."""
        self.cov.save()
    
    def report(self, output_format: str = "terminal") -> Dict[str, Any]:
        """
        Generate coverage report.
        
        Args:
            output_format: 'terminal', 'html', or 'xml'
        
        Returns:
            Dict[str, Any]: Coverage statistics
        """
        self.cov.load()
        
        if output_format == "terminal":
            self.cov.report(show_missing=True, skip_covered=False)
        elif output_format == "html":
            self.cov.html_report(directory="coverage_html_report")
            print("HTML report generated in coverage_html_report/")
        elif output_format == "xml":
            self.cov.xml_report(outfile="coverage.xml")
            print("XML report generated as coverage.xml")
        
        total_coverage = self.cov.report()
        
        return {
            'total_coverage': total_coverage,
            'format': output_format,
            'timestamp': self.cov.get_data().measured_files()
        }
    
    def combine(self, data_files: List[str]):
        """
        Combine multiple coverage data files.
        
        Args:
            data_files: List of coverage data files
        """
        self.cov.combine(data_files)
    
    def get_missing_lines(self, filename: str) -> List[int]:
        """
        Get list of missing lines for a file.
        
        Args:
            filename: Source file name
        
        Returns:
            List[int]: Line numbers not covered
        """
        self.cov.load()
        analysis = self.cov.analysis2(filename)
        return analysis.missing if analysis else []


def measure_coverage(test_command: str = "pytest") -> float:
    """
    Run tests with coverage measurement.
    
    Args:
        test_command: Test command to execute
    
    Returns:
        float: Total coverage percentage
    """
    manager = CoverageManager()
    
    try:
        manager.start()
        
        import subprocess
        result = subprocess.run(
            test_command.split(),
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
    finally:
        manager.stop()
        manager.save()
    
    report = manager.report("terminal")
    return report.get('total_coverage', 0.0)


if __name__ == "__main__":
    coverage = measure_coverage("pytest tests/ -v")
    print(f"\nTotal coverage: {coverage:.2f}%")
