#!/usr/bin/env python3
"""Create submission zip file"""

import zipfile
import argparse
from pathlib import Path


def create_submission_zip(
    output_file: str = "submission.zip",
    exclude_patterns: list = None
):
    """
    Create a zip file with code and report for submission.
    
    Args:
        output_file: Name of output zip file
        exclude_patterns: List of patterns to exclude (e.g., ['__pycache__', '*.pyc'])
    """
    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '*.pth',
            '*.pt',
            '*.png',
            '*.pdf',
            'artifacts',
            'runs',
            'venv',
            '.git',
            '.gitignore',
            '*.log'
        ]
    
    root_dir = Path(__file__).parent.parent
    
    # Files and directories to include
    include_paths = [
        'src',
        'run',
        'scripts',
        'report',
        'README.md',
        'requirements.txt'
    ]
    
    print(f"Creating submission zip: {output_file}")
    print(f"Root directory: {root_dir}")
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for include_path in include_paths:
            path = root_dir / include_path
            if path.exists():
                if path.is_file():
                    zipf.write(path, path.relative_to(root_dir))
                    print(f"  Added: {path.relative_to(root_dir)}")
                elif path.is_dir():
                    for file_path in path.rglob('*'):
                        if file_path.is_file():
                            # Check if file should be excluded
                            should_exclude = False
                            for pattern in exclude_patterns:
                                if pattern in str(file_path) or file_path.match(pattern):
                                    should_exclude = True
                                    break
                            
                            if not should_exclude:
                                zipf.write(file_path, file_path.relative_to(root_dir))
                                print(f"  Added: {file_path.relative_to(root_dir)}")
    
    print(f"\nSubmission zip created: {output_file}")
    print(f"Size: {Path(output_file).stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create submission zip")
    parser.add_argument("--output", type=str, default="submission.zip", help="Output zip file name")
    args = parser.parse_args()
    
    create_submission_zip(args.output)

