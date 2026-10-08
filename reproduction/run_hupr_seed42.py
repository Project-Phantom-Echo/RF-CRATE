"""Compatibility entry point for the completed colleague-account campaign."""
from pathlib import Path
import sys
from run_hupr import main

if __name__ == '__main__':
    if '--campaign' not in sys.argv:
        sys.argv.extend(['--campaign', str(Path(__file__).resolve().parents[1]/'local/hupr-seed42-20261008')])
    main()
