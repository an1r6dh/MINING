import os
import sys

# Ensure parent directory (workspace root) is in Python path for Vercel serverless execution
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main import app
