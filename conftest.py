# Ensures the repo root is importable so `pytest` finds the `regime` package.
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
