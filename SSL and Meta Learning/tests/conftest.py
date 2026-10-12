import os
import sys

os.environ.setdefault("GNR650_NO_REEXEC", "1")   # importing test.py must never re-exec the interpreter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
