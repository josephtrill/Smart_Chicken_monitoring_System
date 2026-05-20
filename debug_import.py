
import sys
import os

sys.path.append(os.getcwd())

try:
    from app import app
    print("Successfully imported app")
except Exception as e:
    print(f"Failed to import app: {e}")
