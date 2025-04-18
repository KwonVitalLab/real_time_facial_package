import os
import sys
import subprocess

def main():
    print("Launching Real-time Facial Features Pipeline...")
    script_path = os.path.join(os.path.dirname(__file__), "facial_features.py")
    subprocess.run([sys.executable, script_path])

if __name__ == "__main__":
    main()
