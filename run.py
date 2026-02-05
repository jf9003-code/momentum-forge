#!/usr/bin/env python3
"""
D5 Robust Master - Run Script
Convenience script to start backend and/or frontend
"""
import subprocess
import sys
import argparse
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_backend():
    """Start the FastAPI backend server"""
    print("Starting Backend (FastAPI)...")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd=PROJECT_ROOT)


def run_frontend():
    """Start the Streamlit frontend"""
    print("Starting Frontend (Streamlit)...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ], cwd=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="D5 Robust Master - Run Script"
    )
    parser.add_argument(
        "component",
        choices=["backend", "frontend", "both"],
        help="Component to start: backend, frontend, or both"
    )

    args = parser.parse_args()

    if args.component == "backend":
        run_backend()
    elif args.component == "frontend":
        run_frontend()
    elif args.component == "both":
        print("To run both, please start in separate terminals:")
        print("  Terminal 1: python run.py backend")
        print("  Terminal 2: python run.py frontend")
        print("\nOr use a process manager like supervisord.")


if __name__ == "__main__":
    main()
