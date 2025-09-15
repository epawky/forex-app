#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).parent.resolve()
VENV = ROOT / ".venv"
IS_WIN = (os.name == "nt")

PY_EXE = sys.executable  # current python
if not PY_EXE:
    print("ERROR: No Python interpreter detected. Please install Python 3.11+ and re-run.")
    sys.exit(1)

def run(cmd, **kwargs):
    print(f"> {' '.join(map(str, cmd))}")
    r = subprocess.run(cmd, cwd=str(ROOT), **kwargs)
    if r.returncode != 0:
        sys.exit(r.returncode)

def venv_python():
    if IS_WIN:
        return str(VENV / "Scripts" / "python.exe")
    return str(VENV / "bin" / "python")

def venv_pip():
    if IS_WIN:
        return str(VENV / "Scripts" / "pip.exe")
    return str(VENV / "bin" / "pip")

def ensure_env_file():
    src = ROOT / ".env.example"
    dst = ROOT / ".env"
    if not dst.exists() and src.exists():
        shutil.copyfile(src, dst)
        print("Created .env from .env.example. Fill POLYGON_API_KEY in .env as needed.")

def ensure_dirs():
    (ROOT / "data" / "ticks").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)

def read_env_port():
    # fallback if .env missing or no var set
    port = "5001"
    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.split("=", 1)[0].strip() == "FLASK_RUN_PORT":
                    port = line.split("=", 1)[1].strip()
                    break
        except Exception:
            pass
    return int(port or "5001")

def main():
    print("=== Forex App Bootstrap ===")
    print(f"Python: {PY_EXE}")
    print(f"Root:   {ROOT}")

    # 1) Create venv if missing
    if not VENV.exists():
        print("Creating virtual environment...")
        run([PY_EXE, "-m", "venv", str(VENV)])

    py = venv_python()
    pip = venv_pip()

    # 2) Upgrade pip and install requirements
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    req = ROOT / "requirements.txt"
    if req.exists():
        run([pip, "install", "-r", str(req)])
    else:
        print("WARNING: requirements.txt not found. Skipping package install.")

    # 3) Prepare environment and folders
    ensure_env_file()
    ensure_dirs()

    # 4) Show quick note if ui/dist missing
    ui_dist = ROOT / "ui" / "dist"
    if not ui_dist.exists():
        print(dedent("""
        NOTE: ui/dist not found.
        If you expect the backend to serve the UI (SERVE_UI=1 in .env),
        make sure you built the frontend and included ui/dist in the package.
        """).strip())

    # 5) Run the app
    port = read_env_port()
    print(f"Starting app on http://127.0.0.1:{port}/")
    env = os.environ.copy()
    # Ensure Flask uses .env values; app.py already reads .env via python-dotenv
    run([py, "app.py"], env=env)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

