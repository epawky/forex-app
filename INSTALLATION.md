
FOREX APP – INSTALLATION & USAGE (Client Hand‑Off)

==================================================
Overview
==================================================
This package contains:
- backend/          (Python Flask server + pipeline)
- ui/dist/          (prebuilt web UI served by Flask)
- data/             (created on first run; tick CSVs live here)
- models/           (trained models + manifests + policies)
- .env.example      (template env file)
- bootstrap.py      (one‑shot installer/runner)
- requirements.txt  (Python dependencies)

You do NOT need Node/npm. The UI is already built.

Supported OS:
- Windows 10/11 (x64)
- macOS 12+ (Intel or Apple Silicon)

Python:
- Python 3.11+ is required.
  • Windows: https://www.python.org/downloads/  (check “Add to PATH” during install)
  • macOS:   brew install python@3.11  (or install from python.org)


==================================================
Quick Start (First‑Time Setup)
==================================================
1) Unzip the package to a simple path
   • Windows: C:\forex-app
   • macOS:   ~/forex-app

2) Open a terminal in that folder
   • Windows: Right‑click > “Open in Terminal” (PowerShell)
   • macOS:   Terminal > cd ~/forex-app

3) Run the bootstrap
   • Windows (PowerShell):
       python .\bootstrap.py
   • macOS (Terminal):
       python3 ./bootstrap.py

What bootstrap.py does:
  - Creates a local virtual environment (.venv/)
  - Installs requirements
  - Copies .env.example → .env (if missing)
  - Verifies environment and starts the app


==================================================
Open the App
==================================================
Once the server starts, open your browser to:
  http://127.0.0.1:5001/

(If you change the port in .env, use that port instead.)


==================================================
Configuration (.env)
==================================================
The bootstrap creates a .env if missing. Edit it to configure:
  SERVE_UI=1
  FLASK_RUN_PORT=5001
  POLYGON_API_KEY=your_polygon_key_here
  # Optional MetaTrader 5 (Windows only)
  MT5_USE_SUBPROCESS=false
  MT5_PYTHON=
  MT5_SCRIPT=

Notes:
- POLYGON_API_KEY is required to download historical FX bars from Polygon.
- Change FLASK_RUN_PORT if 5001 is already used on your machine.
- MT5 integration (tick fetch) is Windows‑only. Predictions & Polygon work on macOS.


==================================================
Everyday Workflow (UI)
==================================================
In the UI you can:
- Set Polygon API key (stores locally and server‑side)
- Download Data (Polygon): choose universe, minute bars are appended to data/ticks/
- Fetch MT5 Ticks (Windows): pulls recent ticks into data/ticks/
- Build: ingest ticks → features/labels → dataset parquet
- Train: trains models; saves manifests & policies under models/
- Predict: select ticker & objective (acc/mcc/pnl) to see snapshot and 24h overlay

The price chart shows the last N hours and the prediction overlay (flat by design: one probability per window repeated over time until rolling inference is enabled).


==================================================
Command‑Line Snippets (Optional)
==================================================
Health:
  curl -s http://127.0.0.1:5001/api/health

Predict (snapshot):
  curl -s -X POST "http://127.0.0.1:5001/api/predict?objective=acc" ^
       -H "Content-Type: application/json" ^
       -d "{\"symbol\":\"EURUSD\"}"

Recent ticks (48h):
  curl -s "http://127.0.0.1:5001/api/ticks/recent?symbol=EURUSD&hours=48"


==================================================
Updating: “Models‑Only” Package
==================================================
You may receive models-update-YYYYMMDD.zip with updated models.

To apply:
1) Stop the running app (Ctrl+C in the server terminal).
2) Unzip the archive and replace your existing models/ folder.
3) Start again:
   • Windows: python .\bootstrap.py
   • macOS:   python3 ./bootstrap.py

No code reinstall needed.


==================================================
Offline / Restricted Environments (Optional)
==================================================
If the target machine is offline:
- Prewarm wheels: run bootstrap once on a connected machine to build .venv.
- Zip and transfer the whole project folder, including .venv/ (same OS/arch).
- Update .env on target machine and start:
    Windows: .\.venv\Scripts\python.exe app.py
    macOS:   ./.venv/bin/python3 app.py


==================================================
Troubleshooting
==================================================
• Port already in use
  - Edit .env → FLASK_RUN_PORT=5002 (or other), rerun bootstrap.py

• “Python not found” (Windows)
  - Reinstall Python 3.11+ from python.org and check “Add to PATH”
  - Then: python --version  (should show 3.11+)

• Blank chart / empty ticks
  - Ensure data/ticks/ contains CSVs (use “Download Data” or “Fetch MT5 Ticks”)
  - Check the server logs in the terminal for errors

• Polygon 401/403
  - Verify POLYGON_API_KEY in .env, and re‑set via the UI button if needed

• macOS MetaTrader
  - MT5 integration is Windows‑only. Use Polygon for data on macOS.

• Dependency issues
  - Delete the .venv/ folder, rerun bootstrap.py

• Antivirus/Firewall prompts (Windows)
  - Allow Python for local loopback (127.0.0.1). No external exposure by default.


==================================================
Uninstall
==================================================
Just delete the project folder. No system‑wide installs are made.
