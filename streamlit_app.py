"""
Root Streamlit Entry Point for Streamlit Community Cloud and Direct Git URL Execution.
Directs to the Astro Sales Forecasting Inference Platform UI.
"""

import sys
import os
import subprocess
import tempfile

# Determine base directory
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.join(current_dir, "ui")
include_dir = os.path.join(current_dir, "include")

# If run directly via git URL (e.g., streamlit run https://raw.githubusercontent.com/...)
# where only this single file was fetched, automatically clone the full repo
if not os.path.exists(ui_dir):
    cache_dir = os.path.join(tempfile.gettempdir(), "sales_forecast_repo")
    if not os.path.exists(os.path.join(cache_dir, "ui")):
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/BKiran27/sales-forecast-prediction.git", cache_dir],
                check=True,
                capture_output=True
            )
        except Exception as e:
            raise RuntimeError(f"Could not clone repository for remote execution: {e}")
    current_dir = cache_dir
    ui_dir = os.path.join(current_dir, "ui")
    include_dir = os.path.join(current_dir, "include")

# Add paths to sys.path
for path in [current_dir, ui_dir, include_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Change working directory so relative model/data paths resolve
os.chdir(current_dir)

# Run the complete inference app
if __name__ == "__main__" or True:
    app_path = os.path.join(ui_dir, "inference_app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, app_path, 'exec'), globals())
