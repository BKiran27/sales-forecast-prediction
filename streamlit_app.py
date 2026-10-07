"""
Root Streamlit Entry Point for Streamlit Community Cloud and Local Execution.
Directs to the Astro Sales Forecasting Inference Platform UI.
"""

import sys
import os

# Set paths so ui and include modules resolve cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
ui_dir = os.path.join(current_dir, "ui")
include_dir = os.path.join(current_dir, "include")

for path in [current_dir, ui_dir, include_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Run the complete inference app
if __name__ == "__main__" or True:
    app_path = os.path.join(ui_dir, "inference_app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, app_path, 'exec'))
