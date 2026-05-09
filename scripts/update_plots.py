"""
Bake assessed_scores.csv directly into feature_steering_experiment_charts.html
so the page loads without any file upload or local server.

Usage:
    uv run scripts/update_plots.py
"""

import json
import sys
from pathlib import Path

HTML   = Path("web/feature_steering_experiment_charts.html")
CSV    = Path("experiment_scripts/experiment_data/grid_search/assessed_scores.csv")
MARKER = "// __BAKED_CSV__"

if not CSV.exists():
    sys.exit(f"CSV not found: {CSV}\nRun the grid search and process-results first.")

if not HTML.exists():
    sys.exit(f"HTML not found: {HTML}")

csv_text = CSV.read_text(encoding="utf-8")
n_rows   = max(0, csv_text.count("\n") - 1)  # subtract header

# json.dumps escapes \n, ", etc.
# Also escape <script / </script so the HTML parser can't close the <script> tag
# mid-string.  <\/ and <\x73 evaluate to </ and <s at runtime.
csv_json = (
    json.dumps(csv_text)
    .replace("<script",  "<\\x73cript")
    .replace("</script", "<\\/script")
)

lines = HTML.read_text(encoding="utf-8").splitlines(keepends=True)

# Find the FIRST line that starts the BAKED_CSV declaration and the line that
# holds the marker.  If a previous broken run left a multi-line raw-CSV block
# between them, we replace the entire range so the file is left clean.
start_idx = end_idx = None
for i, line in enumerate(lines):
    if start_idx is None and line.lstrip().startswith("const BAKED_CSV"):
        start_idx = i
    if MARKER in line:
        end_idx = i
        break

if end_idx is None:
    sys.exit(f"Marker '{MARKER}' not found in {HTML}. Was it removed?")

if start_idx is None or start_idx > end_idx:
    start_idx = end_idx  # marker and declaration were already on one line

ending   = "\r\n" if lines[end_idx].endswith("\r\n") else "\n"
new_line = f"const BAKED_CSV = {csv_json}; {MARKER}{ending}"
lines[start_idx : end_idx + 1] = [new_line]

HTML.write_text("".join(lines), encoding="utf-8")
print(f"Done — embedded {n_rows} rows ({len(csv_text):,} chars) into {HTML}")
