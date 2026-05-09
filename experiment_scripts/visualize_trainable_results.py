from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def project_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def short_text(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".html")


def normalize_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not raw_rows:
        raise ValueError("CSV contains no rows.")

    fieldnames = set(raw_rows[0])
    if {"response_1", "response_2", "baseline_response"}.issubset(fieldnames):
        return [
            {
                "sae_path": row.get("sae_path", ""),
                "prompt_index": row.get("prompt_index", ""),
                "prompt": row.get("prompt", ""),
                "experiment": row.get("experiment", ""),
                "variable_1": row.get("variable_1", ""),
                "variable_2": row.get("variable_2", ""),
                "steering_method": row.get("steering_method", ""),
                "factor": row.get("factor", ""),
                "baseline_response": row.get("baseline_response", ""),
                "response_1": row.get("response_1", ""),
                "response_2": row.get("response_2", ""),
                "error": row.get("error", ""),
            }
            for row in raw_rows
        ]

    if {"direction", "output_prompt", "baseline_output"}.issubset(fieldnames):
        return normalize_direction_rows(raw_rows)

    missing = "response_1/response_2/baseline_response"
    raise ValueError(f"Unrecognized trainable-results CSV schema; expected {missing}.")


def normalize_direction_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in raw_rows:
        key = (
            row.get("sae_path", ""),
            row.get("prompt_index", ""),
            row.get("input_prompt", row.get("prompt", "")),
            row.get("experiment", ""),
            row.get("steering_method", ""),
            row.get("factor", ""),
        )
        item = grouped.setdefault(
            key,
            {
                "sae_path": row.get("sae_path", ""),
                "prompt_index": row.get("prompt_index", ""),
                "prompt": row.get("input_prompt", row.get("prompt", "")),
                "experiment": row.get("experiment", ""),
                "variable_1": row.get("positive_label", ""),
                "variable_2": row.get("negative_label", ""),
                "steering_method": row.get("steering_method", ""),
                "factor": row.get("factor", ""),
                "baseline_response": row.get("baseline_output", ""),
                "response_1": "",
                "response_2": "",
                "error": row.get("error", ""),
            },
        )
        direction = row.get("direction", "")
        if direction == "positive":
            item["response_1"] = row.get("output_prompt", "")
        elif direction == "negative":
            item["response_2"] = row.get("output_prompt", "")
        if row.get("direction_label") and direction == "positive":
            item["variable_1"] = row["direction_label"]
        elif row.get("direction_label") and direction == "negative":
            item["variable_2"] = row["direction_label"]
        if row.get("error"):
            item["error"] = row["error"]
    return list(grouped.values())


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return normalize_rows(list(reader))


def html_template(title: str, csv_label: str, rows: list[dict[str, Any]]) -> str:
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    title_json = json.dumps(title)
    csv_json = json.dumps(csv_label)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201b;
      --muted: #607066;
      --line: #d9e1dc;
      --panel: #ffffff;
      --page: #f4f7f5;
      --green: #1b7f5f;
      --green-dark: #0d5f47;
      --red: #a03a42;
      --shadow: 0 8px 22px rgba(27, 45, 35, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--page);
      color: var(--ink);
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 700;
    }}
    .status {{
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }}
    main {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: calc(100vh - 54px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: #fbfcfb;
      padding: 16px;
      overflow: auto;
    }}
    section.workspace {{
      padding: 16px;
      display: grid;
      grid-template-rows: auto auto minmax(260px, 1fr);
      gap: 14px;
      overflow: auto;
    }}
    .group, .output, .table-panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .group {{
      padding: 12px;
      margin-bottom: 12px;
    }}
    .group h2, .table-panel h2 {{
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0;
    }}
    label {{
      display: block;
      font-size: 13px;
      font-weight: 650;
      margin: 10px 0 5px;
    }}
    input, select, textarea, button {{
      font: inherit;
    }}
    select {{
      width: 100%;
      border: 1px solid #c9d4ce;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px 9px;
      font-size: 14px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .meta-item {{
      border: 1px solid #edf1ee;
      border-radius: 6px;
      background: #fbfcfb;
      padding: 9px 10px;
      min-width: 0;
    }}
    .meta-item b {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }}
    .meta-item span {{
      display: block;
      overflow-wrap: anywhere;
      font-size: 13px;
      font-weight: 650;
    }}
    .outputs {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .output {{
      min-height: 220px;
      overflow: hidden;
    }}
    .output h2 {{
      margin: 0;
      padding: 10px 12px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
      background: #f9fbfa;
    }}
    pre {{
      margin: 0;
      padding: 12px;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
    }}
    .table-panel {{
      overflow: hidden;
    }}
    .table-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #f9fbfa;
    }}
    .table-scroll {{
      overflow: auto;
      max-height: 420px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #edf1ee;
      padding: 7px 9px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #fbfcfb;
      color: var(--muted);
      position: sticky;
      top: 0;
    }}
    tr.active {{
      background: #e6f1ed;
    }}
    .hint {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .error {{
      color: var(--red);
      font-weight: 650;
    }}
    @media (max-width: 1000px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .outputs, .meta {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1 id="title"></h1>
    <div class="status" id="status"></div>
  </header>
  <main>
    <aside>
      <div class="group">
        <h2>Result Set</h2>
        <div class="hint" id="sourceFile"></div>
        <div class="hint" id="rowCount"></div>
      </div>
      <div class="group">
        <h2>Filters</h2>
        <label for="promptSelect">Prompt</label>
        <select id="promptSelect"></select>
        <label for="saeSelect">SAE</label>
        <select id="saeSelect"></select>
        <label for="experimentSelect">Experiment</label>
        <select id="experimentSelect"></select>
        <label for="methodSelect">Method</label>
        <select id="methodSelect"></select>
        <label for="factorSelect">Factor</label>
        <select id="factorSelect"></select>
        <div class="hint">Selections are populated from the CSV. The page shows the first matching run.</div>
      </div>
    </aside>
    <section class="workspace">
      <div class="group">
        <h2>Selected Prompt</h2>
        <pre id="promptText"></pre>
        <div class="hint error" id="errorText"></div>
      </div>
      <div class="meta">
        <div class="meta-item"><b>SAE</b><span id="metaSae"></span></div>
        <div class="meta-item"><b>Experiment</b><span id="metaExperiment"></span></div>
        <div class="meta-item"><b>Variables</b><span id="metaVariables"></span></div>
        <div class="meta-item"><b>Method</b><span id="metaMethod"></span></div>
      </div>
      <div class="outputs">
        <div class="output">
          <h2>Baseline</h2>
          <pre id="baselineOut"></pre>
        </div>
        <div class="output">
          <h2 id="response1Title">Response 1</h2>
          <pre id="response1Out"></pre>
        </div>
        <div class="output">
          <h2 id="response2Title">Response 2</h2>
          <pre id="response2Out"></pre>
        </div>
      </div>
      <div class="table-panel">
        <div class="table-header">
          <h2>Matching Rows</h2>
          <span class="status" id="matchCount"></span>
        </div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Prompt</th>
                <th>SAE</th>
                <th>Experiment</th>
                <th>Method</th>
                <th>Factor</th>
                <th>Variables</th>
              </tr>
            </thead>
            <tbody id="matchRows"></tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
  <script>
    const DATA = {data};
    const TITLE = {title_json};
    const CSV_LABEL = {csv_json};
    const $ = (id) => document.getElementById(id);
    const selects = {{
      prompt: $("promptSelect"),
      sae: $("saeSelect"),
      experiment: $("experimentSelect"),
      method: $("methodSelect"),
      factor: $("factorSelect"),
    }};

    function normalize(value) {{
      return String(value ?? "");
    }}

    function promptKey(row) {{
      return `${{normalize(row.prompt_index)}}::${{normalize(row.prompt)}}`;
    }}

    function promptLabel(row) {{
      return `${{normalize(row.prompt_index)}} | ${{shorten(row.prompt, 92)}}`;
    }}

    function shorten(value, limit) {{
      const text = normalize(value).replace(/\\s+/g, " ").trim();
      return text.length <= limit ? text : text.slice(0, limit - 1).trimEnd() + "...";
    }}

    function uniqueOptions(rows, getter, labeler) {{
      const seen = new Set();
      const options = [];
      for (const row of rows) {{
        const value = getter(row);
        if (seen.has(value)) continue;
        seen.add(value);
        options.push({{ value, label: labeler ? labeler(row) : value }});
      }}
      return options;
    }}

    function setOptions(select, options, previous) {{
      select.innerHTML = "";
      for (const option of options) {{
        const node = document.createElement("option");
        node.value = option.value;
        node.textContent = option.label || option.value || "(empty)";
        select.appendChild(node);
      }}
      if (options.some(option => option.value === previous)) {{
        select.value = previous;
      }} else if (options.length) {{
        select.value = options[0].value;
      }}
    }}

    function currentFilters() {{
      return {{
        prompt: selects.prompt.value,
        sae: selects.sae.value,
        experiment: selects.experiment.value,
        method: selects.method.value,
        factor: selects.factor.value,
      }};
    }}

    function matches(row, filters, through = "factor") {{
      if (filters.prompt && promptKey(row) !== filters.prompt) return false;
      if (through === "prompt") return true;
      if (filters.sae && normalize(row.sae_path) !== filters.sae) return false;
      if (through === "sae") return true;
      if (filters.experiment && normalize(row.experiment) !== filters.experiment) return false;
      if (through === "experiment") return true;
      if (filters.method && normalize(row.steering_method) !== filters.method) return false;
      if (through === "method") return true;
      if (filters.factor && normalize(row.factor) !== filters.factor) return false;
      return true;
    }}

    function refreshOptions(changed) {{
      const previous = currentFilters();
      if (!changed || changed === "prompt") {{
        setOptions(selects.prompt, uniqueOptions(DATA, promptKey, promptLabel), previous.prompt);
      }}
      const afterPrompt = {{ ...previous, prompt: selects.prompt.value }};
      if (!changed || changed === "prompt" || changed === "sae") {{
        setOptions(
          selects.sae,
          uniqueOptions(DATA.filter(row => matches(row, afterPrompt, "prompt")), row => normalize(row.sae_path)),
          previous.sae
        );
      }}
      const afterSae = {{ ...afterPrompt, sae: selects.sae.value }};
      if (!changed || ["prompt", "sae", "experiment"].includes(changed)) {{
        setOptions(
          selects.experiment,
          uniqueOptions(DATA.filter(row => matches(row, afterSae, "sae")), row => normalize(row.experiment), row => normalize(row.experiment).replace("_", " / ")),
          previous.experiment
        );
      }}
      const afterExperiment = {{ ...afterSae, experiment: selects.experiment.value }};
      if (!changed || ["prompt", "sae", "experiment", "method"].includes(changed)) {{
        setOptions(
          selects.method,
          uniqueOptions(DATA.filter(row => matches(row, afterExperiment, "experiment")), row => normalize(row.steering_method)),
          previous.method
        );
      }}
      const afterMethod = {{ ...afterExperiment, method: selects.method.value }};
      setOptions(
        selects.factor,
        uniqueOptions(DATA.filter(row => matches(row, afterMethod, "method")), row => normalize(row.factor)),
        previous.factor
      );
    }}

    function selectedRows() {{
      return DATA.filter(row => matches(row, currentFilters()));
    }}

    function render() {{
      const rows = selectedRows();
      const row = rows[0] || null;
      $("matchCount").textContent = `${{rows.length}} match${{rows.length === 1 ? "" : "es"}}`;
      if (!row) {{
        $("promptText").textContent = "No matching row.";
        $("baselineOut").textContent = "";
        $("response1Out").textContent = "";
        $("response2Out").textContent = "";
        $("errorText").textContent = "";
        return;
      }}
      $("promptText").textContent = normalize(row.prompt);
      $("errorText").textContent = normalize(row.error);
      $("metaSae").textContent = normalize(row.sae_path);
      $("metaExperiment").textContent = normalize(row.experiment).replace("_", " / ");
      $("metaVariables").textContent = `${{normalize(row.variable_1)}} / ${{normalize(row.variable_2)}}`;
      $("metaMethod").textContent = `${{normalize(row.steering_method)}} factor=${{normalize(row.factor)}}`;
      $("baselineOut").textContent = normalize(row.baseline_response) || "(empty)";
      $("response1Title").textContent = `+ ${{capitalize(row.variable_1)}}`;
      $("response2Title").textContent = `- ${{capitalize(row.variable_2)}}`;
      $("response1Out").textContent = normalize(row.response_1) || "(empty)";
      $("response2Out").textContent = normalize(row.response_2) || "(empty)";
      renderTable(rows, row);
    }}

    function renderTable(rows, activeRow) {{
      $("matchRows").innerHTML = rows.map((row, index) => `
        <tr class="${{row === activeRow ? "active" : ""}}">
          <td>${{escapeHtml(shorten(row.prompt, 70))}}</td>
          <td>${{escapeHtml(shorten(row.sae_path, 62))}}</td>
          <td>${{escapeHtml(normalize(row.experiment).replace("_", " / "))}}</td>
          <td>${{escapeHtml(row.steering_method)}}</td>
          <td>${{escapeHtml(row.factor)}}</td>
          <td>${{escapeHtml(row.variable_1)}} / ${{escapeHtml(row.variable_2)}}</td>
        </tr>
      `).join("");
    }}

    function capitalize(value) {{
      const text = normalize(value);
      return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Response";
    }}

    function escapeHtml(value) {{
      return normalize(value).replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }}[char]));
    }}

    for (const [name, select] of Object.entries(selects)) {{
      select.addEventListener("change", () => {{
        refreshOptions(name);
        render();
      }});
    }}

    $("title").textContent = TITLE;
    $("sourceFile").textContent = CSV_LABEL;
    $("rowCount").textContent = `${{DATA.length}} row${{DATA.length === 1 ? "" : "s"}} loaded`;
    $("status").textContent = `${{DATA.length}} trainable result row${{DATA.length === 1 ? "" : "s"}}`;
    refreshOptions();
    render();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interactive HTML viewer for trainable-results CSVs.")
    parser.add_argument("csv_file", type=Path, help="CSV file from run_trainable_batch_results.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to the CSV path with .html suffix.",
    )
    parser.add_argument("--title", default="Trainable SAE Results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv_file.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(csv_path)
    )
    rows = load_rows(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        html_template(args.title, project_path(csv_path), rows),
        encoding="utf-8",
    )
    print(f"Loaded {len(rows)} row(s) from {csv_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
