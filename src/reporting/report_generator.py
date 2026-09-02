from datetime import datetime
from typing import Dict, Any, List
import json
from src.utils.helpers import NpEncoder

def generate_html_report(
    dataset_name: str,
    before_profile: Dict[str, Any],
    after_profile: Dict[str, Any],
    before_score: Dict[str, Any],
    after_score: Dict[str, Any],
    audit_log: Dict[str, Any]
) -> str:
    """
    Generates a standalone, beautiful HTML audit and data quality report.
    """
    b_score = before_score.get("overall_score", 0.0)
    a_score = after_score.get("overall_score", 0.0)
    delta_score = round(a_score - b_score, 2)
    delta_sign = "+" if delta_score >= 0 else ""

    actions_summary = audit_log.get("summary_by_action", {})
    action_rows = "".join([
        f"<tr><td><code>{act}</code></td><td><strong>{cnt}</strong></td></tr>"
        for act, cnt in actions_summary.items()
    ]) or "<tr><td colspan='2'>No actions logged</td></tr>"

    dim_b = before_score.get("dimensions", {})
    dim_a = after_score.get("dimensions", {})

    dim_rows = ""
    for dim_name in ["completeness", "consistency", "validity", "uniqueness", "anomaly_quality"]:
        vb = dim_b.get(dim_name, 0.0)
        va = dim_a.get(dim_name, 0.0)
        diff = round(va - vb, 2)
        diff_str = f"+{diff}%" if diff >= 0 else f"{diff}%"
        color = "#10b981" if diff >= 0 else "#ef4444"
        dim_rows += f"""
        <tr>
            <td style="text-transform: capitalize;"><strong>{dim_name.replace('_', ' ')}</strong></td>
            <td>{vb}%</td>
            <td><strong>{va}%</strong></td>
            <td style="color: {color}; font-weight: bold;">{diff_str}</td>
        </tr>
        """

    recent_actions = audit_log.get("applied_actions", [])[:25]
    audit_table_rows = ""
    for act in recent_actions:
        audit_table_rows += f"""
        <tr>
            <td>{act.get('row', '—')}</td>
            <td><strong>{act.get('column', '—')}</strong></td>
            <td><code>{str(act.get('original_value', ''))[:20]}</code></td>
            <td><span class="badge badge-success">{act.get('action')}</span></td>
            <td><code>{str(act.get('corrected_value', ''))[:20]}</code></td>
            <td>{act.get('confidence', 1.0)}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Audit Report - {dataset_name}</title>
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-light: #6366f1;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            margin-bottom: 30px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 20px;
        }}
        .header h1 {{
            font-size: 28px;
            margin: 0 0 10px 0;
            color: #ffffff;
            font-weight: 700;
        }}
        .header p {{
            color: var(--text-muted);
            margin: 0;
            font-size: 14px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .card h3 {{
            margin-top: 0;
            font-size: 18px;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 12px;
            color: #ffffff;
        }}
        .score-display {{
            display: flex;
            align-items: center;
            justify-content: space-around;
            text-align: center;
            padding: 10px 0;
        }}
        .score-box {{
            font-size: 36px;
            font-weight: 800;
        }}
        .score-before {{
            color: var(--warning);
        }}
        .score-after {{
            color: var(--success);
        }}
        .delta {{
            font-size: 18px;
            font-weight: 600;
            color: var(--success);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--card-border);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
        }}
        code {{
            background: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 13px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-success {{
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
        }}
        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 50px;
            border-top: 1px solid var(--card-border);
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Data Quality & Cleaning Audit Report</h1>
            <p>Dataset: <strong>{dataset_name}</strong> | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Quality Score Progression</h3>
                <div class="score-display">
                    <div>
                        <div class="score-box score-before">{b_score}</div>
                        <div style="color: var(--text-muted); font-size: 12px;">Before Cleaning</div>
                    </div>
                    <div style="font-size: 24px; color: var(--text-muted);">➔</div>
                    <div>
                        <div class="score-box score-after">{a_score}</div>
                        <div style="color: var(--text-muted); font-size: 12px;">After Cleaning</div>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    Net Improvement: <span class="delta">{delta_sign}{delta_score} Points</span>
                </div>
            </div>

            <div class="card">
                <h3>Pipeline Execution Summary</h3>
                <table>
                    <thead><tr><th>Operation</th><th>Count</th></tr></thead>
                    <tbody>{action_rows}</tbody>
                </table>
            </div>
        </div>

        <div class="card" style="margin-bottom: 30px;">
            <h3>Quality Dimensions Breakdown</h3>
            <table>
                <thead><tr><th>Quality Dimension</th><th>Before Cleaning</th><th>After Cleaning</th><th>Delta</th></tr></thead>
                <tbody>{dim_rows}</tbody>
            </table>
        </div>

        <div class="card">
            <h3>Audit Trail (Sample of Operations)</h3>
            <table>
                <thead>
                    <tr><th>Row</th><th>Column</th><th>Original</th><th>Action</th><th>Corrected</th><th>Confidence</th></tr>
                </thead>
                <tbody>{audit_table_rows}</tbody>
            </table>
        </div>

        <div class="footer">
            Intelligent Dataset Quality Assessment & Automated Cleaning System • Production-Style Modular Engine
        </div>
    </div>
</body>
</html>
"""
    return html
