from pathlib import Path
import joblib
import datetime
import platform
import html

base = Path(__file__).resolve().parent
models_dir = base / 'models'
model_file = models_dir / 'best_model.joblib'
out_file = models_dir / 'report.html'

if not model_file.exists():
    print('Error: model file not found:', model_file)
    raise SystemExit(1)

data = joblib.load(model_file)
reports = data.get('reports', {})
best_name = None
if isinstance(data, dict) and 'model' in data:
    # models/best_model.joblib stores {'model': model, 'reports': reports}
    for k in reports.keys():
        if reports[k].get('accuracy') is not None:
            if best_name is None or reports[k]['accuracy'] > reports.get(best_name, {}).get('accuracy', -1):
                best_name = k

now = datetime.datetime.utcnow().isoformat() + 'Z'
py = platform.python_version()

html_parts = []
html_parts.append('<!doctype html>')
html_parts.append('<html><head><meta charset="utf-8"><title>Model Report</title>')
html_parts.append('<style>body{font-family:Arial,Helvetica,sans-serif;padding:20px} h1{color:#333} table{border-collapse:collapse;width:100%;margin-bottom:20px} th,td{border:1px solid #ddd;padding:8px} th{background:#f4f4f4;text-align:left} pre{background:#f8f8f8;padding:10px;border-radius:4px;overflow:auto}</style>')
html_parts.append('</head><body>')
html_parts.append(f'<h1>Model Training Report</h1>')
html_parts.append(f'<p><strong>Generated:</strong> {html.escape(now)}</p>')
html_parts.append(f'<p><strong>Python:</strong> {html.escape(py)}</p>')

html_parts.append('<h2>Models & Metrics</h2>')
if not reports:
    html_parts.append('<p>No reports available in the joblib file.</p>')
else:
    html_parts.append('<table>')
    html_parts.append('<tr><th>Model</th><th>Accuracy</th><th>Classification report</th><th>Confusion matrix</th></tr>')
    for name, r in reports.items():
        acc = r.get('accuracy')
        rep = r.get('report')
        rep_escaped = html.escape(rep) if rep is not None else ''
        cm_path = models_dir / f'confusion_{name}.png'
        if cm_path.exists():
            cm_html = f'<img src="{cm_path.name}" alt="confusion_{name}" style="max-width:350px">'
        else:
            cm_html = '<em>Not found</em>'
        html_parts.append('<tr>')
        html_parts.append(f'<td>{html.escape(name)}</td>')
        html_parts.append(f'<td>{acc:.4f}' if acc is not None else '<td>n/a')
        html_parts.append(f'<td><pre>{rep_escaped}</pre></td>')
        html_parts.append(f'<td>{cm_html}</td>')
        html_parts.append('</tr>')
    html_parts.append('</table>')

html_parts.append('<h2>Notes</h2>')
html_parts.append('<ul>')
html_parts.append(f'<li>Best model (by saved reports): <strong>{html.escape(best_name or "unknown")}</strong></li>')
html_parts.append('<li>Images and models are relative to the <code>models/</code> directory.</li>')
html_parts.append('</ul>')

html_parts.append('<p>To view this report, open the file in a web browser.</p>')
html_parts.append('</body></html>')

out_file.write_text('\n'.join(html_parts), encoding='utf-8')
print('Saved HTML report to', out_file)
