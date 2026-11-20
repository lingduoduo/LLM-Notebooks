"""Self-contained, escaped HTML execution report; no scripts or remote assets."""
import html
import json
from pathlib import Path

CSS = """
:root { color-scheme:light; --ink:#20344a; --muted:#516579; --blue:#234f88;
        --paper:#ffffff; --wash:#f0f5fa; --line:#c9d7e5; --green:#24664e; }
* { box-sizing:border-box; }
body { margin:0; background:var(--wash); color:var(--ink); font:16px/1.6 'Avenir Next','Segoe UI',sans-serif; }
header { background:var(--blue); color:white; padding:48px max(24px,calc((100vw - 1120px)/2)); }
h1 { font-size:clamp(30px,5vw,48px); line-height:1.2; letter-spacing:-1.2px; margin:0 0 16px; }
header p { max-width:65ch; margin:0; color:#e4edf8; }
main { max-width:1168px; margin:auto; padding:32px 24px 64px; }
nav { display:flex; flex-wrap:wrap; gap:8px 20px; margin-bottom:32px; }
a { color:var(--blue); text-underline-offset:4px; }
a:focus-visible,summary:focus-visible { outline:3px solid #976900; outline-offset:4px; }
section { background:var(--paper); margin:0 0 24px; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
.section-heading { display:flex; align-items:baseline; gap:20px; padding:20px 24px 0; }
.step { font-size:28px; color:var(--green); font-weight:600; }
h2 { font-size:23px; line-height:1.3; margin:0; }
.description { color:var(--muted); margin:8px 24px 20px; max-width:76ch; }
.stats { display:flex; flex-wrap:wrap; gap:12px 32px; padding:0 24px 20px; }
.stats span { color:var(--muted); } .stats strong { color:var(--ink); margin-right:6px; }
details { border-top:1px solid var(--line); padding:14px 24px; }
summary { cursor:pointer; color:var(--blue); font-weight:600; }
pre { font:13px/1.65 'SFMono-Regular',Consolas,monospace; overflow:auto; max-height:520px;
      background:var(--wash); padding:16px; border-radius:3px; tab-size:2; }
.results { padding:0 24px 24px; }
.table-scroll { overflow-x:auto; margin:0 0 24px; }
.table-scroll:last-child { margin-bottom:0; }
table { width:100%; border-collapse:collapse; font-size:14px; }
caption { text-align:left; font-weight:600; font-size:17px; margin:0 0 10px; }
th { text-align:left; background:var(--wash); font-weight:600; }
th,td { padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top;
        overflow-wrap:anywhere; white-space:pre-line; min-width:110px; }
tr:last-child td { border-bottom:0; }
.empty { color:var(--muted); }
footer { max-width:76ch; color:var(--muted); font-size:14px; }
@media(max-width:560px) { header { padding:28px 20px; } main { padding:20px 12px; }
.section-heading { gap:12px; padding:16px 16px 0; } details { padding:12px 16px; }
.description { margin:8px 16px 16px; } .results { padding:0 16px 16px; } .stats { padding:0 16px 16px; } }
@media print { body { background:white; } header { background:white; color:var(--ink); }
header p { color:var(--muted); } pre { max-height:none; white-space:pre-wrap; } }
"""



def render_table(table):
    """Keep actual results visible; escape every caption, header and cell."""
    def escape(value):
        return html.escape(str(value))
    headers = ''.join(f'<th scope="col">{escape(value)}</th>' for value in table['headers'])
    rows = ''.join('<tr>' + ''.join(f'<td>{escape(value)}</td>' for value in row) + '</tr>'
                   for row in table['rows'])
    if not rows:
        rows = f'<tr><td colspan="{len(table["headers"])}" class="empty">No results for this stage.</td></tr>'
    return ('<div class="table-scroll"><table><caption>' + escape(table['caption']) +
            '</caption><thead><tr>' + headers + '</tr></thead><tbody>' + rows + '</tbody></table></div>')


def render_report(stages: list[dict], output: Path) -> Path:
    escape = html.escape
    navigation, sections = [], []
    for number, stage in enumerate(stages, 1):
        title = escape(str(stage['title']))
        navigation.append(f'<a href="#stage-{number}">{number}. {title}</a>')
        description = escape(str(stage.get('description', '')))
        stats = ''.join(f'<span><strong>{escape(str(value))}</strong>{escape(str(label))}</span>'
                        for label, value in stage.get('stats', {}).items())
        tables = ''.join(render_table(table) for table in stage.get('tables', []))
        data = escape(json.dumps(stage.get('data', {}), ensure_ascii=False, indent=2))
        sections.append(f'<section id="stage-{number}"><div class="section-heading">'
                        f'<span class="step">{number}</span><h2>{title}</h2></div>'
                        f'<p class="description">{description}</p><div class="stats">{stats}</div>'
                        f'<div class="results">{tables}</div>'
                        '<details><summary>Raw JSON and execution data</summary>'
                        f'<pre>{data}</pre></details></section>')
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Ontology workflow report</title><style>' + CSS + '</style></head><body>'
                '<header><h1>From taxonomy to action</h1><p>A six-stage room ontology workflow. '
                'Follow the evidence from mapped records to classification and a sandbox review task.</p></header>'
                '<main><nav aria-label="Workflow stages">' + ''.join(navigation) + '</nav>' + ''.join(sections) +
                '<footer>This offline demonstration uses the existing room taxonomy and classifier. '
                'Sandbox changes stay in the output artifacts. Tables show the computed results. Expand raw JSON to inspect the full inputs, '
                'evidence, or execution trace.</footer></main></body></html>')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding='utf-8')
    return output
