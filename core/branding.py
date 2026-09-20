"""Lab identity: logo, lab name and page header shared by every page."""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
LOGO_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".svg")

# Fixed identity: shown on every page and NOT editable inside the app.
# To change it, edit the values below (and put your logo file at assets/logo.png).
DEFAULT_CONFIG = {
    "lab_name": "ARAtronics Research Center",
    "tagline": "GUC - Faculty of Engineering and Material Science",
    "university": "GUC - Faculty of Engineering and Material Science",
    "project_title": "Sensor Characterization and Machine-Learning Studio",
    "student": "",
    "supervisor": "Prof. Amir Roushdy",
}


def load_config() -> dict:
    return dict(DEFAULT_CONFIG)


def find_logo() -> Path:
    """User logo (assets/logo.*) if present, otherwise the built-in placeholder."""
    for ext in LOGO_EXTS:
        p = ASSETS / f"logo{ext}"
        if p.exists():
            return p
    return ASSETS / "logo_placeholder.png"


def has_user_logo() -> bool:
    return any((ASSETS / f"logo{e}").exists() for e in LOGO_EXTS)


def save_uploaded_logo(data: bytes, filename: str) -> Path:
    ext = Path(filename).suffix.lower()
    if ext not in LOGO_EXTS:
        raise ValueError("Logo must be PNG, JPG, WEBP or SVG.")
    for e in LOGO_EXTS:                                  # remove older logos
        old = ASSETS / f"logo{e}"
        if old.exists():
            old.unlink()
    dest = ASSETS / f"logo{ext}"
    dest.write_bytes(data)
    return dest


def remove_user_logo() -> None:
    for e in LOGO_EXTS:
        old = ASSETS / f"logo{e}"
        if old.exists():
            old.unlink()


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    if path.suffix.lower() == ".svg":
        mime = "image/svg+xml"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def header_html(title: str, subtitle: str = "") -> str:
    cfg = load_config()
    logo = data_uri(find_logo())
    sub = f'<div class="lab-sub">{subtitle}</div>' if subtitle else ""
    return f"""
<div class="lab-header">
  <div class="lab-logo-box"><img src="{logo}" alt="lab logo" class="lab-logo"/></div>
  <div class="lab-text">
    <div class="lab-name">{cfg['lab_name']}<span class="lab-tag"> &middot; {cfg['tagline']}</span></div>
    <div class="lab-title">{title}</div>
    {sub}
  </div>
</div>
"""


# Palette: dark red / white / gray / black, Times New Roman.
DARKRED, DARKRED2, TINT = "#8b1a1a", "#5e0f0f", "#f6ecec"
FONT_STACK = "'Times New Roman', Times, 'Liberation Serif', 'Nimbus Roman', serif"

CSS = """
<style>
:root { --red:#8b1a1a; --red2:#5e0f0f; --tint:#f6ecec; --ink:#111111; --ink2:#4d4d4d; --line:#d6d6d6; --panel:#f2f2f2; }
html { font-size: 17.5px; }
.stApp, .stApp div, .stApp p, .stApp label, .stApp li, .stApp a, .stApp button, .stApp input, .stApp textarea,
.stApp select, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp td, .stApp th, .stApp code,
.stApp span:not([data-testid="stIconMaterial"]):not([class*="material"]):not(.katex *) {
  font-family: 'Times New Roman', Times, 'Liberation Serif', 'Nimbus Roman', serif !important; }
.stApp { color: var(--ink); }
.block-container { padding-top: 3.6rem; max-width: 1400px; }
h1, h2, h3, h4 { color: var(--red2) !important; font-weight: 700 !important; letter-spacing: 0 !important; }
h2, h3 { border-bottom: 1px solid var(--line); padding-bottom: 4px; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size: 1.05rem; line-height: 1.5; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, .small-note { color: var(--ink2) !important; opacity:1 !important; font-size: 0.95rem; }

/* header band */
.lab-header { display:flex; align-items:center; gap:22px; padding:18px 24px; border-radius:6px; margin-bottom:18px;
  background:linear-gradient(100deg, var(--red2) 0%, var(--red) 100%); border-bottom:4px solid #111; }
.lab-logo-box { background:#fff; border-radius:6px; padding:8px 12px; display:flex; align-items:center; justify-content:center;
  min-width:84px; }
.lab-logo { height:64px; width:auto; max-width:260px; object-fit:contain; }
.lab-text { min-width:0; }
.lab-name { font-size:1.05rem; letter-spacing:.03em; color:#f0d9d9; font-weight:700; }
.lab-tag { font-weight:400; text-transform:none; letter-spacing:0; color:#e6c6c6; }
.lab-title { font-size:2.1rem; font-weight:700; color:#fff; line-height:1.15; margin-top:2px; }
.lab-sub { color:#f3e3e3; font-size:1.05rem; margin-top:4px; }

/* KPI cards: one grid = equal heights, even gaps */
.kpi-grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; margin:6px 0 16px 0; }
.kpi-grid.c4 { grid-template-columns:repeat(4, minmax(0, 1fr)); }
.kpi-grid.c5 { grid-template-columns:repeat(5, minmax(0, 1fr)); }
.kpi-grid.c5 .k-value { font-size:2rem; }
.kpi-grid.c5 .k-unit { font-size:0.95rem; }
@media (max-width: 900px) { .kpi-grid, .kpi-grid.c4, .kpi-grid.c5 { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
.kpi { border:1px solid var(--line); border-top:5px solid var(--red); border-radius:6px; padding:12px 16px 12px 16px;
  background:#fff; display:flex; flex-direction:column; justify-content:flex-start; min-width:0; box-shadow:0 1px 2px rgba(0,0,0,.06); }
.kpi .k-label { font-size:0.9rem; color:var(--ink2); text-transform:uppercase; letter-spacing:.05em; font-weight:700; min-height:2.5em; line-height:1.25; }
.kpi .k-value { font-size:2.35rem; font-weight:700; color:var(--red); line-height:1.1; white-space:nowrap; margin-top:2px; }
.kpi .k-unit { font-size:1.05rem; color:var(--ink); font-weight:600; margin-left:6px; }
.kpi .k-desc { font-size:0.85rem; color:#5a5a5a; margin-top:8px; padding-top:6px; border-top:1px dashed var(--line); line-height:1.3; font-style:italic; }
.kpi .k-note { font-size:0.9rem; color:var(--ink2); margin-top:6px; line-height:1.25; }
.kpi .k-delta-up { color:#111; font-weight:700; }
.kpi .k-delta-dn { color:var(--red); font-weight:700; }
.problem-box { border-left:5px solid var(--red); background:var(--panel); padding:12px 16px; border-radius:4px; color:var(--ink); font-size:1.05rem; }

/* sidebar */
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { border-bottom:none; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:2px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-size:1.1rem; font-weight:700; padding:8px 16px; color:var(--ink2); }
.stTabs [aria-selected="true"] { color:var(--red) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color:var(--red) !important; height:3px; }

/* buttons */
.stButton > button, .stDownloadButton > button { border-radius:4px; border:1px solid #111; color:#111; background:#fff; font-weight:700; font-size:1rem; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--red); color:var(--red); background:var(--tint); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background:var(--red); border-color:var(--red2); color:#fff; }
.stButton > button[kind="primary"]:hover { background:var(--red2); color:#fff; }

/* alerts: neutral gray with a dark-red edge */
[data-testid="stAlert"] { background:#f4f4f4 !important; border:1px solid var(--line); border-left:5px solid var(--red); border-radius:4px; color:var(--ink); }
[data-testid="stAlert"] * { color:var(--ink) !important; }
[data-testid="stAlert"] [data-testid="stAlertContainer"], [data-testid="stAlertContainer"] { background:#f4f4f4 !important; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:6px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:4px; }
hr { border-color: var(--line); }
.stApp code { color:var(--red2) !important; background:var(--panel) !important; border-radius:3px; }

/* plain HTML tables */
table.lab-table { width:100%; border-collapse:collapse; font-size:0.98rem; margin:6px 0 10px 0; }
table.lab-table th { background:var(--red2); color:#fff; text-align:left; padding:8px 10px; font-weight:700; }
table.lab-table td { padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
table.lab-table tr:nth-child(even) td { background:#f7f7f7; }
table.lab-table td:first-child { font-weight:700; color:var(--red2); white-space:nowrap; }
table.lab-table.vals td:nth-child(2) { font-weight:700; color:var(--red); font-size:1.08rem; white-space:nowrap; text-align:right; }
table.lab-table.vals td:nth-child(n+4) { color:var(--ink2); font-size:0.94rem; }
table.lab-table.vals { table-layout:auto; }
.howto { border-left:4px solid #999; background:#f7f7f7; padding:8px 14px; border-radius:3px; font-size:0.98rem; color:#222; margin:4px 0 12px 0; }
.term-card { border:1px solid var(--line); border-left:5px solid var(--red); border-radius:5px; padding:10px 14px; margin:8px 0; background:#fff; }
.term-card .t-name { font-weight:700; color:var(--red2); font-size:1.12rem; }
.term-card .t-short { font-weight:600; color:#111; margin:2px 0; }
.term-card .t-long { color:#333; }
.term-card .t-formula { margin-top:4px; color:var(--red2); }
.katex-display { font-size:1.4em; margin:0.6em 0; }
.katex-display > .katex { color:var(--red2); }
</style>
"""


def kpi(label: str, value: str, unit: str = "", note: str = "", delta: str = "", delta_up: bool | None = None,
        desc: str | None = None) -> str:
    """One result card. `desc` (or, if omitted, the glossary text for the label) becomes a one-line
    'what is this?' under the number and a hover tooltip on the whole card."""
    from html import escape
    from . import glossary
    d = ""
    if delta:
        cls = "k-delta-up" if delta_up else "k-delta-dn"
        arrow = "&#9650;" if delta_up else "&#9660;"
        word = "better" if delta_up else "worse"
        d = f'<div class="k-note"><span class="{cls}">{arrow} {delta} ({word})</span> vs baseline</div>'
    n = f'<div class="k-note">{note}</div>' if note else ""
    u = f'<span class="k-unit">{unit}</span>' if unit else ""
    text = glossary.short(label) if desc is None else desc
    tip = f' title="{escape(text, quote=True)}"' if text else ""
    dd = f'<div class="k-desc">{escape(text)}</div>' if text else ""
    return f'<div class="kpi"{tip}><div class="k-label">{label}</div><div class="k-value">{value}{u}</div>{d}{n}{dd}</div>'


def kpi_grid(cards: list) -> str:
    """Render several KPI cards as one responsive grid (equal heights, even gaps)."""
    cls = {4: "kpi-grid c4", 5: "kpi-grid c5"}.get(len(cards), "kpi-grid")
    return f'<div class="{cls}">' + "".join(cards) + "</div>"


def html_table(rows: list) -> str:
    """Simple styled HTML table (wraps long text, unlike the canvas-based dataframe)."""
    if not rows:
        return ""
    cols = list(rows[0])
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<table class="lab-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
