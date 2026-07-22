"""Professional UI styling for the Streamlit dashboard.

Card treatment (translucent glass, hairline borders, soft layered shadow)
and Lucide-style line icons are adapted from patterns the user liked
elsewhere. The accent and status colors are this project's own — the
validated colorblind-safe dark-mode palette from the dataviz skill — not
matched to any other project's brand color.
"""

import urllib.parse

import streamlit as st

# --- Design tokens ---
ACCENT = "#3987e5"
ACCENT_HOVER = "#5598e7"
BG_BASE = "#0d0d0d"
BG_SURFACE = "rgba(26, 26, 25, 0.85)"
BORDER = "#2c2c2a"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
TEXT_MUTED = "#898781"

# Lucide icon inner paths (24x24 viewBox, stroke = currentColor).
_ICON_PATHS = {
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "trending-up": '<path d="M16 7h6v6"/><path d="m22 7-8.5 8.5-5-5L2 17"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "git-compare": '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><path d="M11 18H8a2 2 0 0 1-2-2V9"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "shuffle": '<path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.7-1.1 2-1.7 3.3-1.7H22"/><path d="m18 2 4 4-4 4"/><path d="M2 6h1.9c1.5 0 2.9.9 3.6 2.2"/><path d="M22 18h-5.9c-1.3 0-2.6-.7-3.3-1.8l-.5-.8"/><path d="m18 14 4 4-4 4"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "bar-chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
}


def _icon(name: str, color: str = ACCENT, size: int = 18) -> str:
    inner = _ICON_PATHS[name]
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' "
        f"viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='1.75' "
        f"stroke-linecap='round' stroke-linejoin='round'>{inner}</svg>"
    )
    quoted = urllib.parse.quote(svg)
    return (
        f"<img src='data:image/svg+xml,{quoted}' width='{size}' height='{size}' "
        f"style='vertical-align:-4px;margin-right:9px;' alt=''/>"
    )


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

:root {{
    --accent: {ACCENT}; --accent-hover: {ACCENT_HOVER};
    --bg-base: {BG_BASE}; --bg-surface: {BG_SURFACE};
    --border: {BORDER}; --border-strong: #353a45;
    --text-primary: {TEXT_PRIMARY}; --text-secondary: {TEXT_SECONDARY}; --text-muted: {TEXT_MUTED};
    --radius-lg: 14px;
}}

/* Scrim over the Ferrofluid background canvas (injected behind .stApp at
   z-index -1): keeps text readable while letting the fluid show with character. */
.stApp {{ background: rgba(11, 12, 15, 0.46); font-family: 'Inter', system-ui, sans-serif; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: transparent; }}
/* Set the body font on the app root only — never on `span`, or it clobbers
   Streamlit's Material icon font (the sidebar-collapse arrow etc.). */
.stApp p, .stApp label, .block-container {{ font-family: 'Inter', system-ui, sans-serif; }}

/* Trim Streamlit chrome. Hide only the toolbar *actions* (Deploy / hamburger),
   not the whole toolbar — the collapsed-sidebar expand button lives inside the
   toolbar, so display:none on stToolbar would make the sidebar unreopenable. */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stToolbarActions"] {{ display: none; }}
/* Clear Streamlit's ~60px top header bar so the title isn't clipped, without
   the large empty band the injector gaps used to add. */
.block-container {{ padding-top: 4.5rem; max-width: 1320px; }}

/* Collapse the injected-CSS element container so it doesn't add a flex gap
   above the header. A <style> still applies while its container is hidden. */
[data-testid="stElementContainer"]:has(style) {{ display: none !important; }}

/* Keep the "expand sidebar" button (shown when the sidebar is collapsed)
   visible, sized, and clickable above the banner. Streamlit 1.53 renders it
   as stExpandSidebarButton and can collapse it to zero width. */
[data-testid="stExpandSidebarButton"] {{
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: auto !important;
    min-width: 2.25rem !important;
    height: 2.25rem !important;
    z-index: 1000000 !important;
    pointer-events: auto !important;
}}
[data-testid="stExpandSidebarButton"] svg {{ color: var(--text-primary) !important; }}

/* Metric cards — translucent glass */
[data-testid="stMetric"] {{
    background: var(--bg-surface);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 20px;
    box-shadow: 0 18px 40px -12px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.03);
}}
[data-testid="stMetricLabel"] p {{
    color: var(--text-muted);
    font-size: 0.72rem !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
[data-testid="stMetricValue"] {{
    color: var(--text-primary);
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}

/* App header — clean, restrained typography (no effects) */
.app-hero {{
    padding: 0.2rem 0 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}}
.app-hero__title {{
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: clamp(1.7rem, 2.7vw, 2.3rem);
    font-weight: 700;
    letter-spacing: -0.6px;
    line-height: 1.14;
    color: var(--text-primary);
}}
.app-hero__sub {{
    font-family: 'Inter', sans-serif;
    font-size: 1.0rem;
    font-weight: 400;
    color: var(--text-secondary);
    margin-top: 0.45rem;
}}

/* Tabs */
[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--border); }}
button[data-baseweb="tab"] {{
    font-size: 0.95rem; font-weight: 500; letter-spacing: 0.01em; padding: 10px 6px;
}}
button[data-baseweb="tab"][aria-selected="true"] {{ color: var(--accent); }}
[data-baseweb="tab-highlight"] {{ background-color: var(--accent); }}

/* Sliders — refined: white thumb with an accent glow ring, rounded track */
[data-testid="stSlider"] [role="slider"] {{
    background-color: #ffffff !important;
    border: none !important;
    box-shadow: 0 0 0 3px rgba(57,135,229,0.30), 0 2px 6px rgba(0,0,0,0.45) !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}}
[data-testid="stSlider"] [role="slider"]:hover,
[data-testid="stSlider"] [role="slider"]:focus {{
    transform: scale(1.18);
    box-shadow: 0 0 0 5px rgba(57,135,229,0.22), 0 0 16px rgba(57,135,229,0.55) !important;
    outline: none !important;
}}
[data-baseweb="slider"] > div > div {{ border-radius: 999px !important; }}
[data-testid="stSliderThumbValue"] {{
    color: var(--accent) !important;
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}}
[data-testid="stSliderTickBar"] {{ margin-top: 2px; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{
    color: var(--text-muted) !important;
    font-size: 0.66rem !important;
}}

/* Section headers */
.sec {{ margin: 1.5rem 0 0.7rem; }}
.sec-title {{
    display: flex; align-items: center;
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1.08rem; font-weight: 700; color: var(--text-primary);
}}
.sec-sub {{
    color: var(--text-muted); font-size: 0.85rem;
    margin-top: 5px; padding-left: 27px; max-width: 820px; line-height: 1.45;
}}

/* Sidebar */
[data-testid="stSidebar"] {{ background: #121317; border-right: 1px solid var(--border); }}
.sidebar-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.4rem;
}}

/* Dataframe + buttons */
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; }}
[data-testid="stDownloadButton"] button {{
    border: 1px solid var(--accent); color: var(--accent);
    background: transparent; border-radius: 10px; font-weight: 550;
}}
[data-testid="stDownloadButton"] button:hover {{
    background: var(--accent); color: #fff; border-color: var(--accent);
}}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def section_header(title: str, subtitle: str | None = None, icon: str = "target"):
    icon_html = _icon(icon)
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sec"><div class="sec-title">{icon_html}'
        f'<span>{title}</span></div>{sub}</div>',
        unsafe_allow_html=True,
    )


def sidebar_title(text: str):
    st.sidebar.markdown(
        f'<div class="sidebar-title">{text}</div>', unsafe_allow_html=True
    )
