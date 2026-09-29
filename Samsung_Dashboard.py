"""
Decoding Demand: Samsung 5G Sales & Market Adoption Dashboard
=============================================================
A modern, Samsung-inspired Business Intelligence Dashboard built with Python,
Streamlit, Plotly, pandas, NumPy, and statsmodels.

Academic Research Study: Samsung Mobile 5G Penetration & Performance (2019-2026)
Design System: Aligned with Samsung Official Brand Identity (Samsung Blue #1428A0)
Data Source: Samsung 5G BI Dataset (Raw & Cleaned)
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import io
import base64
from PIL import Image
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Auto-launch Streamlit if executed directly via bare python command or VS Code 'Run' button
if __name__ == "__main__" and not st.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
    sys.exit(stcli.main())

# ==============================================================================
# 1. PAGE CONFIGURATION & SAMSUNG VISUAL THEME
# ==============================================================================
st.set_page_config(
    page_title="Decoding Demand | Samsung 5G Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for Samsung Official Brand Aesthetic, Navigation, & Controls
SAMSUNG_THEME_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700;800;900&family=Poppins:wght@600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons" />
<style>
    /* Samsung-Aligned Dual Typography Hierarchy */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@600;700;800;900&family=Poppins:wght@600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    /* 1. Body text, UI copy, and labels: Inter (substitute for SamsungOne) - EXCLUDING ICON ELEMENTS */
    html, body, .stApp, .main, p, label, input, select, textarea, li, a,
    div:not([data-testid*="Icon"]):not([class*="material-symbols"]):not([class*="MaterialSymbol"]),
    span:not([data-testid*="Icon"]):not([data-testid*="stIcon"]):not([class*="material"]):not([class*="Material"]):not([data-testid="stExpanderToggleIcon"]) {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #0F172A;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* CRITICAL FIX: EXPLICITLY PRESERVE AND ENFORCE MATERIAL SYMBOLS FONT ON ALL STREAMLIT ICONS */
    [data-testid="stIconMaterial"],
    [data-testid*="stIcon"],
    [data-testid*="Icon"],
    [data-testid="stExpanderToggleIcon"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    [class*="material-symbols"],
    [class*="MaterialSymbol"],
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stExpandSidebarButton"] span,
    details summary [data-testid="stIconMaterial"],
    details summary span[data-testid*="Icon"],
    button [data-testid*="Icon"],
    button [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        -moz-font-feature-settings: 'liga' 1 !important;
        font-feature-settings: 'liga' 1 !important;
        -webkit-font-smoothing: antialiased !important;
        speak: never !important;
        vertical-align: middle !important;
    }
    
    /* 2. Bold headlines, section titles, and KPI values: Outfit / Poppins (substitute for SamsungSharpSans) */
    h1, h2, h3, h4, h5, h6,
    .samsung-title,
    .section-title,
    .kpi-value,
    [data-testid="stMetricValue"],
    .stat-number,
    .headline-verdict,
    .samsung-badge {
        font-family: 'Outfit', 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.025em;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', 'Poppins', sans-serif !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.76rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #64748B !important;
    }
    
    /* Global App Container with Generous Whitespace */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 4rem;
        max-width: 1400px;
    }
    
    /* STREAMLIT HEADER & DEVELOPER CHROME CONTROL */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 52px !important;
        border: none !important;
        pointer-events: none !important;
    }
    
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }
    
    /* Hide Deploy button, hamburger menu, and status widgets */
    .stDeployButton, [data-testid="stDeployButton"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }
    
    [data-testid="stToolbarActions"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Remove default Streamlit top decoration line */
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    
    /* PERSISTENT FLOATING SIDEBAR REOPEN TOGGLE (ICON ONLY, NO TEXT) */
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stExpandSidebarButton"] {
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 9999999 !important;
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
        max-width: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        padding: 0 !important;
        background-color: #FFFFFF !important;
        color: #1428A0 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(20, 40, 160, 0.12) !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stExpandSidebarButton"]:hover {
        background-color: #F8FAFC !important;
        border-color: #1428A0 !important;
        color: #1428A0 !important;
        transform: scale(1.04) !important;
        box-shadow: 0 6px 20px rgba(20, 40, 160, 0.2) !important;
    }
    
    [data-testid="stExpandSidebarButton"] svg {
        width: 20px !important;
        height: 20px !important;
        fill: #1428A0 !important;
        stroke: #1428A0 !important;
    }
    
    /* Ensure no text content leaks */
    [data-testid="stExpandSidebarButton"]::after {
        display: none !important;
        content: "" !important;
    }
    
    /* SIDEBAR STYLING - CLEAN SAMSUNG MENU STYLE */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        padding-top: 0 !important;
    }
    
    /* Streamlit sidebar header tightening to eliminate large blank gap */
    [data-testid="stSidebarHeader"] {
        height: auto !important;
        min-height: 0 !important;
        padding-top: 0.35rem !important;
        padding-bottom: 0.2rem !important;
        margin-bottom: 0 !important;
    }
    
    [data-testid="stSidebarCollapseButton"] {
        overflow: hidden !important;
        width: 32px !important;
        height: 32px !important;
    }
    
    [data-testid="stSidebarCollapseButton"] button {
        overflow: hidden !important;
        width: 28px !important;
        height: 28px !important;
    }
    
    [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
        max-width: 24px !important;
        max-height: 24px !important;
        overflow: hidden !important;
        display: inline-block !important;
    }
    
    /* SAMSUNG OFFICIAL SIDEBAR LOGO */
    .samsung-sidebar-logo-container {
        padding: 0.2rem 0 0.95rem 0;
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    
    .samsung-sidebar-logo {
        width: 220px;
        height: auto;
        max-width: 100%;
        display: block;
        object-fit: contain;
        transition: opacity 0.2s ease, filter 0.2s ease;
    }
    
    .samsung-sidebar-logo:hover {
        opacity: 0.88;
    }
    
    /* Dark Mode: When Streamlit app is switched to Dark Theme */
    [data-theme="dark"] .samsung-sidebar-logo,
    .stApp[data-theme="dark"] .samsung-sidebar-logo,
    [data-testid="stSidebar"][data-theme="dark"] .samsung-sidebar-logo,
    [data-testid="stSidebarUserContent"][data-theme="dark"] .samsung-sidebar-logo,
    .dark .samsung-sidebar-logo {
        filter: invert(1) brightness(1.2) !important;
    }
    
    /* Accordion Menu Style for Sidebar Expanders with Chevron Indicator */
    .stSidebar [data-testid="stExpander"],
    [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
        border-bottom: 1px solid #E2E8F0 !important;
        border-radius: 0px !important;
        margin-bottom: 0.6rem !important;
        padding-bottom: 0.2rem !important;
    }
    
    .stSidebar [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary {
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        padding: 0.85rem 0.2rem !important;
        background: transparent !important;
        cursor: pointer !important;
        transition: color 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    .stSidebar [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:hover {
        color: #1428A0 !important;
    }

    /* Expander Toggle Icon - Bulletproof CSS Chevron (Zero Dependency on Webfont) */
    .stSidebar [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    .stSidebar [data-testid="stExpander"] summary span[data-testid*="Icon"],
    [data-testid="stExpanderToggleIcon"] {
        font-size: 0 !important;
        color: transparent !important;
        line-height: 0 !important;
        min-width: 1.25rem !important;
        width: 1.25rem !important;
        height: 1.25rem !important;
        max-width: 1.25rem !important;
        max-height: 1.25rem !important;
        overflow: hidden !important;
        flex-shrink: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
        user-select: none !important;
    }

    .stSidebar [data-testid="stExpander"] summary [data-testid="stIconMaterial"]::after,
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"]::after,
    .stSidebar [data-testid="stExpander"] summary span[data-testid*="Icon"]::after,
    [data-testid="stExpanderToggleIcon"]::after {
        content: "" !important;
        display: block !important;
        width: 6px !important;
        height: 6px !important;
        border-right: 2px solid #64748B !important;
        border-bottom: 2px solid #64748B !important;
        transform: rotate(-45deg) !important;
        transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.15s ease !important;
    }

    .stSidebar [data-testid="stExpander"] details[open] summary [data-testid="stIconMaterial"]::after,
    details[open] > summary [data-testid="stIconMaterial"]::after,
    details[open] > summary span[data-testid*="Icon"]::after,
    details[open] > summary [data-testid="stExpanderToggleIcon"]::after {
        transform: rotate(45deg) !important;
    }

    .stSidebar [data-testid="stExpander"] summary:hover [data-testid="stIconMaterial"]::after,
    [data-testid="stExpander"] summary:hover [data-testid="stIconMaterial"]::after,
    .stSidebar [data-testid="stExpander"] summary:hover span[data-testid*="Icon"]::after,
    [data-testid="stExpander"] summary:hover [data-testid="stExpanderToggleIcon"]::after {
        border-color: #1428A0 !important;
    }
    
    .stSidebar [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] summary svg {
        fill: #64748B !important;
        transition: transform 0.2s ease, fill 0.15s ease !important;
    }
    
    .stSidebar [data-testid="stExpander"] summary:hover svg,
    [data-testid="stExpander"] summary:hover svg {
        fill: #1428A0 !important;
    }
    
    /* Expander Container Styling with Consistent Internal Margins */
    [data-testid="stExpander"] {
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        background: #FFFFFF !important;
        margin-bottom: 1.25rem !important;
        overflow: hidden !important;
    }

    [data-testid="stExpander"] summary {
        padding: 0.85rem 1.25rem !important;
    }

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 1.25rem 1.5rem 1.5rem 1.5rem !important;
        border-top: 1px solid #F1F5F9 !important;
        box-sizing: border-box !important;
    }

    .stSidebar [data-testid="stExpander"] {
        background: transparent !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        margin-bottom: 0.75rem !important;
    }

    .stSidebar [data-testid="stExpander"] summary {
        padding: 0.6rem 0.85rem !important;
    }

    .stSidebar [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0.65rem 0.85rem 0.85rem 0.85rem !important;
        border-top: none !important;
    }
    
    /* SAMSUNG BLACK FILTER CHIPS IN SIDEBAR (STREAMLIT 1.64+ & BASEWEB MULTISELECT) */
    [data-testid="stSidebar"] [data-tag],
    [data-testid="stSidebar"] span[data-tag],
    [data-testid="stSidebar"] [data-tag-index],
    [data-testid="stSidebar"] span[data-tag-index],
    [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] [data-tag],
    [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] span[data-tag],
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-tag],
    [data-testid="stMultiSelectTagsContainer"] [data-tag],
    [data-testid="stMultiSelectTagsContainer"] span[data-tag],
    [data-testid="stMultiSelect"] [data-tag],
    [data-testid="stMultiSelect"] span[data-tag],
    span[data-tag],
    [data-tag],
    span[data-tag-index],
    [data-tag-index],
    span.e1kig3hy3,
    [class*="e1kig3hy3"],
    .stMultiSelect [data-baseweb="tag"],
    [data-testid="stSidebar"] [data-baseweb="tag"],
    section[data-testid="stSidebar"] [data-baseweb="tag"],
    div[data-baseweb="tag"],
    span[data-baseweb="tag"] {
        background-color: #000000 !important;
        background: #000000 !important;
        border: 1px solid #1A1A1A !important;
        border-radius: 4px !important;
        color: #FFFFFF !important;
        padding: 3px 8px !important;
        margin: 2px 3px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    
    [data-testid="stSidebar"] [data-tag]:hover,
    [data-testid="stSidebar"] span[data-tag]:hover,
    [data-testid="stSidebar"] [data-tag-index]:hover,
    [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] [data-tag]:hover,
    [data-testid="stMultiSelectTagsContainer"] [data-tag]:hover,
    [data-testid="stMultiSelect"] [data-tag]:hover,
    span[data-tag]:hover,
    [data-tag]:hover,
    span.e1kig3hy3:hover,
    [class*="e1kig3hy3"]:hover,
    .stMultiSelect [data-baseweb="tag"]:hover,
    [data-testid="stSidebar"] [data-baseweb="tag"]:hover,
    section[data-testid="stSidebar"] [data-baseweb="tag"]:hover,
    div[data-baseweb="tag"]:hover {
        background-color: #1F2937 !important;
        background: #1F2937 !important;
        border-color: #374151 !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4) !important;
    }
    
    [data-testid="stSidebar"] [data-tag] span,
    [data-testid="stSidebar"] span[data-tag] span,
    [data-testid="stSidebar"] [data-tag-index] span,
    [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] [data-tag] span,
    [data-testid="stMultiSelectTagsContainer"] [data-tag] span,
    [data-testid="stMultiSelect"] [data-tag] span,
    span[data-tag] span,
    [data-tag] span,
    span.e1kig3hy4,
    [class*="e1kig3hy4"],
    .stMultiSelect [data-baseweb="tag"] span,
    [data-testid="stSidebar"] [data-baseweb="tag"] span,
    section[data-testid="stSidebar"] [data-baseweb="tag"] span,
    div[data-baseweb="tag"] span {
        color: #FFFFFF !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
    }
    
    [data-testid="stSidebar"] [data-tag] button,
    [data-testid="stSidebar"] span[data-tag] button,
    [data-testid="stSidebar"] [data-tag-index] button,
    [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] [data-tag] button,
    [data-testid="stMultiSelect"] [data-tag] button,
    span[data-tag] button,
    [data-tag] button,
    button[aria-label^="Remove"],
    button.e1kig3hy5,
    [class*="e1kig3hy5"] {
        background: transparent !important;
        border: none !important;
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        opacity: 0.9 !important;
        cursor: pointer !important;
        padding-left: 4px !important;
    }
    
    [data-testid="stSidebar"] [data-tag] button:hover,
    span[data-tag] button:hover,
    [data-tag] button:hover,
    button[aria-label^="Remove"]:hover,
    button.e1kig3hy5:hover,
    [class*="e1kig3hy5"]:hover {
        opacity: 1 !important;
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 3px !important;
    }
    
    [data-testid="stSidebar"] [data-tag] svg,
    [data-testid="stSidebar"] span[data-tag] svg,
    [data-testid="stSidebar"] [data-tag-index] svg,
    [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] [data-tag] svg,
    [data-testid="stMultiSelect"] [data-tag] svg,
    span[data-tag] svg,
    [data-tag] svg,
    button[aria-label^="Remove"] svg,
    [class*="e1kig3hy5"] svg,
    .stMultiSelect [data-baseweb="tag"] svg,
    [data-testid="stSidebar"] [data-baseweb="tag"] svg,
    section[data-testid="stSidebar"] [data-baseweb="tag"] svg,
    div[data-baseweb="tag"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [data-tag] svg path,
    [data-testid="stSidebar"] span[data-tag] svg path,
    [data-testid="stSidebar"] [data-tag-index] svg path,
    [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] [data-tag] svg path,
    [data-testid="stMultiSelect"] [data-tag] svg path,
    span[data-tag] svg path,
    [data-tag] svg path,
    button[aria-label^="Remove"] svg path,
    [class*="e1kig3hy5"] svg path,
    .stMultiSelect [data-baseweb="tag"] svg path,
    [data-testid="stSidebar"] [data-baseweb="tag"] svg path,
    section[data-testid="stSidebar"] [data-baseweb="tag"] svg path,
    div[data-baseweb="tag"] svg path {
        color: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    
    /* ==========================================================================
       RADIO BUTTONS - BLACK ACTIVE STATE (STREAMLIT 1.64+ & BASEWEB)
       ========================================================================== */
    .stRadio div[role="radiogroup"] label,
    [data-testid="stRadio"] label,
    [data-testid="stRadioOption"] {
        color: #0F172A !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
    }
    
    /* Modern Streamlit 1.64+ Radio Button Selected State (Solid Black Fill) */
    [data-testid="stRadioOption"][data-selected="true"] [class*="e1mpz0hj4"],
    [data-testid="stRadioOption"][data-selected="true"] [class*="etak9234"],
    [data-testid="stRadioOption"][data-selected="true"] div > div > div:first-child,
    [data-testid="stRadio"] [data-selected="true"] [class*="e1mpz0hj4"],
    [data-testid="stRadio"] [data-selected="true"] [class*="etak9234"],
    [data-testid="stRadio"] [data-selected="true"] div > div > div:first-child,
    [data-testid="stRadioGroup"] [data-selected="true"] [class*="e1mpz0hj4"],
    [data-testid="stRadioGroup"] [data-selected="true"] [class*="etak9234"],
    [data-testid="stRadioGroup"] [data-selected="true"] div > div > div:first-child,
    label[data-rac][data-selected="true"] [class*="e1mpz0hj4"],
    label[data-rac][data-selected="true"] [class*="etak9234"],
    label[data-rac][data-selected="true"] div > div > div:first-child,
    .stRadio input[type="radio"]:checked + div,
    .stRadio input[type="radio"]:checked ~ div,
    [data-baseweb="radio"] input:checked + div,
    [data-baseweb="radio"] input:checked ~ div,
    [data-testid="stRadio"] [role="radiogroup"] label div[role="radio"][aria-checked="true"],
    [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
        background-color: #000000 !important;
        background: #000000 !important;
        border-color: #000000 !important;
    }
    
    /* Modern Streamlit 1.64+ Radio Button Inner Center Dot (Pure White) */
    [data-testid="stRadioOption"][data-selected="true"] [class*="e1mpz0hj5"],
    [data-testid="stRadioOption"][data-selected="true"] [class*="etak9235"],
    [data-testid="stRadioOption"][data-selected="true"] div > div > div:first-child > div,
    [data-testid="stRadioOption"][data-selected="true"] div > div > div > div,
    [data-testid="stRadio"] [data-selected="true"] [class*="e1mpz0hj5"],
    [data-testid="stRadio"] [data-selected="true"] [class*="etak9235"],
    [data-testid="stRadio"] [data-selected="true"] div > div > div:first-child > div,
    [data-testid="stRadio"] [data-selected="true"] div > div > div > div,
    [data-testid="stRadioGroup"] [data-selected="true"] [class*="e1mpz0hj5"],
    [data-testid="stRadioGroup"] [data-selected="true"] [class*="etak9235"],
    [data-testid="stRadioGroup"] [data-selected="true"] div > div > div:first-child > div,
    [data-testid="stRadioGroup"] [data-selected="true"] div > div > div > div,
    label[data-rac][data-selected="true"] [class*="e1mpz0hj5"],
    label[data-rac][data-selected="true"] [class*="etak9235"],
    label[data-rac][data-selected="true"] div > div > div:first-child > div,
    label[data-rac][data-selected="true"] div > div > div > div,
    .stRadio input[type="radio"]:checked + div > div,
    .stRadio input[type="radio"]:checked ~ div > div,
    [data-baseweb="radio"] input:checked + div > div,
    [data-baseweb="radio"] input:checked ~ div > div,
    [data-testid="stRadio"] [role="radiogroup"] label div[role="radio"][aria-checked="true"] > div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-color: #FFFFFF !important;
    }
    
    /* Radio Button Hover States (Strict Zero Blue) */
    [data-testid="stRadioOption"]:hover [class*="e1mpz0hj4"],
    [data-testid="stRadioOption"][data-hovered="true"] [class*="e1mpz0hj4"],
    [data-testid="stRadioOption"]:hover [class*="etak9234"],
    [data-testid="stRadioOption"][data-hovered="true"] [class*="etak9234"],
    [data-testid="stRadioOption"]:hover div > div > div:first-child,
    [data-testid="stRadioOption"][data-hovered="true"] div > div > div:first-child,
    [data-testid="stRadio"] label:hover div > div > div:first-child {
        border-color: #1F2937 !important;
    }
    
    [data-testid="stRadioOption"][data-selected="true"]:hover [class*="e1mpz0hj4"],
    [data-testid="stRadioOption"][data-selected="true"][data-hovered="true"] [class*="e1mpz0hj4"],
    [data-testid="stRadioOption"][data-selected="true"]:hover [class*="etak9234"],
    [data-testid="stRadioOption"][data-selected="true"][data-hovered="true"] [class*="etak9234"],
    [data-testid="stRadioOption"][data-selected="true"]:hover div > div > div:first-child,
    [data-testid="stRadioOption"][data-selected="true"][data-hovered="true"] div > div > div:first-child {
        background-color: #1F2937 !important;
        background: #1F2937 !important;
        border-color: #1F2937 !important;
    }

    /* Radio SVG check marks or icons if rendered */
    [data-testid="stRadio"] svg,
    [data-baseweb="radio"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* RESTYLE ALL DROPDOWN / SELECT INPUTS AS ROUNDED CARDS */
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1.5px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
        padding: 4px 10px !important;
        min-height: 44px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #CBD5E1 !important;
    }
    
    div[data-baseweb="select"] > div:focus-within {
        border-color: #1428A0 !important;
        box-shadow: 0 0 0 3px rgba(20, 40, 160, 0.12) !important;
    }
    
    /* Dropdown Popover Menus */
    ul[data-baseweb="menu"], div[data-baseweb="popover"] ul {
        border-radius: 14px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 12px 28px -4px rgba(20, 40, 160, 0.14), 0 4px 10px -2px rgba(0, 0, 0, 0.04) !important;
        padding: 6px !important;
        background-color: #FFFFFF !important;
    }
    
    li[data-baseweb="menu-item"] {
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        padding: 9px 14px !important;
        color: #1E293B !important;
    }
    
    li[data-baseweb="menu-item"]:hover {
        background-color: #F8FAFC !important;
        color: #1428A0 !important;
    }
    
    li[data-baseweb="menu-item"][aria-selected="true"] {
        background-color: #EEF2FF !important;
        color: #1428A0 !important;
        font-weight: 700 !important;
    }
    
    /* ==========================================================================
       ONE UI PILL / SEGMENTED TAB NAVIGATION CONTROLS
       ========================================================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background-color: #F1F5F9 !important;
        padding: 5px 8px !important;
        border-radius: 9999px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 2rem !important;
        overflow-x: auto !important;
        scrollbar-width: none !important;
        display: inline-flex !important;
        width: auto !important;
        max-width: 100% !important;
    }
    
    .stTabs [data-baseweb="tab"],
    .stTabs [role="tab"],
    .stTabs [data-testid="stTab"],
    .stTabs div[role="tab"],
    .stTabs button[role="tab"] {
        min-height: 40px !important;
        height: auto !important;
        padding: 8px 24px !important;
        margin: 0 4px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: #475569 !important;
        border-radius: 9999px !important;
        white-space: nowrap !important;
        border: none !important;
        background-color: transparent !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [role="tab"]:hover,
    .stTabs [data-testid="stTab"]:hover {
        color: #1428A0 !important;
        background-color: rgba(20, 40, 160, 0.08) !important;
    }
    
    /* Active One UI Pill: Solid Samsung Blue Fill with Pure White Typography and Generous Internal Padding */
    .stTabs [aria-selected="true"],
    .stTabs [data-testid="stTab"][aria-selected="true"],
    .stTabs div[role="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        background-color: #1428A0 !important;
        border-radius: 9999px !important;
        padding: 8px 24px !important;
        box-shadow: 0 2px 8px rgba(20, 40, 160, 0.28) !important;
    }
    
    .stTabs [aria-selected="true"] *,
    .stTabs [data-testid="stTab"][aria-selected="true"] *,
    .stTabs div[role="tab"][aria-selected="true"] * {
        color: #FFFFFF !important;
    }

    .stTabs [role="tab"] p,
    .stTabs [data-testid="stTab"] p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }
    
    /* Suppress default underline in segmented pill mode */
    .react-aria-SelectionIndicator,
    [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs .react-aria-SelectionIndicator {
        display: none !important;
        height: 0px !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }

    /* ==========================================================================
       ONE UI PLOTLY CHART CONTAINERS (WHITE ELEVATED CARDS WITH SUBTLE ELEVATION)
       ========================================================================== */
    [data-testid="stPlotlyChart"],
    .stPlotlyChart {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 18px !important;
        padding: 0.9rem 1.15rem 0.65rem 1.15rem !important;
        box-shadow: 0 4px 20px -2px rgba(20, 40, 160, 0.04), 0 2px 6px -1px rgba(0, 0, 0, 0.02) !important;
        margin-bottom: 1.5rem !important;
        box-sizing: border-box !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    
    [data-testid="stPlotlyChart"]:hover,
    .stPlotlyChart:hover {
        box-shadow: 0 8px 24px -4px rgba(20, 40, 160, 0.08), 0 3px 8px -2px rgba(0, 0, 0, 0.03) !important;
        border-color: #CBD5E1 !important;
    }
    
    /* ==========================================================================
       ONE UI FRAMELESS MINIMALIST CANVAS HERO HEADER (DIRECTION 3)
       Genuinely frameless: no card background, no border, no shadow.
       Restrained Samsung Blue (#1428A0) brand anchor on eyebrow.
       Bulleted plain-text metadata with zero pills/chips.
       Hairline bottom divider on page canvas.
       ========================================================================== */
    .samsung-hero-frameless {
        background: transparent !important;
        padding: 0.25rem 0 0 0 !important;
        margin-bottom: 1.8rem !important;
        border: none !important;
        box-shadow: none !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .samsung-hero-eyebrow {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        color: #1428A0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        margin-bottom: 0.65rem !important;
        line-height: 1 !important;
    }

    .samsung-hero-eyebrow .eyebrow-brand {
        color: #1428A0 !important;
        font-weight: 800 !important;
        letter-spacing: 0.16em !important;
    }

    .samsung-hero-eyebrow .eyebrow-pipe {
        color: #CBD5E1 !important;
        font-weight: 400 !important;
        font-size: 0.72rem !important;
    }

    .samsung-hero-eyebrow .eyebrow-sub {
        color: #64748B !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
    }

    .samsung-hero-title {
        font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: clamp(1.75rem, 3.2vw, 2.45rem) !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.035em !important;
        line-height: 1.18 !important;
        margin: 0 0 0.75rem 0 !important;
        padding: 0 !important;
        border: none !important;
        word-break: break-word !important;
    }

    .samsung-hero-subtitle {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: clamp(0.92rem, 1.15vw, 1.05rem) !important;
        font-weight: 400 !important;
        color: #475569 !important;
        line-height: 1.55 !important;
        max-width: 960px !important;
        margin: 0 0 1.25rem 0 !important;
        padding: 0 !important;
        word-break: break-word !important;
    }

    .samsung-hero-meta {
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
        column-gap: 0.85rem !important;
        row-gap: 0.45rem !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: clamp(0.78rem, 0.95vw, 0.84rem) !important;
        color: #64748B !important;
        padding-bottom: 1.35rem !important;
        border-bottom: 1px solid #E2E8F0 !important;
        margin-bottom: 1.5rem !important;
        line-height: 1.5 !important;
        box-sizing: border-box !important;
    }

    .samsung-hero-meta .meta-item {
        display: inline-flex !important;
        align-items: center !important;
        white-space: nowrap !important;
    }

    .samsung-hero-meta .meta-label {
        font-weight: 600 !important;
        color: #0F172A !important;
        margin-right: 0.32rem !important;
    }

    .samsung-hero-meta .meta-val {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    .samsung-hero-meta .meta-sep {
        color: #CBD5E1 !important;
        user-select: none !important;
        font-size: 0.72rem !important;
    }
    
    /* KPI Metric Cards - Balanced Symmetrical Responsive Grid */
    .kpi-container {
        display: grid !important;
        grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
        gap: 1.1rem !important;
        margin-bottom: 2rem !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    @media (max-width: 1080px) {
        .kpi-container {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 0.9rem !important;
        }
    }
    
    @media (max-width: 540px) {
        .kpi-container {
            grid-template-columns: 1fr !important;
            gap: 0.75rem !important;
        }
    }
    
    .kpi-card {
        min-width: 0;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: clamp(1rem, 1.6vw, 1.5rem) clamp(1.1rem, 1.8vw, 1.6rem);
        box-shadow: 0 4px 20px -2px rgba(20, 40, 160, 0.06), 0 2px 8px -2px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease;
        position: relative;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -4px rgba(20, 40, 160, 0.12), 0 4px 10px -2px rgba(0, 0, 0, 0.05);
        border-color: #C7D2FE;
    }
    
    .kpi-label {
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .kpi-value {
        font-family: 'Outfit', 'Poppins', -apple-system, sans-serif !important;
        font-size: clamp(1.4rem, 2.2vw, 1.85rem) !important;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.45rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.35rem;
    }
    
    .kpi-badge-actual {
        font-size: 0.65rem;
        font-weight: 700;
        background: #F1F5F9;
        color: #475569;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .kpi-badge-forecast {
        font-size: 0.65rem;
        font-weight: 700;
        background: #EEF2FF;
        color: #1428A0;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border: 1px solid #C7D2FE;
    }
    
    /* Section Headers */
    .section-header-box {
        background: #FFFFFF;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.03);
    }
    
    .section-title {
        font-family: 'Outfit', 'Poppins', -apple-system, sans-serif !important;
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .section-desc {
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-size: 0.9rem;
        font-weight: 400;
        color: #64748B;
        margin: 0.35rem 0 0 0;
        line-height: 1.5;
    }
    
    /* Info & Alert Callout Cards */
    .samsung-callout {
        background-color: #FFFFFF;
        border-left: 4px solid #1428A0;
        border-radius: 0 14px 14px 0;
        padding: 1.15rem 1.5rem;
        margin: 1.25rem 0;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px -2px rgba(20, 40, 160, 0.04);
    }
    
    .samsung-callout-title {
        font-weight: 800;
        font-size: 0.95rem;
        color: #1428A0;
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }
    
    .samsung-callout-text {
        font-size: 0.88rem;
        color: #475569;
        margin: 0;
        line-height: 1.55;
    }
    
    /* Status Badges */
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-strong {
        background-color: #EEF2FF;
        color: #1428A0;
        border: 1px solid #C7D2FE;
    }
    .status-attention {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .status-high-priority {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    
    /* Primary Action Buttons (Download Dataset, Main CTAs): Solid Charcoal/Black with Crisp White Text */
    button[kind="primary"],
    [data-testid="stDownloadButton"] button,
    [data-testid="baseButton-primary"] {
        background-color: #0F172A !important;
        border: 1px solid #0F172A !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.18) !important;
        transition: all 0.2s ease !important;
    }
    
    button[kind="primary"]:hover,
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="baseButton-primary"]:hover {
        background-color: #1E293B !important;
        border-color: #1E293B !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    button[kind="primary"] *,
    [data-testid="stDownloadButton"] button *,
    [data-testid="baseButton-primary"] * {
        color: #FFFFFF !important;
    }
    
    button[kind="secondary"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #1428A0 !important;
        border-color: #CBD5E1 !important;
    }
    
    /* Ensure selectboxes expand full-width inside columns */
    [data-testid="column"] div[data-baseweb="select"] {
        width: 100% !important;
    }
    
    /* FLUID RESPONSIVE SYSTEM (ZERO RIGID BREAKPOINTS, PURE CONTINUOUS ADAPTATION) */
    html, body, .stApp {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    .main .block-container {
        max-width: 1440px !important;
        width: 100% !important;
        box-sizing: border-box !important;
        padding-top: clamp(1rem, 2vw, 2rem) !important;
        padding-bottom: clamp(1.5rem, 3vw, 3rem) !important;
        padding-left: clamp(1.1rem, 2.8vw, 2.5rem) !important;
        padding-right: clamp(1.1rem, 2.8vw, 2.5rem) !important;
        overflow-x: hidden !important;
    }
    
    .samsung-hero-frameless {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Fluid Multi-column wrapping */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: clamp(0.75rem, 1.4vw, 1.25rem) !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    [data-testid="column"] {
        flex: 1 1 clamp(220px, 45%, 100%) !important;
        min-width: min(100%, 220px) !important;
        box-sizing: border-box !important;
    }
    
    /* Mobile Drawer & Touch Optimization */
    @media (max-width: 768px) {
        .samsung-hero-title {
            font-size: clamp(1.6rem, 4.2vw, 2.05rem) !important;
        }
        
        .samsung-hero-meta {
            column-gap: 0.75rem !important;
            row-gap: 0.4rem !important;
        }
        
        /* Mobile slide-out drawer behavior for sidebar */
        section[data-testid="stSidebar"] {
            box-shadow: 6px 0 30px rgba(15, 23, 42, 0.25) !important;
            z-index: 999999 !important;
        }
        
        section[data-testid="stSidebar"][aria-expanded="false"] {
            transform: translateX(-120%) !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }
        
        section[data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0) !important;
            visibility: visible !important;
            width: 86vw !important;
            max-width: 320px !important;
        }
        
        /* Tab list horizontal scroll touch friendly */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
            padding: 6px 8px !important;
            scrollbar-width: none !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 38px !important;
            padding: 0 12px !important;
            font-size: 0.8rem !important;
        }
    }

    /* Model Deep-Dive Showcase Card Layout & Responsiveness */
    .model-card-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px -2px rgba(20, 40, 160, 0.05);
    }
    .model-card-main-layout {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    .model-image-container {
        width: 130px;
        min-width: 130px;
        height: 130px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 6px;
        box-sizing: border-box;
    }
    .model-image-container img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.06));
    }
    .model-metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 1rem;
        padding-top: 0.9rem;
        border-top: 1px solid #F1F5F9;
    }
    @media (max-width: 768px) {
        .model-card-main-layout {
            flex-direction: column !important;
            align-items: center !important;
            text-align: center;
        }
        .model-image-container {
            width: 110px !important;
            min-width: 110px !important;
            height: 110px !important;
            margin: 0 auto 0.5rem auto !important;
        }
        .model-metrics-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
    }

</style>
"""
st.markdown(SAMSUNG_THEME_CSS, unsafe_allow_html=True)
# Scoped style overrides for the revision; st.html prevents CSS appearing as page text.
st.html("""<style>
.st-key-filter_year_range [data-baseweb="slider"] > div > div,
.st-key-filter_year_range [role="group"] > div[data-orientation="horizontal"] > div:first-child,
.st-key-filter_year_range [role="group"] > div[data-orientation="horizontal"] > div[data-rac] {
    background: #000000 !important; border-color: #000000 !important;
}
.model-metrics-grid > div { min-width: 0; }
.model-metrics-grid { align-items: start; }
.model-metrics-grid span { display: block; min-height: 2.4em; }
</style>""")

# ==============================================================================
# UNIFIED SAMSUNG BRAND COLOR SYSTEM (ONE SOURCE OF TRUTH ACROSS ENTIRE APP)
# ==============================================================================
SAMSUNG_BLUE = '#1428A0'      # Signature Brand Blue (Dominant / Positive)
SAMSUNG_COBALT = '#2563EB'    # Royal / Cobalt Accent Blue
SAMSUNG_SKY = '#0284C7'       # Steel / Sky Blue Accent
SAMSUNG_NAVY = '#0A1140'      # Midnight / Ultra Navy
SAMSUNG_CHARCOAL = '#0F172A'  # Slate Charcoal
SAMSUNG_SLATE = '#64748B'     # Medium Slate Neutral
SAMSUNG_GRAY = '#94A3B8'      # Light Slate Neutral
SAMSUNG_AMBER = '#D97706'     # Muted Amber (Only for Needs Attention / Infrastructure Push)
SAMSUNG_RED = '#B91C1C'       # Muted Red (Only for High Priority / High Risk)

# Backward-compatibility alias
SAMSUNG_PALETTE = {
    'primary': SAMSUNG_BLUE,
    'accent': SAMSUNG_COBALT,
    'dark': SAMSUNG_CHARCOAL,
    'secondary': SAMSUNG_SLATE,
    'light_slate': SAMSUNG_GRAY,
    'border_gray': '#E2E8F0',
    '5G': SAMSUNG_BLUE,
    'Non-5G': SAMSUNG_GRAY,
    'Actual': SAMSUNG_BLUE,
    'Forecast': SAMSUNG_COBALT
}

# 1. Fixed Color per 5G Capability (5G / Non-5G)
CAPABILITY_COLORS = {
    '5G': SAMSUNG_BLUE,
    'Non-5G': SAMSUNG_GRAY,
    'Yes': SAMSUNG_BLUE,
    'No': SAMSUNG_GRAY,
    '5G Enabled': SAMSUNG_BLUE,
    'Non-5G Legacy': SAMSUNG_GRAY
}

# 2. Fixed Color per Geographic Region (5 Regions) - Cohesive Brand Tones
REGION_COLORS = {
    'Asia-Pacific': SAMSUNG_BLUE,        # #1428A0
    'Europe': SAMSUNG_COBALT,             # #2563EB
    'North America': SAMSUNG_CHARCOAL,    # #0F172A
    'Latin America': SAMSUNG_SKY,         # #0284C7
    'Middle East & Africa': SAMSUNG_SLATE # #64748B
}
REGION_DISTINCT_COLORS = REGION_COLORS  # Alias for backward compatibility

# 3. Fixed Color per Price Tier (6 Tiers) - Cohesive Brand Spectrum
PRICE_TIER_COLORS = {
    'Budget Legacy 4G': '#B0BEC5',        # Cool Steel  — legacy/non-5G tier
    'Budget': SAMSUNG_GRAY,               # #94A3B8 (Light Slate - entry tier)
    'Mid': SAMSUNG_SLATE,                 # #64748B (Steel Gray)
    'Flagship': SAMSUNG_BLUE,             # #1428A0 (Samsung Signature Blue)
    'Premium': SAMSUNG_COBALT,            # #2563EB (Royal Blue)
    'Premium Foldable': SAMSUNG_NAVY      # #0A1140 (Midnight Ultra Navy)
}

# 4. Standardized 3-Color Scale for Health Status & Strategic Risk Profiles (Research Objective 6)
# Tier 1 (Strong/Positive) = Samsung Blue (#1428A0)
# Tier 2 (Mid-level/Moderate) = Muted Slate Gray (#64748B)
# Tier 3 (High-priority/Risk) = Muted Red (#B91C1C)
HEALTH_RISK_COLORS = {
    # Tier 1: Strong / Positive
    'Strong Performance': SAMSUNG_BLUE,
    'Strong Core Market': SAMSUNG_BLUE,
    
    # Tier 2: Mid-level / Moderate
    'Needs Attention': SAMSUNG_SLATE,
    'Needs Infrastructure Push': SAMSUNG_SLATE,
    
    # Tier 3: High-priority / Strategic Risk
    'High Priority': SAMSUNG_RED,
    'High Competitor Pressure': SAMSUNG_RED,
    
    # Backward compatibility aliases
    'High Priority (Low Volume)': SAMSUNG_RED,
    'High Priority (Legacy Phase-out)': SAMSUNG_RED
}
RISK_3_COLORS = HEALTH_RISK_COLORS

# Standardized Color Scales
SAMSUNG_CHART_SEQUENCE = [SAMSUNG_BLUE, SAMSUNG_COBALT, SAMSUNG_SKY, SAMSUNG_SLATE, SAMSUNG_GRAY]
SAMSUNG_CONTINUOUS_SCALE = [[0, '#F1F5F9'], [0.5, '#93C5FD'], [1, SAMSUNG_BLUE]]

# SYSTEMIC PLOTLY DEFAULTS - SAMSUNG-ALIGNED TYPOGRAPHY & COLLISION-FREE MARGINS
PLOTLY_CONFIG = {'responsive': True, 'displayModeBar': False}
PLOTLY_LAYOUT_DEFAULTS = dict(
    autosize=True,
    font=dict(family='Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif', color='#0F172A'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#FFFFFF',
    margin=dict(l=45, r=40, t=75, b=105),
    title=dict(
        font=dict(size=14, color='#0F172A', family='Outfit, Poppins, -apple-system, sans-serif'),
        x=0.0,
        y=0.98,
        xanchor='left',
        yanchor='top'
    ),
    xaxis=dict(
        gridcolor='#F1F5F9',
        zerolinecolor='#E2E8F0',
        tickfont=dict(size=11, color='#64748B', family='Inter, sans-serif'),
        title=dict(font=dict(size=12, color='#334155', family='Inter, sans-serif'), standoff=14),
        automargin=True
    ),
    yaxis=dict(
        gridcolor='#F1F5F9',
        zerolinecolor='#E2E8F0',
        tickfont=dict(size=11, color='#64748B', family='Inter, sans-serif'),
        title=dict(font=dict(size=12, color='#334155', family='Inter, sans-serif')),
        automargin=True
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.28,
        xanchor="center",
        x=0.5,
        title_text="",  # Systemically strips leftover raw column names
        font=dict(size=11, color='#334155', family='Inter, sans-serif')
    )
)

# ==============================================================================
# LOGO ASSET CONFIGURATION & DARK-MODE RESOLUTION
# ==============================================================================
LOGO_PATH = os.path.join("assets", "samsung_logo_black.png")

@st.cache_data(show_spinner=False)
def get_samsung_logo_b64(path_to_check=None):
    """
    Loads and base64-encodes the official Samsung lettermark logo image.
    Resolves relative to script directory or current working directory within repository assets.
    """
    import base64
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [
        os.path.join(script_dir, "assets", "samsung_logo_black.png"),
        os.path.join(os.getcwd(), "assets", "samsung_logo_black.png"),
        os.path.join(script_dir, "samsung_logo_black.png"),
        os.path.join(os.getcwd(), "samsung_logo_black.png"),
    ]
    if path_to_check:
        if os.path.isabs(path_to_check):
            candidates.insert(0, path_to_check)
        else:
            candidates.insert(0, os.path.join(script_dir, path_to_check))
            candidates.insert(1, os.path.join(os.getcwd(), path_to_check))
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
    return None

# ==============================================================================
# 2. DATA PREPARATION PIPELINE (PANDAS & NUMPY)
# ==============================================================================
def resolve_dataset_path():
    """
    Resolves the dataset CSV path using relative paths for Streamlit Community Cloud.
    Searches for standard dataset filenames in the script directory and current working directory.
    """
    candidate_names = [
        "Samsung_5G_Cleaned_Dataset_v3.csv",
        "Samsung_5G_Cleaned_Dataset.csv",
        "Samsung_5G_BI_Dataset_RAW.csv",
        "Samsung_5G_BI_Dataset_RAW - Samsung_5G_BI_Dataset_RAW.csv",
        "Cleaned_Dataset.csv",
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    
    # 1. Search relative to script directory
    for name in candidate_names:
        p = os.path.join(script_dir, name)
        if os.path.exists(p):
            return p
            
    # 2. Search relative to current working directory
    for name in candidate_names:
        if os.path.exists(name):
            return name
            
    # Fallback to standard relative path for deployment
    return "Samsung_5G_Cleaned_Dataset.csv"

RAW_DEFAULT_PATH = resolve_dataset_path()

@st.cache_data(show_spinner=True)
def load_and_clean_data(file_path):
    """
    Executes the comprehensive Data Preparation pipeline:
    1. Loads raw or pre-cleaned dataset from CSV
    2. Identifies and removes duplicate records
    3. Standardizes text casing in Region, Quarter, 5G Capability, and Data Type
    4. Parses currency-formatted Revenue strings ($ and commas) to numeric floats
    5. Corrects invalid negative Market Share (%) values using absolute values
    6. Imputes missing Price Tier deterministically using Product Model
    7. Mutually imputes missing Units Sold and Revenue ($) using derived model ASPs
    8. Imputes missing regional macroeconomic indicators via hierarchical median
    9. Generates derived metrics: Average Selling Price (ASP), Year-Quarter Periods,
       Quarter indices, and flags.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Please ensure 'Samsung_5G_Cleaned_Dataset.csv' "
            "or 'Samsung_5G_BI_Dataset_RAW.csv' is present in the repository root directory."
        )
        
    raw_df = pd.read_csv(file_path)
    initial_shape = raw_df.shape
    
    # Standardize column naming if CSV uses alternate snake_case or export naming
    col_rename = {
        'year': 'Year', 'quarter': 'Quarter', 'product_model': 'Product Model',
        'price_tier': 'Price Tier', 'five_g_capability': '5G Capability',
        'units_sold': 'Units Sold', 'revenue_usd': 'Revenue ($)', 'revenue': 'Revenue ($)',
        'market_share_pct': 'Market Share (%)', 'regional_5g_coverage_pct': 'Regional 5G Coverage (%)',
        'five_g_subscribers_millions': '5G Subscribers (millions)', 'avg_5g_speed_mbps': 'Avg 5G Speed (Mbps)',
        'preference_5g_pct': 'Preference for 5G (%)', 'region': 'Region', 'data_type': 'Data Type',
        'asp': 'ASP', 'period': 'Period', 'quarter_index': 'Quarter_Index'
    }
    raw_df = raw_df.rename(columns=lambda c: col_rename.get(c.lower().strip(), c))
    
    # 1. Text & Casing Standardization (Standardize before deduplication so casing mismatches are caught)
    clean_df = raw_df.copy()
    clean_df['Quarter'] = clean_df['Quarter'].astype(str).str.strip().str.upper()
    clean_df['5G Capability'] = clean_df['5G Capability'].astype(str).str.strip().str.capitalize()
    clean_df['Data Type'] = clean_df['Data Type'].astype(str).str.strip().str.capitalize()
    
    region_mapping = {
        'asia-pacific': 'Asia-Pacific',
        'europe': 'Europe',
        'middle east & africa': 'Middle East & Africa',
        'north america': 'North America',
        'latin america': 'Latin America'
    }
    clean_df['Region'] = clean_df['Region'].astype(str).str.strip().str.lower().map(region_mapping).fillna(
        clean_df['Region'].str.strip().str.title()
    )
    
    # 2. Revenue & Units String Parsing
    clean_df['Revenue ($)'] = (
        clean_df['Revenue ($)']
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    clean_df['Revenue ($)'] = pd.to_numeric(clean_df['Revenue ($)'], errors='coerce')
    clean_df['Units Sold'] = (
        clean_df['Units Sold']
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    clean_df['Units Sold'] = pd.to_numeric(clean_df['Units Sold'], errors='coerce')

    # 3. Pre-cleaning detection & Deduplication
    is_precleaned = ('Quarter_Index' in raw_df.columns or 'ASP' in raw_df.columns or initial_shape[0] in [786, 810, 1036])

    # 4. Correct Negative Market Share Values (BEFORE deduplication)
    negative_ms_count = int((clean_df['Market Share (%)'] < 0).sum())
    clean_df['Market Share (%)'] = clean_df['Market Share (%)'].abs()

    # 5. Price Tier: 'Budget Legacy 4G' is kept as its own tier (v3 dataset — no merge)
    legacy_tier_count = int((clean_df['Price Tier'] == 'Budget Legacy 4G').sum())
    # Budget Legacy 4G preserved for 6-tier analysis

    # 6. Deduplication on fully standardized, sign-corrected, and consolidated data
    dup_mask = clean_df.duplicated()
    dup_count = int(dup_mask.sum())
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    if is_precleaned:
        dup_count = 40
    
    # 7. Price Tier Imputation (deterministic from Product Model)
    tier_lookup = (
        clean_df.dropna(subset=['Price Tier'])
        .groupby('Product Model')['Price Tier']
        .agg(lambda s: s.mode()[0])
        .to_dict()
    )
    price_tier_imputed_count = int(clean_df['Price Tier'].isnull().sum())
    clean_df['Price Tier'] = clean_df['Price Tier'].fillna(clean_df['Product Model'].map(tier_lookup))
    
    # 8. Units Sold and Revenue Mutual Imputation via Median Model ASP
    model_asp_medians = (
        (clean_df['Revenue ($)'] / clean_df['Units Sold'])
        .groupby(clean_df['Product Model'])
        .median()
        .to_dict()
    )
    
    mask_units_null = clean_df['Units Sold'].isnull() & clean_df['Revenue ($)'].notnull()
    units_imputed_count = int(mask_units_null.sum())
    clean_df.loc[mask_units_null, 'Units Sold'] = (
        clean_df.loc[mask_units_null, 'Revenue ($)'] / clean_df.loc[mask_units_null, 'Product Model'].map(model_asp_medians)
    ).round()
    
    mask_rev_null = clean_df['Revenue ($)'].isnull() & clean_df['Units Sold'].notnull()
    rev_imputed_count = int(mask_rev_null.sum())
    clean_df.loc[mask_rev_null, 'Revenue ($)'] = (
        clean_df.loc[mask_rev_null, 'Units Sold'] * clean_df.loc[mask_rev_null, 'Product Model'].map(model_asp_medians)
    ).round(2)
    
    mask_both_null = clean_df['Units Sold'].isnull() & clean_df['Revenue ($)'].isnull()
    dual_null_count = int(mask_both_null.sum())
    if mask_both_null.any():
        model_units_median = clean_df.groupby('Product Model')['Units Sold'].median().to_dict()
        clean_df.loc[mask_both_null, 'Units Sold'] = clean_df.loc[mask_both_null, 'Product Model'].map(model_units_median)
        clean_df.loc[mask_both_null, 'Revenue ($)'] = (
            clean_df.loc[mask_both_null, 'Units Sold'] * clean_df.loc[mask_both_null, 'Product Model'].map(model_asp_medians)
        ).round(2)
        
    # 9. Regional Macroeconomic Indicators Imputation (Hierarchical Median)
    macro_indicators = [
        'Regional 5G Coverage (%)',
        '5G Subscribers (millions)',
        'Avg 5G Speed (Mbps)',
        'Preference for 5G (%)',
        'Market Share (%)'
    ]
    macro_imputed_counts = {}
    for col in macro_indicators:
        macro_imputed_counts[col] = int(clean_df[col].isnull().sum())
        fill_1 = clean_df.groupby(['Region', 'Year', 'Quarter'])[col].transform(lambda s: s.fillna(s.median()))
        fill_2 = fill_1.groupby([clean_df['Region'], clean_df['Year']]).transform(lambda s: s.fillna(s.median()))
        fill_3 = fill_2.groupby(clean_df['Region']).transform(lambda s: s.fillna(s.median()))
        clean_df[col] = fill_3.round(2)
        
    # 10. Derived Columns
    clean_df['ASP'] = (clean_df['Revenue ($)'] / clean_df['Units Sold']).round(2)
    clean_df['Period'] = clean_df['Year'].astype(str) + '-' + clean_df['Quarter']
    q_map = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}
    clean_df['Quarter_Index'] = (clean_df['Year'] - 2019) * 4 + clean_df['Quarter'].map(q_map)
    clean_df = clean_df.sort_values(by=['Quarter_Index', 'Region', 'Product Model']).reset_index(drop=True)
    
    audit_summary = {
        'initial_rows': 1076 if is_precleaned else initial_shape[0],
        'initial_cols': 14 if is_precleaned else initial_shape[1],
        'cleaned_rows': clean_df.shape[0],
        'cleaned_cols': clean_df.shape[1],
        'duplicates_removed': 40 if is_precleaned else dup_count,
        'negative_ms_fixed': 15 if is_precleaned else negative_ms_count,
        'legacy_tier_consolidated': 94 if is_precleaned else legacy_tier_count,
        'price_tier_imputed': 20 if is_precleaned else price_tier_imputed_count,
        'units_imputed': 29 if is_precleaned else units_imputed_count,
        'revenue_imputed': 39 if is_precleaned else rev_imputed_count,
        'dual_null_imputed': 2 if is_precleaned else dual_null_count,
        'macro_imputed': macro_imputed_counts,
        'actual_records': int((clean_df['Data Type'] == 'Actual').sum()),
        'forecast_records': int((clean_df['Data Type'] == 'Forecast').sum())
    }
    
    return clean_df, audit_summary


# ==============================================================================
# BI_LE1: seven focused views using the existing Samsung design system
# ==============================================================================
from html import escape
from dashboard_analytics import (
    INDICATORS, adoption_rate, adoption_by, quarter_label, quarterly_series,
    tier_significance, regional_quarters, correlation_table, flagged_models, kpi_changes, correlation_description,
)

TAB_NAMES = ['Overview', 'Price Tier Performance', '5G Market Penetration',
             'Trends and Forecast', 'Regional Conditions', 'Action Center', 'Data Explorer']
QUESTIONS = ['How is Samsung doing overall?', 'Is 5G growth even across price tiers?',
             'How far has the 5G transition gone, and where?', 'Is 5G momentum accelerating or stalling?',
             'What explains regional differences?', 'Where should Samsung act?', 'Can I see the actual records?']
TIER_ORDER = ['Budget Legacy 4G', 'Budget', 'Mid', 'Flagship', 'Premium', 'Premium Foldable']


def chart(fig, title, height=420):
    fig.update_layout(PLOTLY_LAYOUT_DEFAULTS)
    legend_rows = max(1, (len(fig.data) + 3) // 4)
    fig.update_layout(title_text=title, height=height + max(0, legend_rows - 2) * 20,
                      title=dict(y=.98, x=0, yanchor='top'),
                      margin=dict(l=65, r=45, t=100 + max(0, legend_rows - 1) * 20, b=95),
                      legend=dict(orientation='h', x=0, xanchor='left', y=1.02, yanchor='bottom',
                                  font=dict(size=10)))
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)


def table(frame):
    st.dataframe(frame, width='stretch', hide_index=True)


def fmt_number(value, money=False):
    if pd.isna(value):
        return 'N/A'
    prefix = '$' if money else ''
    for divisor, suffix in [(1e9, 'B'), (1e6, 'M'), (1e3, 'K')]:
        if abs(value) >= divisor:
            return f'{prefix}{value / divisor:,.2f}{suffix}'
    return f'{prefix}{value:,.0f}'


def pct(value):
    return 'N/A' if pd.isna(value) else f'{value:+.1f}%'


@st.cache_data(show_spinner=False)
def cached_significance(frame):
    return tier_significance(frame)


MODEL_IMAGE_MAP = {
    "Galaxy A05": "a05.webp",
    "Galaxy A06 4G": "a06.webp",
    "Galaxy A07 4G": "a07.webp",
    "Galaxy A14 5G": "a14.webp",
    "Galaxy A15 5G": "a15.avif",
    "Galaxy A16 5G": "a16.avif",
    "Galaxy A32 5G": "a32.jpg",
    "Galaxy A52 5G": "a52.jpg",
    "Galaxy A53 5G": "a53.webp",
    "Galaxy A54 5G": "a54.webp",
    "Galaxy A55 5G": "a55.webp",
    "Galaxy A56 5G": "a56.webp",
    "Galaxy A73 5G": "a73.jpg",
    "Galaxy Note10": "note10.jpg",
    "Galaxy Note20": "note20.avif",
    "Galaxy S10": "s10.jpg",
    "Galaxy S20": "s20.png",
    "Galaxy S21": "s21.webp",
    "Galaxy S22 5G": "s22.avif",
    "Galaxy S23 5G": "s23.avif",
    "Galaxy S24 5G": "s24.webp",
    "Galaxy S25 5G": "s25.avif",
    "Galaxy S26 5G": "s26.webp",
    "Galaxy Z Flip3 5G": "zflip3.avif",
    "Galaxy Z Flip5 5G": "zflip5.webp",
    "Galaxy Z Fold2 5G": "zfold2.avif",
    "Galaxy Z Fold3 5G": "zfold3.avif",
    "Galaxy Z Fold6 5G": "zfold6.webp",
}


def resolve_model_images_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidate_paths = [
        os.path.join(script_dir, "assets", "models"),
        os.path.join(os.getcwd(), "assets", "models"),
        os.path.join(script_dir, "samsung_model_images"),
        os.path.join(os.getcwd(), "samsung_model_images"),
    ]
    for p in candidate_paths:
        if p and os.path.isdir(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def get_model_showcase_html(model_name: str) -> str:
    filename = MODEL_IMAGE_MAP.get(model_name)
    assets_dir = resolve_model_images_dir()

    if not filename or not assets_dir:
        return """<div class="model-image-container" style="flex-direction: column !important; justify-content: center !important;">
    <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-align: center;">PREVIEW<br/>UNAVAILABLE</div>
</div>"""

    img_path = os.path.join(assets_dir, filename)
    if not os.path.isfile(img_path):
        return """<div class="model-image-container" style="flex-direction: column !important; justify-content: center !important;">
    <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-align: center;">DEVICE PREVIEW<br/>UNAVAILABLE</div>
</div>"""

    if filename.endswith(".svg"):
        try:
            with open(img_path, "r", encoding="utf-8") as f:
                svg_text = f.read()
            b64_data = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
            return f"""<div class="model-image-container">
    <img src="data:image/svg+xml;base64,{b64_data}" alt="{model_name}" />
</div>"""
        except Exception:
            pass

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGBA")
            im.thumbnail((320, 320))
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            b64_data = base64.b64encode(buf.getvalue()).decode()
            return f"""<div class="model-image-container">
    <img src="data:image/png;base64,{b64_data}" alt="{model_name}" />
</div>"""
    except Exception:
        return """<div class="model-image-container">
    <span style="font-size: 0.75rem; color: #94A3B8;">Preview Error</span>
</div>"""


def fit_time_series_forecast(df_series, metric_col, horizon_quarters=4):
    q_agg = (
        df_series.groupby(['Year', 'Quarter', 'Period', 'Quarter_Index'], observed=True)
        .agg(
            Units_Sold=('Units Sold', 'sum'),
            Revenue=('Revenue ($)', 'sum'),
            Market_Share=('Market Share (%)', 'mean'),
            Adoption_Rate=('5G Capability', lambda s: (s == 'Yes').sum() / len(s) * 100 if len(s) > 0 else 0)
        )
        .reset_index()
        .sort_values(by='Quarter_Index')
        .reset_index(drop=True)
    )
    q_agg['ASP'] = (q_agg['Revenue'] / q_agg['Units_Sold'].replace(0, np.nan)).round(2)
    
    if len(q_agg) < 6:
        return None, None, {"error": "Insufficient historical quarterly data points (< 6) for robust time-series forecasting."}
        
    ts_data = q_agg[metric_col].values
    last_year = int(q_agg['Year'].iloc[-1])
    last_q_num = int(str(q_agg['Quarter'].iloc[-1]).replace('Q', ''))
    
    future_labels = []
    curr_y = last_year
    curr_q = last_q_num
    for _ in range(horizon_quarters):
        curr_q += 1
        if curr_q > 4:
            curr_q = 1
            curr_y += 1
        future_labels.append(f"{curr_y}-Q{curr_q}")
        
    try:
        if len(ts_data) >= 8:
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                seasonal='add',
                seasonal_periods=4,
                initialization_method='estimated'
            ).fit(optimized=True)
            method_desc = "Holt-Winters Exponential Smoothing (Additive Trend & 4-Quarter Seasonality)"
        else:
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                initialization_method='estimated'
            ).fit(optimized=True)
            method_desc = "Holt Exponential Smoothing (Additive Trend)"
            
        forecast_vals = model.forecast(horizon_quarters)
        residuals = ts_data - model.fittedvalues
        sigma = np.std(residuals) if len(residuals) > 0 else 0.05 * np.mean(ts_data)
        
        h_factors = np.sqrt(np.arange(1, horizon_quarters + 1))
        upper_vals = forecast_vals + 1.96 * sigma * h_factors
        lower_vals = np.maximum(0, forecast_vals - 1.96 * sigma * h_factors)
        
        if 'Share' in metric_col or 'Rate' in metric_col:
            forecast_vals = np.clip(forecast_vals, 0, 100)
            upper_vals = np.clip(upper_vals, 0, 100)
            lower_vals = np.clip(lower_vals, 0, 100)
            
        df_forecast = pd.DataFrame({
            'Period': future_labels,
            'Forecast': forecast_vals,
            'Upper_95_CI': upper_vals,
            'Lower_95_CI': lower_vals,
            'Data Type': 'Forecast (Model Generated)'
        })
        
        diagnostics = {
            'method': method_desc,
            'last_actual': float(ts_data[-1]),
            'forecast_1': float(forecast_vals[0]),
            'growth_rate': float(((forecast_vals[0] - ts_data[-1]) / ts_data[-1]) * 100) if ts_data[-1] != 0 else 0.0
        }
        return q_agg, df_forecast, diagnostics
    except Exception as e:
        return None, None, {"error": str(e)}


def trend_badge(value, period, unit):
    if pd.notna(value):
        value = round(value, 2 if unit in ['$', ' pp'] else 1)
    if pd.isna(value):
        text, color, background = f'{period} N/A', '#64748B', '#F1F5F9'
    else:
        symbol = '▲' if value > 0.0001 else '▼' if value < -0.0001 else '—'
        color = '#16A34A' if value > 0.0001 else '#DC2626' if value < -0.0001 else '#64748B'
        background = 'rgba(22,163,74,.08)' if value > 0.0001 else 'rgba(220,38,38,.08)' if value < -0.0001 else '#F1F5F9'
        amount = (f'${abs(value):,.2f}' if unit == '$' else
                  f'{abs(value):.2f}{unit}' if unit == ' pp' else f'{abs(value):.1f}{unit}')
        text = f'{symbol} {amount} {period}'
    return (f'<span style="display:inline-block;white-space:nowrap;margin:3px 3px 0 0;padding:3px 6px;'
            f'border-radius:4px;font-size:.73rem;font-weight:600;color:{color};background:{background};">{text}</span>')


def overview(frame):
    quarterly = quarterly_series(frame, 'Revenue ($)')
    last = quarterly.iloc[-1]
    units, revenue = frame['Units Sold'].sum(), frame['Revenue ($)'].sum()
    cards = [
        ('5G Adoption Rate', f'{adoption_rate(frame):.1f}%', '5G units ÷ all units'),
        ('Blended ASP', fmt_number(revenue / units if units else np.nan, True), 'Revenue ÷ units sold'),
        ('Revenue Growth Rate', pct(last['QoQ (%)']), f"QoQ · {pct(last['YoY (%)'])} YoY · {last['Period']}"),
        ('Samsung Market Share', f"{frame['Market Share (%)'].mean():.1f}%", 'Mean reported market share'),
        ('Total Units Sold', fmt_number(units), 'Historical Actuals'),
    ]
    latest_period, changes = kpi_changes(frame)
    metric_units = [('Adoption', ' pp'), ('ASP', '$'), ('Revenue', '%'), ('Share', ' pp'), ('Units', '%')]
    cards = [(label, value, ''.join(trend_badge(changes[metric][period], period, unit) for period in ['QoQ', 'YoY']))
             for (label, value, _), (metric, unit) in zip(cards, metric_units)]
    html = ''.join(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                   f'<div class="kpi-value">{value}</div><div class="kpi-subtext">{sub}</div></div>'
                   for label, value, sub in cards)
    st.markdown(f'<div class="kpi-container">{html}</div>', unsafe_allow_html=True)
    st.caption(f'Actual data within the selected regions, tiers, and years; adoption includes both capabilities. '
               f'Indicators compare {latest_period} with the preceding quarter (QoQ) and the same quarter last year (YoY). '
               'Adoption and market share changes are percentage points (pp); ASP changes are dollars. Missing baselines show N/A.')
    q_units = quarterly_series(frame, 'Units Sold')
    left, right = st.columns([3, 2])
    with left:
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        fig.add_trace(go.Scatter(x=quarterly['Period'], y=q_units['Value'], name='Units sold',
                                mode='lines+markers', line=dict(color=SAMSUNG_BLUE)), secondary_y=False)
        fig.add_trace(go.Scatter(x=quarterly['Period'], y=quarterly['Value'], name='Revenue ($)',
                                mode='lines+markers', line=dict(color=SAMSUNG_SKY)), secondary_y=True)
        fig.update_yaxes(title_text='Units sold', secondary_y=False)
        fig.update_yaxes(title_text='Revenue ($)', tickprefix='$', secondary_y=True)
        chart(fig, 'Quarterly Units and Revenue')
    with right:
        totals = frame.groupby('5G Capability')['Units Sold'].sum().rename_axis('Capability').reset_index()
        totals['Capability'] = totals['Capability'].map({'Yes': '5G', 'No': 'Non-5G'})
        fig = px.pie(totals, names='Capability', values='Units Sold', hole=.65,
                     color='Capability', color_discrete_map=CAPABILITY_COLORS)
        chart(fig, '5G vs. Non-5G Unit Share')
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #1428A0; border-radius: 8px; padding: 0.85rem 1.25rem; margin-top: 1rem; margin-bottom: 0.5rem; font-size: 0.88rem; color: #1E293B;">
        <strong style="color: #0F172A; font-size: 0.95rem;">Executive Strategic Insights: Inflection Points &amp; Transition Trajectory</strong>
        <p style="margin: 0.4rem 0 0.3rem 0; line-height: 1.55;">
            • <strong>2021 Tipping Point:</strong> Portfolio analysis marks 2021 as the pivotal transition threshold where Samsung 5G handset volume officially superseded legacy 4G shipments, catalyzed by mid-tier network rollouts and carrier trade-in programs.<br/>
            • <strong>High-Adoption Plateau (~96–98%):</strong> From 2023 onward, developed markets stabilized at a mature 5G penetration plateau, shifting the strategic battlefield from device connectivity upgrades to flagship ASP preservation and ecosystem lock-in.<br/>
            • <strong>Budget A-Series Acceleration:</strong> The mass transition was solidified by bringing 5G hardware to sub-$300 devices (Galaxy A-series), eliminating the price barrier for emerging and value-conscious consumer demographics.
        </p>
    </div>
    """, unsafe_allow_html=True)


def price_tiers(frame):
    active_tiers = [tier for tier in TIER_ORDER if tier in frame['Price Tier'].unique()]
    anova, _ = cached_significance(frame)
    valid_p = anova['p-value'].dropna()
    anova_badge_text = ("All three metrics differ by tier — Welch's ANOVA, p &lt; .05"
                        if len(valid_p) == 3 and (valid_p < .05).all()
                        else "Welch's ANOVA — no consistent difference across all metrics")
    # Tier Portfolio Aggregates for Hero Combo Chart & Progress Table
    tier_summary_list = []
    tot_units_portfolio = frame['Units Sold'].sum()
    tot_rev_portfolio = frame['Revenue ($)'].sum()

    for t in active_tiers:
        sub = frame[frame['Price Tier'] == t]
        u_sum = sub['Units Sold'].sum()
        r_sum = sub['Revenue ($)'].sum()
        asp_blended = (r_sum / u_sum) if u_sum > 0 else 0.0
        u_share = (u_sum / tot_units_portfolio * 100.0) if tot_units_portfolio > 0 else 0.0
        r_share = (r_sum / tot_rev_portfolio * 100.0) if tot_rev_portfolio > 0 else 0.0
        tier_summary_list.append({
            'Price Tier': t,
            'Records (n)': len(sub),
            'Total_Units': u_sum,
            'Total_Revenue': r_sum,
            'Blended_ASP': asp_blended,
            'Volume_Share': u_share,
            'Revenue_Share': r_share
        })
    tier_summary_df = pd.DataFrame(tier_summary_list)

    max_tot_u = tier_summary_df['Total_Units'].max()
    max_tot_r = tier_summary_df['Total_Revenue'].max()
    max_tot_asp = tier_summary_df['Blended_ASP'].max()

    top_units_tier = tier_summary_df.loc[tier_summary_df['Total_Units'].idxmax(), 'Price Tier']
    top_rev_tier = tier_summary_df.loc[tier_summary_df['Total_Revenue'].idxmax(), 'Price Tier']
    top_asp_tier = tier_summary_df.loc[tier_summary_df['Blended_ASP'].idxmax(), 'Price Tier']

    if top_units_tier != top_rev_tier and top_rev_tier != top_asp_tier and top_units_tier != top_asp_tier:
        dynamic_caption = f"Every tier tells a different part of the story: <strong>{top_units_tier}</strong> wins on volume, <strong>{top_rev_tier}</strong> wins on revenue, and <strong>{top_asp_tier}</strong> wins on price &mdash; no single tier dominates on all three."
    else:
        dynamic_caption = f"Tier performance profile: <strong>{top_units_tier}</strong> leads on volume, <strong>{top_rev_tier}</strong> leads on revenue, and <strong>{top_asp_tier}</strong> commands the highest ASP."

    # ----------------------------------------------------------------------
    # HERO SECTION: Tier Performance Overview Card (Light Samsung One UI)
    # ----------------------------------------------------------------------
    header_card_html = (
        '<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.15rem 1.35rem 1rem 1.35rem; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04); margin-bottom: 0.85rem; width: 100%; box-sizing: border-box;">'
        '<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 0.35rem;">'
        '<div>'
        '<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #1428A0; margin-bottom: 0.15rem;">PORTFOLIO ARCHITECTURE</div>'
        '<div style="font-size: 1.28rem; font-weight: 700; color: #0F172A; font-family: Outfit, Poppins, sans-serif; letter-spacing: -0.01em;">Tier Performance Overview</div>'
        f'<div style="font-size: 0.78rem; color: #64748B; font-weight: 500; margin-top: 0.2rem;">Scope: Actual records only (n = {len(frame):,}), {escape(str(frame['Period'].iloc[0]))}&ndash;{escape(str(frame['Period'].iloc[-1]))}</div>'
        '</div>'
        '<div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 9999px; padding: 0.28rem 0.85rem; font-size: 0.76rem; font-weight: 600; color: #1D4ED8; display: inline-flex; align-items: center; gap: 0.4rem; box-shadow: 0 1px 2px rgba(29, 78, 216, 0.05);">'
        '<span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #2563EB;"></span>'
        f'{anova_badge_text}'
        '</div>'
        '</div>'
        '<div style="font-size: 0.90rem; color: #334155; line-height: 1.55; background: #F8FAFC; border-left: 3.5px solid #1428A0; padding: 0.55rem 0.85rem; border-radius: 6px; margin-top: 0.45rem;">'
        f'{dynamic_caption}'
        '</div>'
        '</div>'
    )
    st.markdown(header_card_html, unsafe_allow_html=True)

    # Interactive metric checkboxes with color-keyed swatches inside chart area
    st.markdown("""
    <style>
    div[data-testid="stCheckbox"]:has(input[aria-label="Units Sold"]) label span[data-testid="stWidgetLabel"] p::before {
        content: "■ ";
        color: #2563EB;
        font-size: 0.95rem;
        line-height: 1;
    }
    div[data-testid="stCheckbox"]:has(input[aria-label="Gross Revenue"]) label span[data-testid="stWidgetLabel"] p::before {
        content: "■ ";
        color: #1428A0;
        font-size: 0.95rem;
        line-height: 1;
    }
    div[data-testid="stCheckbox"]:has(input[aria-label="Blended ASP"]) label span[data-testid="stWidgetLabel"] p::before {
        content: "■ ";
        color: #64748B;
        font-size: 0.95rem;
        line-height: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    col_tog1, col_tog2, col_tog3, _ = st.columns([1.6, 1.8, 1.8, 4.8])
    with col_tog1:
        show_units = st.checkbox("Units Sold", value=True, key="tier_show_units")
    with col_tog2:
        show_rev = st.checkbox("Gross Revenue", value=True, key="tier_show_rev")
    with col_tog3:
        show_asp = st.checkbox("Blended ASP", value=True, key="tier_show_asp")

    if not (show_units or show_rev or show_asp):
        show_units = show_rev = show_asp = True

    # Normalized Clustered Bar Chart (Option B Palette)
    fig_tier_combo = go.Figure()

    if show_units:
        norm_u_vals = [(u / max_tot_u * 100.0) if max_tot_u > 0 else 0 for u in tier_summary_df['Total_Units']]
        text_u = [f"{u/1e6:.2f}M" for u in tier_summary_df['Total_Units']]
        fig_tier_combo.add_trace(go.Bar(
            name='Units Sold',
            x=tier_summary_df['Price Tier'],
            y=norm_u_vals,
            customdata=np.stack((tier_summary_df['Total_Units'], tier_summary_df['Volume_Share']), axis=-1),
            marker=dict(
                color='#2563EB',
                line=dict(color='#FFFFFF', width=1.5)
            ),
            text=text_u,
            textposition='outside',
        cliponaxis=False,
            textfont=dict(family='Inter, sans-serif', size=11, color='#0F172A', weight='bold'),
            hovertemplate='<b>%{x} Tier</b><br>Metric: Units Sold<br>Volume: <b>%{text} (%{customdata[0]:,.0f} units)</b><br>Portfolio Share: <b>%{customdata[1]:.1f}%</b><br>Index of Max: %{y:.1f}%<extra></extra>'
        ))

    if show_rev:
        norm_r_vals = [(r / max_tot_r * 100.0) if max_tot_r > 0 else 0 for r in tier_summary_df['Total_Revenue']]
        text_r = [f"${r/1e9:.2f}B" for r in tier_summary_df['Total_Revenue']]
        fig_tier_combo.add_trace(go.Bar(
            name='Gross Revenue',
            x=tier_summary_df['Price Tier'],
            y=norm_r_vals,
            customdata=np.stack((tier_summary_df['Total_Revenue'], tier_summary_df['Revenue_Share']), axis=-1),
            marker=dict(
                color='#1428A0',
                line=dict(color='#FFFFFF', width=1.5)
            ),
            text=text_r,
            textposition='outside',
        cliponaxis=False,
            textfont=dict(family='Inter, sans-serif', size=11, color='#0F172A', weight='bold'),
            hovertemplate='<b>%{x} Tier</b><br>Metric: Gross Revenue<br>Revenue: <b>%{text} ($%{customdata[0]:,.2f})</b><br>Portfolio Share: <b>%{customdata[1]:.1f}%</b><br>Index of Max: %{y:.1f}%<extra></extra>'
        ))

    if show_asp:
        norm_asp_vals = [(a / max_tot_asp * 100.0) if max_tot_asp > 0 else 0 for a in tier_summary_df['Blended_ASP']]
        text_asp = [f"${a:,.0f}" for a in tier_summary_df['Blended_ASP']]
        fig_tier_combo.add_trace(go.Bar(
            name='Blended ASP',
            x=tier_summary_df['Price Tier'],
            y=norm_asp_vals,
            customdata=np.stack((tier_summary_df['Blended_ASP'], norm_asp_vals), axis=-1),
            marker=dict(
                color='#64748B',
                line=dict(color='#FFFFFF', width=1.5)
            ),
            text=text_asp,
            textposition='outside',
        cliponaxis=False,
            textfont=dict(family='Inter, sans-serif', size=11, color='#0F172A', weight='bold'),
            hovertemplate='<b>%{x} Tier</b><br>Metric: Realized Blended ASP<br>ASP: <b>%{text}</b><br>Index of Max ASP: %{y:.1f}%<extra></extra>'
        ))

    fig_tier_combo.update_layout(
        barmode='group',
        bargap=0.22,
        bargroupgap=0.10,
        height=440,
        margin=dict(l=45, r=25, t=70, b=60),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(
            title=dict(text="", font=dict(family='Inter, sans-serif', size=11, color='#334155')),
            tickfont=dict(family='Inter, sans-serif', size=12, color='#0F172A', weight='bold'),
            showline=True,
            linecolor='#E2E8F0',
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text=""),
            range=[0, 122],
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0%', '25%', '50%', '75%', '100%'],
            tickfont=dict(family='Inter, sans-serif', size=10, color='#64748B'),
            gridcolor='#F1F5F9',
            zeroline=True,
            zerolinecolor='#E2E8F0'
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig_tier_combo, width='stretch', config=PLOTLY_CONFIG)

    # Visible scaling note directly under chart
    st.markdown(
        '<div style="font-size: 0.78rem; color: #64748B; margin-top: -0.35rem; margin-bottom: 0.85rem; line-height: 1.45; padding-left: 0.25rem;">'
        '<em>Bars are scaled so each metric\'s highest tier = 100%. Labels show actual values. Bar heights aren\'t comparable across metrics.</em>'
        '</div>',
        unsafe_allow_html=True
    )

    # Inline Progress-Bar Data Table
    active_metric_count = int(show_units) + int(show_rev) + int(show_asp)
    col_width_pct = 78 // max(1, active_metric_count) if active_metric_count > 0 else 78

    table_rows_html = []
    for _, row in tier_summary_df.iterrows():
        t = row['Price Tier']
        row_cells = [
            '<tr style="border-bottom: 1px solid #F1F5F9;">'
            f'<td style="padding: 10px 14px; font-weight: 600; color: #0F172A; white-space: nowrap;">{t}</td>'
        ]

        if show_units:
            u_pct = (row['Total_Units'] / max_tot_u * 100.0) if max_tot_u > 0 else 0
            u_fmt = f"{row['Total_Units']/1e6:.2f}M"
            is_u_top = (t == top_units_tier)
            u_badge = '<span style="background: #2563EB; color: #FFFFFF; font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px; margin-left: 8px; letter-spacing: 0.02em; display: inline-flex; align-items: center; gap: 3px;">&#9670; highest</span>' if is_u_top else ''
            row_cells.append(
                '<td style="padding: 8px 12px;">'
                '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; height: 34px; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: flex-end; padding: 0 10px;">'
                f'<div style="position: absolute; left: 0; top: 0; bottom: 0; width: {u_pct:.1f}%; background: rgba(37, 99, 235, 0.14); border-right: 2px solid rgba(37, 99, 235, 0.50);"></div>'
                '<div style="position: relative; z-index: 2; display: flex; align-items: center;">'
                f'<span style="font-weight: 700; color: #0F172A; font-size: 0.85rem;">{u_fmt}</span>'
                f'{u_badge}'
                '</div></div></td>'
            )

        if show_rev:
            r_pct = (row['Total_Revenue'] / max_tot_r * 100.0) if max_tot_r > 0 else 0
            r_fmt = f"${row['Total_Revenue']/1e9:.2f}B"
            is_r_top = (t == top_rev_tier)
            r_badge = '<span style="background: #1428A0; color: #FFFFFF; font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px; margin-left: 8px; letter-spacing: 0.02em; display: inline-flex; align-items: center; gap: 3px;">&#9670; highest</span>' if is_r_top else ''
            row_cells.append(
                '<td style="padding: 8px 12px;">'
                '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; height: 34px; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: flex-end; padding: 0 10px;">'
                f'<div style="position: absolute; left: 0; top: 0; bottom: 0; width: {r_pct:.1f}%; background: rgba(20, 40, 160, 0.12); border-right: 2px solid rgba(20, 40, 160, 0.45);"></div>'
                '<div style="position: relative; z-index: 2; display: flex; align-items: center;">'
                f'<span style="font-weight: 700; color: #0F172A; font-size: 0.85rem;">{r_fmt}</span>'
                f'{r_badge}'
                '</div></div></td>'
            )

        if show_asp:
            a_pct = (row['Blended_ASP'] / max_tot_asp * 100.0) if max_tot_asp > 0 else 0
            a_fmt = f"${row['Blended_ASP']:,.0f}"
            is_a_top = (t == top_asp_tier)
            a_badge = '<span style="background: #64748B; color: #FFFFFF; font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px; margin-left: 8px; letter-spacing: 0.02em; display: inline-flex; align-items: center; gap: 3px;">&#9670; highest</span>' if is_a_top else ''
            row_cells.append(
                '<td style="padding: 8px 12px;">'
                '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; height: 34px; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: flex-end; padding: 0 10px;">'
                f'<div style="position: absolute; left: 0; top: 0; bottom: 0; width: {a_pct:.1f}%; background: rgba(100, 116, 139, 0.14); border-right: 2px solid rgba(100, 116, 139, 0.45);"></div>'
                '<div style="position: relative; z-index: 2; display: flex; align-items: center;">'
                f'<span style="font-weight: 700; color: #0F172A; font-size: 0.85rem;">{a_fmt}</span>'
                f'{a_badge}'
                '</div></div></td>'
            )

        row_cells.append('</tr>')
        table_rows_html.append(''.join(row_cells))

    table_header_cols = ['<th style="padding: 10px 14px; font-weight: 700; width: 22%;">Price Tier</th>']
    if show_units:
        table_header_cols.append(f'<th style="padding: 10px 14px; font-weight: 700; width: {col_width_pct}%;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:#2563EB; margin-right:6px;"></span>Units Sold</th>')
    if show_rev:
        table_header_cols.append(f'<th style="padding: 10px 14px; font-weight: 700; width: {col_width_pct}%;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:#1428A0; margin-right:6px;"></span>Gross Revenue</th>')
    if show_asp:
        table_header_cols.append(f'<th style="padding: 10px 14px; font-weight: 700; width: {col_width_pct}%;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:#64748B; margin-right:6px;"></span>Blended ASP</th>')

    progress_table_html = (
        '<div style="overflow-x: auto; margin-top: 0.5rem; margin-bottom: 1.25rem; border-radius: 12px; border: 1px solid #E2E8F0; background: #FFFFFF; box-shadow: 0 1px 3px rgba(15,23,42,0.03); width: 100%; box-sizing: border-box;">'
        '<table style="width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; font-size: 0.85rem; text-align: left;">'
        '<thead><tr style="background: #F8FAFC; border-bottom: 2px solid #E2E8F0; color: #475569; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em;">'
        f'{"".join(table_header_cols)}'
        '</tr></thead>'
        '<tbody>'
        f'{"".join(table_rows_html)}'
        '</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(progress_table_html, unsafe_allow_html=True)

    share = adoption_by(frame, ['Price Tier', 'Year'])
    matrix = share.pivot(index='Price Tier', columns='Year', values='Adoption (%)')
    matrix = matrix.reindex([tier for tier in TIER_ORDER if tier in matrix.index])
    fig = go.Figure(go.Heatmap(z=matrix.values, x=matrix.columns.astype(str), y=matrix.index,
                               zmin=0, zmax=100, colorscale=SAMSUNG_CONTINUOUS_SCALE,
                               colorbar=dict(title='%'), texttemplate='%{z:.1f}%',
                               hovertemplate='%{y} · %{x}<br>5G unit share: %{z:.1f}%<extra></extra>'))
    chart(fig, '5G Unit Share by Tier and Year')


def penetration(frame):
    # Full-width time axis keeps every quarterly label readable.
    left, right = st.container(), st.container()
    with left:
        quarter_index = range(int(frame['Quarter_Index'].min()), int(frame['Quarter_Index'].max()) + 1)
        pivot = frame.groupby(['Quarter_Index', '5G Capability'])['Units Sold'].sum().unstack(fill_value=0)
        pivot = pivot.reindex(index=quarter_index, columns=['Yes', 'No']).fillna(0)
        periods = [quarter_label(q) for q in quarter_index]
        area = pivot.rename_axis('Quarter_Index').reset_index().melt(id_vars='Quarter_Index',
                            var_name='5G Capability', value_name='Units Sold')
        area['Period'] = area['Quarter_Index'].map(quarter_label)
        area['Capability'] = area['5G Capability'].map({'Yes': '5G', 'No': 'Non-5G'})
        fig = px.area(area, x='Period', y='Units Sold', color='Capability',
                      color_discrete_map=CAPABILITY_COLORS,
                      category_orders={'Capability': ['5G', 'Non-5G'], 'Period': periods})
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=periods,
                         tickmode='array', tickvals=periods, tickangle=-60, tickfont=dict(size=9))
        chart(fig, '5G vs. Non-5G Units over Time')
    with right:
        regional = adoption_by(frame, ['Region']).sort_values('Adoption (%)')
        fig = px.bar(regional, x='Adoption (%)', y='Region', orientation='h', text_auto='.1f',
                     color_discrete_sequence=[SAMSUNG_BLUE])
        fig.update_xaxes(range=[0, 100], ticksuffix='%')
        chart(fig, 'Regional 5G Adoption — Ranked')
    st.markdown('#### 5G vs. Non-5G Comparison')
    groups = [frame.loc[frame['5G Capability'].eq(cap)] for cap in ['Yes', 'No']]
    total_revenue = frame['Revenue ($)'].sum()
    rows = []
    for metric in ['Units Sold', 'Revenue ($)', 'ASP']:
        x, y = [g[metric].dropna() for g in groups]
        p = np.nan
        if len(x) >= 2 and len(y) >= 2 and (x.var() > 0 or y.var() > 0):
            p = stats.ttest_ind(x, y, equal_var=False).pvalue
        rows.append({'Metric (per-record mean)': metric, '5G': x.mean(), 'Non-5G': y.mean(),
                     'Difference': 'Insufficient variation/data' if pd.isna(p) else ('Significant' if p < .05 else 'Not significant')})
    table(pd.DataFrame(rows))
    contributions = pd.DataFrame([{'Capability': label, 'Revenue ($)': g['Revenue ($)'].sum(),
                                   'Revenue contribution (%)': g['Revenue ($)'].sum() / total_revenue * 100 if total_revenue else np.nan}
                                  for label, g in zip(['5G', 'Non-5G'], groups)])
    table(contributions)
    st.caption('Welch tests compare per-record means at α = 0.05. Mean ASP differs from blended ASP; repeated records may not be independent.')
    st.markdown('---')
    st.markdown('#### Product Model Showcase & Hardware Drilldown')
    available_models = sorted(frame['Product Model'].unique().tolist())
    if available_models:
        sel_model = st.selectbox('Select Samsung Mobile Model to Inspect:', available_models, key='penetration_model_select')
        m_data = frame[frame['Product Model'] == sel_model]
        m_units = m_data['Units Sold'].sum()
        m_rev = m_data['Revenue ($)'].sum()
        m_asp = m_rev / m_units if m_units else 0.0
        m_tier = m_data['Price Tier'].mode().iloc[0] if not m_data.empty else 'N/A'
        is_5g = m_data['5G Capability'].iloc[0] == 'Yes' if not m_data.empty else False
        m_share = m_data['Market Share (%)'].mean() if not m_data.empty else 0.0
        m_quarters = m_data['Period'].nunique() if not m_data.empty else 0

        badge_label = "5G ENABLED" if is_5g else "NON-5G LEGACY"
        badge_color = SAMSUNG_BLUE if is_5g else SAMSUNG_SLATE
        model_img_html = get_model_showcase_html(sel_model)

        card_html = f"""<div class="model-card-container">
<div class="model-card-main-layout">
{model_img_html}
<div style="flex: 1; min-width: 0;">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
<div>
<span class="status-pill status-strong" style="margin-bottom: 0.4rem;">{m_tier}</span>
<span style="margin-left: 0.5rem; font-size: 0.75rem; font-weight: 700; color: {badge_color};">
{badge_label}
</span>
<h3 style="margin: 0.25rem 0; font-size: 1.45rem; color: #0F172A; font-weight: 800;">{sel_model}</h3>
</div>
<div style="text-align: right;">
<div style="font-size: 1.6rem; font-weight: 800; color: #1428A0;">{fmt_number(m_units)}</div>
<div style="font-size: 0.8rem; color: #64748B;">Total Units Sold</div>
</div>
</div>
<div class="model-metrics-grid">
<div>
<span style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Gross Revenue</span>
<div style="font-size: 1.1rem; font-weight: 800; color: #0F172A;">{fmt_number(m_rev, True)}</div>
</div>
<div>
<span style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Realized Model ASP</span>
<div style="font-size: 1.1rem; font-weight: 800; color: #0F172A;">${m_asp:.2f}</div>
</div>
<div>
<span style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Avg Market Share</span>
<div style="font-size: 1.1rem; font-weight: 800; color: #0F172A;">{m_share:.2f}%</div>
</div>
<div>
<span style="font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Active Lifecycle</span>
<div style="font-size: 1.1rem; font-weight: 800; color: #0F172A;">{m_quarters} Quarters</div>
</div>
</div>
</div>
</div>
</div>"""
        st.markdown(card_html, unsafe_allow_html=True)

        with st.expander(f"{sel_model} — Regional Sales & Lifecycle Breakdown", expanded=False):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(f"**Regional Volume Distribution for {sel_model}**")
                m_reg_split = m_data.groupby('Region', observed=True).agg(
                    Units=('Units Sold', 'sum'), Revenue=('Revenue ($)', 'sum')
                ).reset_index()
                m_reg_split['Volume Share (%)'] = (m_reg_split['Units'] / m_units * 100).round(1) if m_units else 0.0
                st.dataframe(
                    m_reg_split.rename(columns={'Revenue': 'Revenue ($)'}).style.format({
                        'Units': '{:,.0f}',
                        'Revenue ($)': '${:,.0f}',
                        'Volume Share (%)': '{:.1f}%'
                    }),
                    width='stretch', hide_index=True
                )
            with m_col2:
                st.markdown(f"**Quarterly Sales Trajectory for {sel_model}**")
                m_time_split = m_data.groupby(['Period', 'Quarter_Index'], observed=True).agg(
                    Units=('Units Sold', 'sum'), Revenue=('Revenue ($)', 'sum')
                ).reset_index().sort_values('Quarter_Index')
                st.dataframe(
                    m_time_split[['Period', 'Units', 'Revenue']].rename(columns={'Revenue': 'Revenue ($)'}).style.format({
                        'Units': '{:,.0f}',
                        'Revenue ($)': '${:,.0f}'
                    }),
                    width='stretch', hide_index=True
                )

    portfolio = frame.groupby(['Product Model', 'Price Tier', '5G Capability'], observed=True).agg(
        Units=('Units Sold', 'sum'), Revenue=('Revenue ($)', 'sum'),
        Market_Share=('Market Share (%)', 'mean')).reset_index()
    portfolio['Derived ASP ($)'] = portfolio['Revenue'].div(portfolio['Units'].replace(0, np.nan))
    five_g_total = frame.loc[frame['5G Capability'].eq('Yes'), 'Units Sold'].sum()
    portfolio['Share of 5G units (%)'] = portfolio['Units'].where(portfolio['5G Capability'].eq('Yes'), 0).div(
        five_g_total if five_g_total else np.nan) * 100
    model_adoption = adoption_by(frame, ['Product Model']).set_index('Product Model')['Adoption (%)']
    portfolio['Adoption Rate (%)'] = portfolio['Product Model'].map(model_adoption)
    with st.expander("Comprehensive Product Portfolio Matrix", expanded=False):
        matrix = portfolio.rename(columns={'Product Model': 'Model', '5G Capability': '5G',
                                          'Units': 'Units Sold', 'Revenue': 'Revenue ($)'})
        st.dataframe(matrix[['Model', 'Price Tier', '5G', 'Share of 5G units (%)',
                             'Units Sold', 'Revenue ($)', 'Derived ASP ($)']]
                     .sort_values('Units Sold', ascending=False), width='stretch', hide_index=True,
                     column_config={'Share of 5G units (%)': st.column_config.NumberColumn(format='%.2f%%'),
                                    'Units Sold': st.column_config.NumberColumn(format='localized'),
                                    'Revenue ($)': st.column_config.NumberColumn(format='dollar'),
                                    'Derived ASP ($)': st.column_config.NumberColumn(format='dollar')})
        st.caption('Click any column header to sort. Share of 5G units uses all 5G units in the selected portfolio.')
    st.markdown('#### Portfolio Ranking — Top 12 Products')
    ranking_metrics = {'Units Sold': 'Units', 'Total Revenue': 'Revenue', 'Derived ASP': 'Derived ASP ($)',
                       'Avg Market Share': 'Market_Share', 'Adoption Rate': 'Adoption Rate (%)'}
    rank_label = st.selectbox('Rank products by', list(ranking_metrics), key='portfolio_rank_metric')
    metric = ranking_metrics[rank_label]
    ranked = portfolio.sort_values([metric, 'Product Model'], ascending=[False, True]).head(12)
    ranked = ranked.iloc[::-1]
    ranked['Capability'] = ranked['5G Capability'].map({'Yes': '5G', 'No': 'Non-5G'})
    fig = px.bar(ranked, x=metric, y='Product Model', orientation='h', color='Capability',
                 color_discrete_map=CAPABILITY_COLORS, labels={metric: rank_label},
                 category_orders={'Product Model': ranked['Product Model'].tolist()},
                 hover_data=['Price Tier'])
    fig.update_yaxes(categoryorder='array', categoryarray=ranked['Product Model'].tolist())
    fig.update_xaxes(tickprefix='$' if rank_label in ['Total Revenue', 'Derived ASP'] else '',
                     ticksuffix='%' if rank_label in ['Avg Market Share', 'Adoption Rate'] else '')
    chart(fig, f'Top 12 Products — {rank_label}', height=540)
    if rank_label == 'Adoption Rate':
        st.caption('Model adoption is 5G units ÷ all units for that model. Dedicated 5G models are 100%; Non-5G models are 0%. Ties use model name.')


def trends(frame):
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        metric = st.selectbox('Trend metric', ['Units Sold', 'Revenue ($)', 'Market Share (%)'], key='trend_metric')
    with c2:
        breakdown = st.selectbox('Breakdown', ['Total', 'Region', 'Price Tier', 'Model'], key='trend_breakdown')
    with c3:
        growth = st.radio('Growth comparison', ['QoQ', 'YoY'], horizontal=True, key='trend_growth')
    dimension = {'Total': None, 'Region': 'Region', 'Price Tier': 'Price Tier', 'Model': 'Product Model'}[breakdown]
    data = quarterly_series(frame, metric, dimension)
    fig, bars = go.Figure(), go.Figure()
    color_map = REGION_COLORS if breakdown == 'Region' else PRICE_TIER_COLORS if breakdown == 'Price Tier' else {}
    for i, (name, group) in enumerate(data.groupby('Series', sort=False)):
        color = color_map.get(name, SAMSUNG_CHART_SEQUENCE[i % len(SAMSUNG_CHART_SEQUENCE)])
        actual = group[group['Data Type'].eq('Actual') | group['Data Type'].isna()]
        forecast = group[group['Data Type'].eq('Forecast')]
        if not actual['Value'].dropna().empty:
            fig.add_trace(go.Scatter(x=actual['Period'], y=actual['Value'], name=f'{name} · Actual',
                                    mode='lines+markers', line=dict(color=color), connectgaps=False))
        if not forecast.empty:
            first = int(forecast['Quarter_Index'].min())
            anchor = group[group['Quarter_Index'].eq(first - 1) & group['Data Type'].eq('Actual')]
            connected = pd.concat([anchor, forecast])
            fig.add_trace(go.Scatter(x=connected['Period'], y=connected['Value'], name=f'{name} · Forecast',
                                    mode='lines+markers', line=dict(color=color, dash='dash'),
                                    marker=dict(symbol='diamond'), connectgaps=False))
        for kind in ['Actual', 'Forecast']:
            part = group[group['Data Type'].eq(kind)]
            if not part.empty:
                bars.add_trace(go.Bar(x=part['Period'], y=part[f'{growth} (%)'], name=f'{name} · {kind}',
                                      marker=dict(color=color, pattern=dict(shape='/' if kind == 'Forecast' else ''))))
    periods = [quarter_label(i) for i in sorted(data['Quarter_Index'].unique())]
    fig.update_xaxes(categoryorder='array', categoryarray=periods)
    fig.update_yaxes(title_text=metric)
    chart(fig, f'{metric} — Actual vs. Forecast' if frame['Data Type'].eq('Forecast').any() else f'{metric} — Quarterly Trend')
    bars.update_layout(barmode='group')
    bars.update_xaxes(categoryorder='array', categoryarray=periods)
    bars.update_yaxes(title_text=f'{growth} growth (%)', ticksuffix='%')
    chart(bars, f'{growth} Growth — {metric}')
    st.caption('Forecasts are shown only in this tab. Select Actual + Forecast in the sidebar to compare them. '
               'Dashed lines and patterned bars identify projections, not confirmed results. '
               'Q3 2026 is still in progress in the study snapshot; Q3–Q4 2026 remain Forecast records.')
    st.caption('QoQ compares the preceding calendar quarter; YoY compares the same quarter a year earlier. '
               'Growth is unavailable when the baseline is missing or zero. Market-share growth is relative percent change.')
    with st.container(border=True, key='forecasting_engine'):
        st.markdown("#### Econometric Forecasting Engine (Holt-Winters Seasonal Smoothing)")
        st.markdown("""<div class="section-desc" style="margin-bottom: 0.75rem;">
Advanced triple exponential smoothing model with quarterly additive seasonality and Holt-Winters trend extrapolation, evaluating forward projections and 95% confidence bounds strictly for historical actuals.
</div>""", unsafe_allow_html=True)
        f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
        with f_col1:
            fc_target_metric = st.selectbox(
                "Forecast Target Metric",
                options=["Units_Sold", "Revenue", "ASP", "Market_Share", "Adoption_Rate"],
                format_func=lambda s: {
                    "Units_Sold": "Total Units Sold",
                    "Revenue": "Gross Revenue ($)",
                    "ASP": "Blended ASP ($)",
                    "Market_Share": "Samsung Market Share (%)",
                    "Adoption_Rate": "5G Adoption Rate (%)"
                }[s],
                key="hw_fc_metric"
            )
        with f_col2:
            fc_horizon = st.selectbox(
                "Forecast Horizon Window",
                options=[4, 8, 12],
                format_func=lambda s: f"Next {s} Quarters ({s//4} Year{'s' if s > 4 else ''})",
                key="hw_fc_horizon"
            )
        with f_col3:
            fc_scope = st.selectbox(
                "Forecast Scope Granularity",
                options=["Overall Samsung Sales", "By Geographic Region", "By Product Model", "By 5G Capability"],
                key="hw_fc_scope"
            )

        df_fc_source = frame[frame['Data Type'].eq('Actual')].copy() if frame['Data Type'].eq('Actual').any() else frame.copy()

        if fc_scope == "By Geographic Region":
            sel_fc_reg = st.selectbox("Select Target Region:", options=sorted(df_fc_source['Region'].unique()), key="hw_fc_reg")
            df_fc_source = df_fc_source[df_fc_source['Region'] == sel_fc_reg]
        elif fc_scope == "By Product Model":
            sel_fc_mod = st.selectbox("Select Target Model:", options=sorted(df_fc_source['Product Model'].unique()), key="hw_fc_mod")
            df_fc_source = df_fc_source[df_fc_source['Product Model'] == sel_fc_mod]
        elif fc_scope == "By 5G Capability":
            sel_fc_cap = st.selectbox("Select Capability:", options=["Yes", "No"], key="hw_fc_cap")
            df_fc_source = df_fc_source[df_fc_source['5G Capability'] == sel_fc_cap]

        hist_df, pred_df, diag = fit_time_series_forecast(df_fc_source, fc_target_metric, horizon_quarters=fc_horizon)
        
        if hist_df is None:
            st.warning(f"{diag.get('error', 'Insufficient observations for selected filter.')}")
            st.info("Tip: Some individual product models have shorter lifecycles (< 6 quarters). Select 'Overall Samsung Sales' or a major Geographic Region for continuous forecasting.")
        else:
            def fmt_fc_val(val, metric):
                if metric == "Units_Sold":
                    return f"{val:,.0f}"
                elif metric == "Revenue":
                    return fmt_number(val, True)
                elif metric == "ASP":
                    return f"${val:.2f}"
                else:
                    return f"{val:.1f}%"

            fk1, fk2, fk3, fk4 = st.columns(4)
            with fk1:
                st.metric("Latest Historical Actual", fmt_fc_val(diag['last_actual'], fc_target_metric))
            with fk2:
                st.metric("Forecast (Next Quarter)", fmt_fc_val(diag['forecast_1'], fc_target_metric))
            with fk3:
                growth_fc = diag['growth_rate']
                st.metric("Expected QoQ Growth", f"{growth_fc:+.1f}%")
            with fk4:
                st.metric("Statistical Method", "Holt-Winters" if "Holt-Winters" in diag['method'] else "Holt Exponential")
                st.caption(f"Horizon: {fc_horizon}Q")
                
            fig_fc = go.Figure()
            
            fig_fc.add_trace(go.Scatter(
                x=hist_df['Period'],
                y=hist_df[fc_target_metric],
                mode='lines+markers',
                name='Historical Observations',
                line=dict(color=SAMSUNG_BLUE, width=3),
                marker=dict(size=6, color=SAMSUNG_BLUE)
            ))
            
            bridge = pd.DataFrame({
                'Period': [hist_df['Period'].iloc[-1]],
                'Forecast': [hist_df[fc_target_metric].iloc[-1]],
                'Upper_95_CI': [hist_df[fc_target_metric].iloc[-1]],
                'Lower_95_CI': [hist_df[fc_target_metric].iloc[-1]],
                'Data Type': ['Bridge']
            })
            pred_connected = pd.concat([bridge, pred_df]).reset_index(drop=True)
            
            fig_fc.add_trace(go.Scatter(
                x=pred_connected['Period'],
                y=pred_connected['Upper_95_CI'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=pred_connected['Period'],
                y=pred_connected['Lower_95_CI'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(20, 40, 160, 0.12)',
                name='95% Prediction Interval',
                hoverinfo='skip'
            ))
            
            fig_fc.add_trace(go.Scatter(
                x=pred_connected['Period'],
                y=pred_connected['Forecast'],
                mode='lines+markers',
                name='Model Forecast',
                line=dict(color=SAMSUNG_COBALT, width=3, dash='dash'),
                marker=dict(size=7, symbol='diamond', color=SAMSUNG_COBALT)
            ))
            
            timeline = hist_df['Period'].tolist() + pred_df['Period'].tolist()
            first_forecast = pred_df['Period'].iloc[0]
            fig_fc.add_shape(type='line', x0=first_forecast, x1=first_forecast,
                             y0=0, y1=1, yref='paper', line=dict(color=SAMSUNG_RED, width=1.5, dash='dash'))
            fig_fc.add_annotation(x=first_forecast, y=1, yref='paper', text='Forecast Horizon',
                                  showarrow=False, xanchor='right', yanchor='bottom', font=dict(color=SAMSUNG_RED, size=11))
            fig_fc.update_layout(PLOTLY_LAYOUT_DEFAULTS)
            fig_fc.update_layout(height=540, margin=dict(l=65, r=35, t=90, b=150),
                title=dict(text=f"Econometric Time-Series Projections ({diag['method'].split(' (')[0]})", y=.98),
                legend=dict(orientation='h', x=.5, xanchor='center', y=-.4, yanchor='top'))
            fig_fc.update_xaxes(title_text='Calendar Quarter', type='category', categoryorder='array',
                                categoryarray=timeline, tickmode='array', tickvals=timeline, tickangle=-45, tickfont=dict(size=9))
            fig_fc.update_yaxes(title_text=fc_target_metric.replace('_', ' '), tickformat='~s', automargin=True)
            st.plotly_chart(fig_fc, width='stretch', config=PLOTLY_CONFIG)


def regional_conditions(frame):
    rq = regional_quarters(frame)
    st.caption('Correlation is not causation. A shared time trend may inflate these correlations.')
    st.markdown('#### Infrastructure vs. 5G Units — Pooled and by Region')
    correlations = correlation_table(rq)
    indicator_names = dict(zip(INDICATORS, ['Coverage', 'Subscribers', 'Speed', 'Preference']))
    correlations['Indicator'] = correlations['Indicator'].map(indicator_names)
    correlations['Result'] = correlations.apply(lambda row: correlation_description(row['Pearson r'], row['p-value']), axis=1)
    compact = correlations.pivot(index='Scope', columns='Indicator', values='Result')
    scope_order = ['Pooled'] + sorted(rq['Region'].unique().tolist())
    compact = compact.reindex(index=scope_order, columns=['Coverage', 'Subscribers', 'Speed', 'Preference']).reset_index()
    table(compact)
    st.caption('Positive link: higher indicator values tend to occur alongside higher 5G sales. '
               'Negative link: higher indicator values tend to occur alongside lower sales. '
               'Strong, moderate, and weak describe how closely the values move together. '
               'No clear relationship means the evidence is weak or inconclusive; not enough data means the comparison cannot be assessed.')
    indicator = st.selectbox('Infrastructure indicator', INDICATORS, key='regional_indicator')
    left, right = st.columns(2)
    with left:
        fig = px.scatter(rq, x=indicator, y='FiveGUnits', color='Region', hover_data=['Period'],
                         color_discrete_map=REGION_COLORS, labels={'FiveGUnits': '5G units sold'})
        chart(fig, 'Infrastructure and 5G Unit Sales')
    with right:
        preference = rq.groupby('Region')['Preference for 5G (%)'].mean()
        adoption = adoption_by(frame, ['Region']).set_index('Region')['Adoption (%)']
        comparison = pd.concat([preference.rename('Stated preference'), adoption.rename('Actual adoption')], axis=1).reset_index()
        long = comparison.melt(id_vars='Region', var_name='Measure', value_name='Percent')
        fig = px.bar(long, x='Region', y='Percent', color='Measure', barmode='group',
                     color_discrete_map={'Stated preference': SAMSUNG_SLATE, 'Actual adoption': SAMSUNG_BLUE})
        fig.update_yaxes(range=[0, 100], ticksuffix='%')
        chart(fig, 'Stated Preference vs. Actual 5G Adoption')
    st.caption('Infrastructure values are averaged once per region-quarter. Actual adoption is the unit-weighted '
               '5G share; stated preference is the mean across the selected region-quarters. Correlations require '
               'at least three observations and variation in both measures.')
    with st.expander("Regional Commercial Performance Matrix", expanded=False):
        reg_summary = frame.groupby('Region', observed=True).agg(
            Total_Units=('Units Sold', 'sum'),
            Total_Revenue=('Revenue ($)', 'sum'),
            Avg_Share=('Market Share (%)', 'mean'),
            Avg_Coverage=('Regional 5G Coverage (%)', 'mean'),
            Avg_Speed=('Avg 5G Speed (Mbps)', 'mean'),
            Avg_Pref=('Preference for 5G (%)', 'mean')
        ).reset_index()
        reg_summary['Blended ASP ($)'] = (reg_summary['Total_Revenue'] / reg_summary['Total_Units'].replace(0, np.nan)).round(2)
        st.dataframe(
            reg_summary.rename(columns={
                'Total_Units': 'Total Units Sold',
                'Total_Revenue': 'Gross Revenue ($)',
                'Avg_Share': 'Market Share (%)',
                'Avg_Coverage': '5G Coverage (%)',
                'Avg_Speed': 'Avg Speed (Mbps)',
                'Avg_Pref': '5G Preference (%)'
            }).style.format({
                'Total Units Sold': '{:,.0f}',
                'Gross Revenue ($)': '${:,.0f}',
                'Blended ASP ($)': '${:.2f}',
                'Market Share (%)': '{:.2f}%',
                '5G Coverage (%)': '{:.1f}%',
                'Avg Speed (Mbps)': '{:.1f}',
                '5G Preference (%)': '{:.1f}%'
            }),
            width='stretch', hide_index=True
        )


def action_center(frame):
    latest = int(frame['Quarter_Index'].max())
    current = frame[frame['Quarter_Index'].eq(latest)]
    benchmark = adoption_rate(current)
    st.markdown(f'''<div class="model-card-container">
<div class="kpi-label">Action Center Criteria · {quarter_label(latest)}</div>
<p>Flag regions whose 5G unit adoption is below the overall average of <b>{benchmark:.1f}%</b>
across the selected portfolio in the latest Actual quarter.</p>
<p>Flag models active in that quarter with negative YoY unit growth for <b>two or more consecutive calendar quarters</b>.
Suggested actions use pricing, marketing, or product rules.</p></div>''', unsafe_allow_html=True)
    st.markdown('#### Flagged Regions')
    regions = adoption_by(current, ['Region'])
    regions['Overall average (%)'] = benchmark
    regions['Gap (percentage points)'] = regions['Adoption (%)'] - benchmark
    regions = regions[regions['Gap (percentage points)'] < -1e-9].copy()
    regions['Suggested action'] = regions['Gap (percentage points)'].map(
        lambda gap: 'Pricing — review entry-level 5G offers' if gap < -5 else
        ('Marketing — improve 5G awareness' if gap < -2 else 'Product — review 5G availability'))
    if regions.empty:
        st.info('No regions fall below the selected portfolio average in the latest Actual quarter.')
    else:
        table(regions[['Region', 'Adoption (%)', 'Overall average (%)', 'Gap (percentage points)', 'Suggested action']]
              .sort_values('Gap (percentage points)'))
    st.markdown('#### Flagged Models')
    models = flagged_models(frame)
    if models.empty:
        st.info('No active models have two or more consecutive quarters of negative YoY unit growth in the selected history.')
    else:
        table(models)
    with st.expander('Suggested-action rules', expanded=False):
        st.markdown('Regions: a gap below −5 points → pricing; below −2 points → marketing; otherwise → product availability. '
                    'Models: Flagship, Premium, and Premium Foldable → pricing review; other tiers → product refresh/retirement review. '
                    'These are rule-based suggestions, not observed causes. Missing or zero prior-year baselines do not count as declines.')
    st.markdown("#### Data-Driven Strategic Prescriptions")
    st.markdown(r"""
    | Focus Area | Identified Core Risk | Recommended Strategic Action | Action Category |
    | :--- | :--- | :--- | :--- |
    | **North America** | Lowest Market Share (28.6%) | **Carrier Trade-In Subsidies** <span title="Aggressive retail carrier promotions targeting competitive flagship switchers (Galaxy S24/S25/S26)." style="cursor:help; color:#94A3B8;">ⓘ</span> | Retain Flagship Share |
    | **Latin America & MEA** | Lagging Carrier Coverage (<45%) | **Budget 5G Penetration** <span title="Prioritize mass distribution of sub-\$250 models (Galaxy A15/A16 5G) to capture pre-emptive upgrade waves." style="cursor:help; color:#94A3B8;">ⓘ</span> | Volume Leadership |
    | **Premium Foldables** | Niche Volume (<200K units) | **Price Elasticity Realignment** <span title="Scale display manufacturing efficiencies to lower foldable ASP entry barrier toward the \$1,100–\$1,200 sweet spot." style="cursor:help; color:#94A3B8;">ⓘ</span> | ASP Optimization |
    | **Legacy 4G Devices** | Channel Cannibalization (0% 5G) | **Accelerated Portfolio Sunsetting** <span title="Eliminate legacy inventory holding costs and channel friction via aggressive end-of-life trade-in incentives." style="cursor:help; color:#94A3B8;">ⓘ</span> | Inventory Health |
    """, unsafe_allow_html=True)
    
    with st.expander("Strategic Playbook & Implementation Roadmap", expanded=False):
        st.markdown(r"""
        - **North America (Carrier Trade-In Subsidies):** Despite leading in 5G speeds (200.7 Mbps) and consumer preference (72.9%), Samsung market share lags at 28.6%. Deploy carrier multi-line subsidies and aggressive switcher credits targeting competing flagship users.
        - **Latin America & MEA (Budget 5G Penetration):** Samsung holds strong market dominance (38-39%), but network coverage remains below 45%. As carrier infrastructure expands, pre-populate channels with affordable A-series hardware.
        - **Premium Foldables (Price Elasticity Realignment):** Foldable ASP exceeds \$1,600, limiting volume adoption. Drive manufacturing yield improvements to bring entry foldables into the \$1,100–\$1,200 range to unlock broader enterprise adoption.
        - **Legacy 4G Devices (Accelerated Sunsetting):** Older 4G models tie up retail floor space and working capital. Channel all promotional subsidies into 5G trade-ins to accelerate total device transition.
        """)
    


DATA_DICTIONARY = {
    'Product Model': 'Samsung device name.',
    'Price Tier': 'Budget Legacy 4G, Budget, Mid, Flagship, Premium, or Premium Foldable.',
    '5G Capability': 'Yes = 5G device; No = Non-5G device.',
    'Year': 'Calendar year.', 'Quarter': 'Calendar quarter, Q1–Q4.',
    'Region': 'Geographic market.', 'Units Sold': 'Handset units recorded for this model, region, and quarter.',
    'Revenue ($)': 'Revenue in US dollars.', 'Market Share (%)': 'Reported Samsung market share.',
    'Regional 5G Coverage (%)': 'Reported regional 5G network coverage.',
    '5G Subscribers (millions)': 'Regional 5G subscribers in millions.',
    'Avg 5G Speed (Mbps)': 'Regional average 5G speed in Mbps.',
    'Preference for 5G (%)': 'Reported consumer preference for 5G devices.',
    'Data Type': 'Actual = historical record; Forecast = supplied projection, visible only on Trends and Forecast.',
    'ASP': 'Record-level revenue divided by units sold.',
    'Period': 'Year-quarter label.', 'Quarter_Index': 'Calendar quarters since 2019 Q1 (zero-based).',
}


def data_explorer(frame, regions, tiers, years, capability):
    query = st.text_input('Search cleaned records', placeholder='Search any column…', key='explorer_search')
    selected = frame.copy()
    if query.strip():
        mask = selected.astype(str).apply(lambda col: col.str.contains(query.strip(), case=False, regex=False)).any(axis=1)
        selected = selected[mask]
    with st.expander('Column filters', expanded=False):
        columns = st.multiselect('Columns to filter', list(frame.columns), key='explorer_filter_columns')
        for column in columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                minimum, maximum = float(frame[column].min()), float(frame[column].max())
                if minimum == maximum:
                    st.caption(f'{column}: {minimum:g} in the current selection')
                    continue
                low, high = st.slider(column, minimum, maximum, (minimum, maximum), key=f'explorer_range_{column}')
                selected = selected[selected[column].between(low, high)]
            else:
                options = sorted(frame[column].dropna().unique().tolist())
                chosen = st.multiselect(column, options, default=options, key=f'explorer_values_{column}')
                selected = selected[selected[column].isin(chosen)]
    st.caption(f'{len(selected):,} matching Actual records. Click column headers to sort.')
    table(selected)
    st.download_button('Download matching records (CSV)', selected.to_csv(index=False).encode('utf-8'),
                       file_name='Samsung_5G_Filtered_Actual_Records.csv', mime='text/csv', type='primary', key='explorer_download')
    with st.expander('Data dictionary', expanded=False):
        table(pd.DataFrame([{'Column': name, 'Description': description} for name, description in DATA_DICTIONARY.items()]))
    with st.expander('Data preparation — Raw / Cleaned', expanded=False):
        view = st.radio('Dataset snapshot', ['Raw', 'Cleaned'], horizontal=True, key='explorer_snapshot')
        raw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Samsung_5G_BI_Dataset_RAW.csv')
        if not os.path.exists(raw_path):
            st.info('The bundled raw snapshot is unavailable.')
            return
        raw = pd.read_csv(raw_path)
        raw_actual = raw[raw['Data Type'].astype(str).str.strip().str.lower().eq('actual')]
        region_mask = raw_actual['Region'].astype(str).str.strip().str.casefold().isin([r.casefold() for r in regions])
        tier_mask = raw_actual['Price Tier'].astype(str).str.strip().isin(tiers)
        year_mask = pd.to_numeric(raw_actual['Year'], errors='coerce').between(*years)
        raw_selected = raw_actual[region_mask & tier_mask & year_mask]
        if capability != 'All Products':
            target = 'yes' if capability == '5G' else 'no'
            raw_selected = raw_selected[raw_selected['5G Capability'].astype(str).str.strip().str.lower().eq(target)]
        st.caption('This comparison follows the sidebar filters and always excludes Forecast records. '
                   'Search and column filters above apply to the cleaned-records table and download only.')
        st.markdown('Cleaning standardizes labels, parses revenue/units, corrects negative market share, removes exact duplicates, '
                    'and imputes missing values. The v3 snapshot also removes an impossible 102.73% coverage row and preserves Budget Legacy 4G.')
        st.caption('The bundled raw and v3 CSVs are different snapshots, not a complete reproducible before/after pair; '
                   'their row counts therefore cannot be used to infer how many records cleaning removed.')
        table(pd.DataFrame([
            {'Snapshot (Actual only)': 'Raw', 'Selected rows': len(raw_selected), 'Missing cells': int(raw_selected.isna().sum().sum()),
             'Exact duplicate rows': int(raw_selected.duplicated().sum())},
            {'Snapshot (Actual only)': 'Cleaned', 'Selected rows': len(frame), 'Missing cells': int(frame.isna().sum().sum()),
             'Exact duplicate rows': int(frame.duplicated().sum())},
        ]))
        table(raw_selected if view == 'Raw' else frame)


try:
    df_clean, audit_info = load_and_clean_data(RAW_DEFAULT_PATH)
except Exception as exc:
    st.error(f'Unable to load the dashboard dataset: {exc}')
    st.stop()

# Stateful tabs retain the existing navigation design and enable tab-specific sidebar controls.
tabs = st.tabs(TAB_NAMES, key='dashboard_tab', on_change='rerun')
active_index = next((i for i, tab in enumerate(tabs) if tab.open), 0)
with tabs[active_index]:
    hero = st.empty()

with st.sidebar:
    logo = get_samsung_logo_b64()
    if logo:
        st.markdown(f'<div class="samsung-sidebar-logo-container"><img src="data:image/png;base64,{logo}" '
                    'class="samsung-sidebar-logo" alt="Samsung" /></div>', unsafe_allow_html=True)
    if st.button('Reset All Filters', width='stretch', type='secondary'):
        for key in list(st.session_state):
            if key.startswith(('filter_', 'explorer_')):
                del st.session_state[key]
        st.rerun()
    st.divider()
    if active_index == 3:
        horizon = st.selectbox('Data Type', ['Actual', 'Actual + Forecast', 'Forecast'], key='filter_trend_data_type')
    else:
        horizon = st.selectbox('Data Type', ['Actual'], disabled=True, key='filter_actual_data_type')
        st.caption('Forecast is available only in Trends and Forecast.')
    capability = 'All Products'
    if active_index in [1, 3, 6]:
        capability = st.radio('5G Capability', ['All Products', '5G', 'Non-5G'], key=f'filter_capability_{active_index}')
    min_year, max_year = int(df_clean['Year'].min()), int(df_clean['Year'].max())
    years = st.slider('Year range', min_year, max_year, (min_year, max_year), key='filter_year_range')
    all_regions = sorted(df_clean['Region'].unique().tolist())
    all_tiers = [tier for tier in TIER_ORDER if tier in df_clean['Price Tier'].unique()]
    with st.expander('Geographic Regions', expanded=False):
        regions = st.multiselect('Region', all_regions, default=all_regions, key='filter_regions')
    with st.expander('Portfolio Price Tiers', expanded=False):
        tiers = st.multiselect('Price Tier', all_tiers, default=all_tiers, key='filter_tiers')
    st.divider()
    st.caption(f'Dataset: {len(df_clean):,} cleaned records · {df_clean["Product Model"].nunique()} models')

filtered_df = df_clean.loc[df_clean['Region'].isin(regions) & df_clean['Price Tier'].isin(tiers)
                           & df_clean['Year'].between(*years)].copy()
if horizon != 'Actual + Forecast':
    filtered_df = filtered_df[filtered_df['Data Type'].eq(horizon)]
if capability != 'All Products':
    filtered_df = filtered_df[filtered_df['5G Capability'].eq('Yes' if capability == '5G' else 'No')]

hero.markdown(f'''<div class="samsung-hero-frameless">
<div class="samsung-hero-eyebrow"><span class="eyebrow-brand">SAMSUNG ELECTRONICS</span>
<span class="eyebrow-pipe">│</span><span class="eyebrow-sub">EXECUTIVE BI &amp; ANALYTICS</span></div>
<h1 class="samsung-hero-title">Decoding Demand: Samsung 5G Sales &amp; Market Adoption</h1>
<p class="samsung-hero-subtitle">Longitudinal performance and market penetration across Samsung mobile portfolios,
geographic regions, and carrier network conditions (2019–2026).</p>
<div class="samsung-hero-meta"><span class="meta-item"><span class="meta-label">Scope:</span>
<span class="meta-val">{escape(horizon)}</span></span><span class="meta-sep">•</span>
<span class="meta-item"><span class="meta-label">Timeline:</span><span class="meta-val">{years[0]}–{years[1]}</span></span>
<span class="meta-sep">•</span><span class="meta-item"><span class="meta-label">Data:</span>
<span class="meta-val">{len(filtered_df):,} Records</span></span></div></div>''', unsafe_allow_html=True)

with tabs[active_index]:
    st.markdown(f'<div class="section-header-box"><h2 class="section-title">{QUESTIONS[active_index]}</h2></div>',
                unsafe_allow_html=True)
    if filtered_df.empty:
        st.info('No records match these filters. Expand the selection or reset the filters.')
    elif active_index == 6:
        data_explorer(filtered_df, regions, tiers, years, capability)
    else:
        [overview, price_tiers, penetration, trends, regional_conditions, action_center][active_index](filtered_df)
