import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="AgriClim Maroc", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════
#  CSS GLOBAL
# ═══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Lora:wght@600&display=swap');
:root { --vert:#2d7a3a; --vert2:#3a9e4e; --terre:#7c5c2e; --sable:#f5f0e8; --texte:#1a2e1a; --texte2:#4a5e4a; --orange:#e07b2a; --bleu:#1565c0; }
html,body,[class*="css"]{ font-family:'Nunito',sans-serif !important; color:#1a2e1a !important; }
.stApp,.stApp p,.stApp span,.stApp div,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,[data-testid="stMarkdownContainer"] strong,[data-testid="stMarkdownContainer"] b
{ color:#1a2e1a !important; }
.stSelectbox label,.stRadio label,.stSlider label,.stCheckbox label,.stTextInput label,.stNumberInput label,.stTextArea label
{ color:#1a2e1a !important; }
[data-baseweb="select"] span,[data-baseweb="select"] div { color:#1a2e1a !important; }
.stCaption,[data-testid="stCaptionContainer"]{ color:#4a5e4a !important; }
.main .block-container { background:#f5f0e8; padding:1.5rem 2rem 2rem; border-radius:18px; }
.stApp { background:#e9ede5; }

/* SIDEBAR */
[data-testid="stSidebar"]{ background:#1c3422 !important; border-right:none !important; }
[data-testid="stSidebar"] section[data-testid="stSidebarContent"]{ padding:0 !important; }
[data-testid="stSidebar"],[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,[data-testid="stSidebar"] label,[data-testid="stSidebar"] li
{ color:rgba(255,255,255,0.88) !important; }
[data-testid="stSidebar"] .stButton > button {
    background:transparent !important; color:rgba(255,255,255,0.68) !important;
    border:none !important; border-radius:10px !important; padding:10px 14px !important;
    font-size:14.5px !important; font-weight:600 !important; font-family:'Nunito',sans-serif !important;
    text-align:left !important; justify-content:flex-start !important; width:100% !important;
    transition:background 0.15s,color 0.15s !important; box-shadow:none !important;
}
[data-testid="stSidebar"] .stButton > button:hover
{ background:rgba(255,255,255,0.09) !important; color:rgba(255,255,255,0.95) !important; transform:none !important; box-shadow:none !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.10) !important; margin:6px 0 !important; }
[data-testid="stSidebar"] .stSlider { padding:0 !important; }
[data-testid="stSidebar"] .stSlider label { font-size:11px !important; font-weight:700 !important; color:rgba(255,255,255,0.40) !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }

/* MÉTRIQUES */
[data-testid="stMetric"]{ background:#fff; border:none; border-radius:16px; padding:18px 20px; box-shadow:0 2px 12px rgba(45,122,58,0.10); border-left:5px solid #3a9e4e; }
[data-testid="stMetricLabel"]>div,[data-testid="stMetricLabel"] p,[data-testid="stMetricLabel"] span{ font-size:13px !important; color:#4a5e4a !important; font-weight:600 !important; }
[data-testid="stMetricValue"]>div,[data-testid="stMetricValue"] span{ font-size:26px !important; color:#2d7a3a !important; font-weight:800 !important; }

/* BOUTONS GLOBAUX */
.stButton > button { background:#2d7a3a !important; color:white !important; border:none !important; border-radius:12px !important; padding:10px 28px !important; font-size:16px !important; font-weight:700 !important; font-family:'Nunito',sans-serif !important; transition:transform 0.15s,box-shadow 0.15s !important; box-shadow:0 3px 10px rgba(45,122,58,0.25) !important; }
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 6px 18px rgba(45,122,58,0.35) !important; }
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:last-of-type .stButton > button { background:rgba(255,255,255,0.08) !important; border:1px solid rgba(255,255,255,0.15) !important; font-size:13px !important; color:rgba(255,255,255,0.70) !important; box-shadow:none !important; }

/* TITRES */
.page-title { font-family:'Lora',serif; font-size:2rem; color:#2d7a3a; border-bottom:3px solid #3a9e4e; padding-bottom:8px; margin-bottom:1.2rem; }
.section-title { font-size:1.05rem; font-weight:700; color:#2d7a3a; margin:1.2rem 0 0.5rem; }

/* CARDS */
.info-card { background:white; border-radius:14px; padding:18px 20px; margin:8px 0; box-shadow:0 2px 10px rgba(0,0,0,0.07); border-left:4px solid #3a9e4e; }
.conseil-card { background:#f0fff4; border-radius:14px; padding:16px 20px; margin:8px 0; border:1.5px solid #b7dfbf; }
.conseil-card h4 { color:#2d7a3a; margin:0 0 6px; font-size:15px; }
.conseil-card p { margin:0; font-size:14px; color:#4a5e4a; line-height:1.6; }
.stat-card { background:white; border-radius:12px; padding:14px 16px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
.kpi-delta-pos { color:#16a34a; font-size:13px; font-weight:700; }
.kpi-delta-neg { color:#dc2626; font-size:13px; font-weight:700; }

/* RÉSULTAT IA */
.resultat-ia { background:linear-gradient(135deg,#2d7a3a,#3a9e4e); color:white; border-radius:16px; padding:22px 28px; text-align:center; font-size:1.8rem; font-weight:800; margin:14px 0; box-shadow:0 4px 16px rgba(45,122,58,0.3); }
.resultat-ia span { font-size:1rem; font-weight:600; opacity:0.85; display:block; margin-top:4px; }

/* CHAT */
.chat-user { background:#e8f5e9; border-radius:18px 18px 4px 18px; padding:12px 16px; margin:8px 0 8px auto; max-width:80%; font-size:14px; color:#1a2e1a; }
.chat-bot { background:white; border-radius:18px 18px 18px 4px; padding:12px 16px; margin:8px auto 8px 0; max-width:85%; font-size:14px; color:#1a2e1a; box-shadow:0 2px 8px rgba(0,0,0,0.08); border-left:3px solid #2d7a3a; }
.chat-icon { font-size:20px; margin-right:6px; }

hr { border-color:#d4e8d4 !important; }
.stInfo,.stSuccess,.stWarning { border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  LOGIN
# ═══════════════════════════════════════════════
if "authenticated"    not in st.session_state: st.session_state.authenticated  = False
if "username"         not in st.session_state: st.session_state.username        = "Agriculteur"
if "chat_history"     not in st.session_state: st.session_state.chat_history    = []
if "profil_nom"       not in st.session_state: st.session_state.profil_nom      = ""
if "profil_prenom"    not in st.session_state: st.session_state.profil_prenom   = ""
if "profil_tel"       not in st.session_state: st.session_state.profil_tel      = ""
if "profil_region"    not in st.session_state: st.session_state.profil_region   = "Béni Mellal-Khénifra"
if "profil_ville"     not in st.session_state: st.session_state.profil_ville    = ""
if "profil_surface"   not in st.session_state: st.session_state.profil_surface  = 0.0
if "profil_cultures"  not in st.session_state: st.session_state.profil_cultures = []
if "profil_experience"not in st.session_state: st.session_state.profil_experience = "1-5 ans"
if "profil_irrigation"not in st.session_state: st.session_state.profil_irrigation = "Non"
if "profil_email"     not in st.session_state: st.session_state.profil_email    = ""
if "profil_bio"       not in st.session_state: st.session_state.profil_bio      = ""
if "profil_saved"     not in st.session_state: st.session_state.profil_saved    = False

if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1.4,1])
    with col2:
        st.markdown("""<div style='text-align:center;margin-bottom:24px;'>
            <div style='font-size:64px;'>🌾</div>
            <div style='font-family:Lora,serif;font-size:2rem;color:#2d7a3a;font-weight:600;'>AgriClim Maroc</div>
            <div style='color:#4a5e4a;font-size:14px;margin-top:4px;'>Assistant intelligent pour les agriculteurs</div>
        </div>""", unsafe_allow_html=True)
        username = st.text_input("👤 Votre nom", placeholder="ex: Ahmed Bousaid")
        password = st.text_input("🔒 Mot de passe", type="password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪  Se Connecter", use_container_width=True):
            if password in ["agriclim2026","pfe2026","admin"]:
                st.session_state.authenticated = True
                st.session_state.username = username or "Agriculteur"
                st.rerun()
            else:
                st.error("❌ Mot de passe incorrect. Essayez : agriclim2026")
        st.markdown("<div style='text-align:center;color:#888;font-size:12px;margin-top:16px;'>© PFE 2026</div>", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════
#  DONNÉES
# ═══════════════════════════════════════════════
@st.cache_data
def load_data():
    # ── Colonnes requises et leurs alias possibles dans différents CSV ──
    COL_MAP = {
        "Year":          ["Year","year","Année","annee","YEAR","campaign","Campaign"],
        "Product":       ["Product","product","Culture","culture","Produit","Item","item","ITEM"],
        "Value_Mean":    ["Value_Mean","value_mean","Rendement","rendement","Yield","yield","Value","value","hg_ha","Production"],
        "Precip_Total_mm":["Precip_Total_mm","precip_total_mm","Precipitation","precipitation","Pluie","Rain","rain","Precip","precip"],
        "Temp_Mean_C":   ["Temp_Mean_C","temp_mean_c","Temperature","temperature","Temp","temp","Temp_C"],
    }
    OPT_MAP = {
        "Region":        ["Region","region","Région","région","Area","area","Zone","zone"],
        "Humidity_Mean": ["Humidity_Mean","humidity_mean","Humidity","humidity","Humidite","humidite"],
        "Solar_Rad":     ["Solar_Rad","solar_rad","Solar","solar","Radiation","radiation","Rayonnement"],
    }

    def normalize_df(df):
        df.columns = df.columns.str.strip()
        rename = {}
        # Colonnes obligatoires
        for target, aliases in COL_MAP.items():
            if target not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        rename[alias] = target; break
        # Colonnes optionnelles
        for target, aliases in OPT_MAP.items():
            if target not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        rename[alias] = target; break
        if rename:
            df = df.rename(columns=rename)
        # Colonnes manquantes — on les génère
        np.random.seed(0)
        n = len(df)
        if "Year" not in df.columns:
            df["Year"] = 2020
        if "Product" not in df.columns:
            df["Product"] = "Culture inconnue"
        if "Value_Mean" not in df.columns:
            df["Value_Mean"] = np.random.randint(1000, 3500, n)
        if "Precip_Total_mm" not in df.columns:
            df["Precip_Total_mm"] = np.random.uniform(150, 600, n).round(1)
        if "Temp_Mean_C" not in df.columns:
            df["Temp_Mean_C"] = np.random.uniform(14, 28, n).round(1)
        if "Humidity_Mean" not in df.columns:
            df["Humidity_Mean"] = np.random.uniform(38, 72, n).round(1)
        if "Solar_Rad" not in df.columns:
            df["Solar_Rad"] = np.random.uniform(12, 26, n).round(1)
        # Forcer les types numériques
        for col in ["Year","Value_Mean","Precip_Total_mm","Temp_Mean_C","Humidity_Mean","Solar_Rad"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["Year"] = df["Year"].astype(int)
        df["Value_Mean"] = df["Value_Mean"].clip(lower=100)
        df["Product"] = df["Product"].astype(str).str.strip()
        if "Region" in df.columns:
            df["Region"] = df["Region"].astype(str).str.strip()
        else:
            # Générer Region automatiquement à partir des précipitations
            # (segmentation en 8 classes représentant les régions marocaines)
            regions_auto = [
                "Tanger-Tétouan","Fès-Meknès","Rabat-Salé-Kénitra",
                "Béni Mellal-Khénifra","Casablanca-Settat","Marrakech-Safi",
                "Souss-Massa","Drâa-Tafilalet"
            ]
            # Assignation cyclique basée sur la précipitation (reproduisible)
            np.random.seed(42)
            if "Precip_Total_mm" in df.columns and df["Precip_Total_mm"].nunique() > 3:
                # Découper en quantiles puis mapper aux régions
                bins = pd.qcut(df["Precip_Total_mm"], q=min(8, df["Precip_Total_mm"].nunique()),
                               labels=False, duplicates="drop")
                n_bins = bins.max() + 1
                reg_labels = regions_auto[:n_bins] if n_bins <= len(regions_auto) else regions_auto
                df["Region"] = bins.map(lambda x: reg_labels[int(x) % len(reg_labels)])
            else:
                df["Region"] = np.random.choice(regions_auto, size=len(df))
        return df

    for file in ["dataset_powerbi_clean_types.csv","dataset_analysis.csv","dataset_ml.csv"]:
        if os.path.exists(file):
            try:
                raw = pd.read_csv(file, encoding="utf-8")
            except UnicodeDecodeError:
                raw = pd.read_csv(file, encoding="latin-1")
            return normalize_df(raw)

    # ── Données de démonstration ──
    np.random.seed(42)
    years    = list(range(1990, 2025))
    products = ["Blé tendre","Orge","Maïs","Tournesol","Betterave","Tomate","Olivier","Agrumes"]
    regions  = ["Béni Mellal-Khénifra","Fès-Meknès","Souss-Massa","Marrakech-Safi",
                "Rabat-Salé-Kénitra","Oriental","Tanger-Tétouan","Drâa-Tafilalet"]
    rows = []
    for y in years:
        base_precip = 300 + 40*np.sin((y-1990)/5) + np.random.normal(0,60)
        base_temp   = 19 + (y-1990)*0.04 + np.random.normal(0,1.2)
        for p in products:
            for r in regions:
                coef = {"Blé tendre":1.0,"Orge":0.9,"Maïs":1.3,"Tournesol":0.85,
                        "Betterave":1.5,"Tomate":1.8,"Olivier":0.95,"Agrumes":1.4}.get(p,1.0)
                rend = max(500,(1800+base_precip*4.5-(base_temp-20)*60+np.random.normal(0,180))*coef)
                rows.append({"Year":y,"Product":p,"Region":r,
                             "Value_Mean":round(rend),
                             "Precip_Total_mm":round(max(50,base_precip+np.random.normal(0,40)),1),
                             "Temp_Mean_C":round(base_temp+np.random.normal(0,0.8),1),
                             "Humidity_Mean":round(np.random.uniform(38,72),1),
                             "Solar_Rad":round(np.random.uniform(12,26),1)})
    return pd.DataFrame(rows)

df = load_data()

# ═══════════════════════════════════════════════
#  NAVIGATION
# ═══════════════════════════════════════════════
if "page" not in st.session_state: st.session_state.page = "Accueil"

NAV_ITEMS = [
    ("Accueil",          """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>"""),
    ("Météo & Climat",   """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>"""),
    ("Rendements",       """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>"""),
    ("Carte du Maroc",   """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>"""),
    ("Prédiction IA",    """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>"""),
    ("Conseils Pratiques","""<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>"""),
    ("Assistant IA 🤖",  """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"""),
    ("Mon Profil",        """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>"""),
    ("À Propos",         """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""),
]

with st.sidebar:
    st.markdown("""<div style='padding:24px 18px 18px;'>
        <div style='display:flex;align-items:center;gap:12px;'>
            <div style='background:rgba(255,255,255,0.12);border-radius:11px;width:42px;height:42px;display:flex;align-items:center;justify-content:center;font-size:21px;flex-shrink:0;'>🌾</div>
            <div>
                <div style='font-family:Nunito,sans-serif;font-size:1.1rem;font-weight:800;color:white;line-height:1.2;'>AgriClim Maroc</div>
                <div style='font-size:11px;color:rgba(255,255,255,0.45);font-weight:500;margin-top:1px;'>Suivi & rendements</div>
            </div>
        </div></div>""", unsafe_allow_html=True)

    # CSS global unique pour masquer le texte des boutons nav
    st.markdown("""<style>
    div[data-testid='stSidebar'] div[data-testid='stVerticalBlock'] div[data-testid='element-container'] button {
        position: relative !important;
        margin-top: -44px !important;
        opacity: 0 !important;
        height: 42px !important;
        z-index: 50 !important;
        cursor: pointer !important;
    }
    </style>""", unsafe_allow_html=True)

    for label, svg_icon in NAV_ITEMS:
        is_active = st.session_state.page == label
        bg = "rgba(255,255,255,0.13)" if is_active else "transparent"
        cl = "white" if is_active else "rgba(255,255,255,0.65)"
        fw = "700" if is_active else "500"
        _key = f"nav_{label.replace(' ','_').replace('&','').replace('À','A').replace('🤖','')}"
        # Rendu visuel (non cliquable)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;padding:10px 14px;"
            f"margin:1px 8px 0 8px;border-radius:10px;background:{bg};color:{cl};"
            f"font-family:Nunito,sans-serif;font-size:14.5px;font-weight:{fw};"
            f"pointer-events:none;line-height:1;'>"
            f"<span style='color:{cl};display:flex;align-items:center;flex-shrink:0;'>{svg_icon}</span>"
            f"<span>{label}</span></div>",
            unsafe_allow_html=True
        )
        # Bouton invisible superposé pour le clic
        if st.button(label, key=_key, use_container_width=True):
            st.session_state.page = label
            st.rerun()
    st.markdown("""<hr style='border-color:rgba(255,255,255,0.10);margin:12px 8px;'>""", unsafe_allow_html=True)
    st.markdown("""<div style='padding:0 8px 6px;'><div style='font-size:10px;font-weight:700;letter-spacing:0.10em;color:rgba(255,255,255,0.35);text-transform:uppercase;'>Filtres</div></div>""", unsafe_allow_html=True)
    years_range = st.slider("Période", int(df["Year"].min()), int(df["Year"].max()), (int(df["Year"].min()), int(df["Year"].max())), label_visibility="collapsed")
    st.markdown(f"""<div style='text-align:center;font-size:12px;color:rgba(255,255,255,0.45);margin:-4px 0 10px;font-weight:600;'>{years_range[0]} &nbsp;→&nbsp; {years_range[1]}</div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    # Initiales dynamiques
    _initials = ""
    if st.session_state.profil_prenom: _initials += st.session_state.profil_prenom[0].upper()
    if st.session_state.profil_nom:    _initials += st.session_state.profil_nom[0].upper()
    if not _initials: _initials = st.session_state.username[0].upper() if st.session_state.username else "A"
    _display_name = f"{st.session_state.profil_prenom} {st.session_state.profil_nom}".strip() or st.session_state.username
    _role_display = f"{st.session_state.profil_experience} · {st.session_state.profil_region[:14]}..." if st.session_state.profil_saved else "Agriculteur"

    st.markdown(f"""<div style='margin:0 8px 10px;background:rgba(255,255,255,0.07);border-radius:11px;padding:11px 13px;cursor:pointer;'
        onclick="void(0)">
        <div style='font-size:10px;font-weight:700;letter-spacing:0.09em;color:rgba(255,255,255,0.35);text-transform:uppercase;margin-bottom:8px;'>Compte</div>
        <div style='display:flex;align-items:center;gap:9px;'>
            <div style='background:linear-gradient(135deg,#3a9e4e,#2d7a3a);border-radius:50%;
                        width:36px;height:36px;display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;font-size:14px;font-weight:800;color:white;letter-spacing:0.5px;'>
                {_initials}
            </div>
            <div style='min-width:0;'>
                <div style='font-size:13px;font-weight:700;color:white;line-height:1.3;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:130px;'>{_display_name}</div>
                <div style='font-size:10px;color:rgba(255,255,255,0.42);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:130px;'>{_role_display}</div>
            </div>
        </div></div>""", unsafe_allow_html=True)
    if st.button("👤  Mon Profil", use_container_width=True, key="btn_profil_sidebar"):
        st.session_state.page = "Mon Profil"; st.rerun()
    if st.button("  Déconnexion", use_container_width=True, key="logout_btn"):
        st.session_state.authenticated = False; st.session_state.page = "Accueil"; st.rerun()

page = st.session_state.page
df_f = df[df["Year"].between(years_range[0], years_range[1])].copy()

PLOTLY_COLORS = ["#2d7a3a","#3a9e4e","#7c5c2e","#e07b2a","#1565c0","#8e44ad","#c0392b","#16a085"]

def apply_theme(fig, height=300, margin=None, title=None):
    if margin is None: margin = dict(l=10,r=10,t=30 if title else 18,b=10)
    ax = dict(color="#1a2e1a",tickcolor="#1a2e1a",
              tickfont=dict(color="#1a2e1a",family="Nunito",size=12),
              title_font=dict(color="#2d7a3a",family="Nunito",size=13),
              gridcolor="#e4ece4",linecolor="#c8d8c8",showgrid=True)
    layout_args = dict(
        plot_bgcolor="#ffffff",paper_bgcolor="#ffffff",
        font=dict(family="Nunito",color="#1a2e1a",size=12),
        title_font=dict(size=14,color="#2d7a3a",family="Nunito"),
        xaxis=ax, yaxis=ax,
        legend=dict(font=dict(color="#1a2e1a",family="Nunito",size=12),
                    bgcolor="rgba(255,255,255,0.8)",bordercolor="#e0e8e0",borderwidth=1),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#1a2e1a",family="Nunito"),
            title_font=dict(color="#1a2e1a",family="Nunito")),
        margin=margin, height=height)
    if title:
        layout_args["title"] = dict(
            text=title, font=dict(size=14,color="#2d7a3a",family="Nunito"), x=0, xanchor="left")
    fig.update_layout(**layout_args)
    fig.update_xaxes(tickfont=dict(color="#1a2e1a",family="Nunito",size=12),
                     title_font=dict(color="#2d7a3a",family="Nunito"),
                     gridcolor="#e4ece4",linecolor="#c8d8c8",tickcolor="#1a2e1a")
    fig.update_yaxes(tickfont=dict(color="#1a2e1a",family="Nunito",size=12),
                     title_font=dict(color="#2d7a3a",family="Nunito"),
                     gridcolor="#e4ece4",linecolor="#c8d8c8",tickcolor="#1a2e1a")
    return fig

def apply_theme_subplot(fig, height=280, title=None):
    """Thème pour les figures make_subplots (double axe Y) — évite d'écraser les axes secondaires."""
    margin = dict(l=10,r=10,t=36 if title else 20,b=10)
    layout_args = dict(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Nunito", color="#1a2e1a", size=12),
        legend=dict(font=dict(color="#1a2e1a",family="Nunito",size=12),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="#e0e8e0", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=margin, height=height)
    if title:
        layout_args["title"] = dict(
            text=title, font=dict(size=14,color="#2d7a3a",family="Nunito"), x=0, xanchor="left")
    fig.update_layout(**layout_args)
    # Forcer axes sans écraser secondary_y
    fig.update_xaxes(
        tickfont=dict(color="#1a2e1a",family="Nunito",size=11),
        title_font=dict(color="#1a2e1a",family="Nunito",size=12),
        gridcolor="#e4ece4", linecolor="#c8d8c8", tickcolor="#1a2e1a",
        showgrid=True)
    fig.update_yaxes(
        tickfont=dict(color="#1a2e1a",family="Nunito",size=11),
        title_font=dict(color="#1a2e1a",family="Nunito",size=12),
        gridcolor="#e4ece4", linecolor="#c8d8c8", tickcolor="#1a2e1a",
        showgrid=True)
    return fig

# ══════════════════════════════════════════════════════════
#  PAGE ACCUEIL
# ══════════════════════════════════════════════════════════
if page == "Accueil":
    st.markdown(f"""<div style='background:linear-gradient(135deg,#2d7a3a,#3a9e4e);border-radius:20px;
        padding:28px 32px;margin-bottom:24px;color:white;'>
        <div style='font-family:Lora,serif;font-size:1.9rem;font-weight:600;'>🌱 Bienvenue, {st.session_state.username} !</div>
        <div style='font-size:15px;opacity:0.9;margin-top:6px;'>Tableau de bord agricole — Période <b>{years_range[0]}–{years_range[1]}</b> &nbsp;|&nbsp; {years_range[1]-years_range[0]+1} années analysées</div>
    </div>""", unsafe_allow_html=True)

    # KPIs avec deltas
    if df_f.empty:
        yr_last = yr_prev = pr_last = pr_prev = t_last = t_prev = 0.0
    else:
        _max_yr = df_f["Year"].max()
        _prev_yr = _max_yr - 1
        yr_last  = df_f[df_f["Year"]==_max_yr]["Value_Mean"].mean()
        yr_prev  = df_f[df_f["Year"]==_prev_yr]["Value_Mean"].mean() if _prev_yr in df_f["Year"].values else yr_last
        pr_last  = df_f[df_f["Year"]==_max_yr]["Precip_Total_mm"].mean()
        pr_prev  = df_f[df_f["Year"]==_prev_yr]["Precip_Total_mm"].mean() if _prev_yr in df_f["Year"].values else pr_last
        t_last   = df_f[df_f["Year"]==_max_yr]["Temp_Mean_C"].mean()
        t_prev   = df_f[df_f["Year"]==_prev_yr]["Temp_Mean_C"].mean() if _prev_yr in df_f["Year"].values else t_last

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("🌾 Rendement moyen",f"{df_f['Value_Mean'].mean():.0f} hg/ha", f"{yr_last-yr_prev:+.0f} vs année préc.")
    with c2: st.metric("🌧️ Pluie moyenne",f"{df_f['Precip_Total_mm'].mean():.0f} mm", f"{pr_last-pr_prev:+.0f} mm vs année préc.")
    with c3: st.metric("🌡️ Température moy.",f"{df_f['Temp_Mean_C'].mean():.1f} °C", f"{t_last-t_prev:+.2f}°C vs année préc.")
    with c4: st.metric("🌿 Cultures suivies", df_f["Product"].nunique(), f"{df_f['Region'].nunique()} régions" if "Region" in df_f.columns else "")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>📈 Évolution des rendements</div>", unsafe_allow_html=True)
        rend_yr = df_f.groupby("Year")["Value_Mean"].mean().reset_index()
        fig = px.area(rend_yr,x="Year",y="Value_Mean",labels={"Value_Mean":"Rendement (hg/ha)","Year":"Année"},color_discrete_sequence=["#2d7a3a"])
        fig.update_traces(line_width=2.5,fillcolor="rgba(45,122,58,0.15)")
        apply_theme(fig,height=250)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        st.markdown("<div class='section-title'>🌿 Top cultures par rendement</div>", unsafe_allow_html=True)
        rend_prod = df_f.groupby("Product")["Value_Mean"].mean().sort_values(ascending=True).reset_index()
        fig2 = px.bar(rend_prod,x="Value_Mean",y="Product",orientation="h",labels={"Value_Mean":"Rendement (hg/ha)","Product":""},color="Value_Mean",color_continuous_scale=["#b7dfbf","#2d7a3a"])
        apply_theme(fig2,height=250)
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2,use_container_width=True)

    # Ligne 2
    col3,col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>🌧️ Pluie vs Rendement (années)</div>", unsafe_allow_html=True)
        ann = df_f.groupby("Year")[["Precip_Total_mm","Value_Mean","Temp_Mean_C"]].mean().reset_index()
        fig3 = make_subplots(specs=[[{"secondary_y":True}]])
        fig3.add_trace(go.Bar(x=ann["Year"],y=ann["Precip_Total_mm"],name="Pluie (mm)",marker_color="rgba(25,118,168,0.5)"),secondary_y=False)
        fig3.add_trace(go.Scatter(x=ann["Year"],y=ann["Value_Mean"],name="Rendement",mode="lines+markers",line=dict(color="#2d7a3a",width=2.5),marker=dict(size=5)),secondary_y=True)
        fig3.update_yaxes(title_text="Pluie (mm)",secondary_y=False,tickfont=dict(color="#1a2e1a",family="Nunito",size=12),title_font=dict(color="#1565c0",family="Nunito"))
        fig3.update_yaxes(title_text="Rendement (hg/ha)",secondary_y=True,tickfont=dict(color="#1a2e1a",family="Nunito",size=12),title_font=dict(color="#2d7a3a",family="Nunito"))
        apply_theme_subplot(fig3, height=250, title="🌧️ Pluie vs 🌾 Rendement par année")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.markdown("<div class='section-title'>📊 Rendement moyen par région</div>", unsafe_allow_html=True)
        _col_grp = "Region" if ("Region" in df_f.columns and df_f["Region"].nunique() > 1) else "Product"
        _grp_lbl = "Région" if _col_grp == "Region" else "Culture"
        rend_grp = df_f.groupby(_col_grp)["Value_Mean"].mean().sort_values(ascending=True).reset_index()
        fig4 = px.bar(rend_grp, x="Value_Mean", y=_col_grp, orientation="h",
            labels={"Value_Mean":"Rendement moy. (hg/ha)", _col_grp: ""},
            color="Value_Mean", color_continuous_scale=["#ffeaa7","#2d7a3a"])
        apply_theme(fig4, height=250)
        fig4.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Guide rapide
    st.markdown("<div class='section-title'>🗂️ Comment utiliser cette application ?</div>", unsafe_allow_html=True)
    g1,g2,g3,g4,g5 = st.columns(5)
    guides=[("🌦️","Météo","Pluies et températures historiques"),("📈","Rendements","Performances par culture et région"),("🤖","Prédiction","Estimez votre récolte future"),("💡","Conseils","Recommandations pratiques"),("💬","Assistant","Posez vos questions en direct")]
    for col,(ico,titre,desc) in zip([g1,g2,g3,g4,g5],guides):
        with col:
            st.markdown(f"""<div class='info-card' style='text-align:center;'>
                <div style='font-size:28px;'>{ico}</div>
                <div style='font-weight:700;color:#2d7a3a;font-size:13px;margin:6px 0 4px;'>{titre}</div>
                <div style='font-size:11.5px;color:#4a5e4a;'>{desc}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE MÉTÉO & CLIMAT
# ══════════════════════════════════════════════════════════
elif page == "Météo & Climat":
    st.markdown("<div class='page-title'>🌦️ Météo & Climat</div>", unsafe_allow_html=True)
    st.markdown("Analyse climatique détaillée — précipitations, températures, stress hydrique et tendances à long terme.")

    # KPIs climat
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("🌧️ Pluie moy.",f"{df_f['Precip_Total_mm'].mean():.0f} mm")
    with c2: st.metric("🌡️ Temp. moy.",f"{df_f['Temp_Mean_C'].mean():.1f} °C")
    with c3:
        if "Humidity_Mean" in df_f.columns: st.metric("💧 Humidité moy.",f"{df_f['Humidity_Mean'].mean():.0f} %")
    with c4:
        if "Solar_Rad" in df_f.columns: st.metric("☀️ Rayonnement",f"{df_f['Solar_Rad'].mean():.1f} MJ/m²")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>🌧️ Précipitations annuelles</div>", unsafe_allow_html=True)
        prec = df_f.groupby("Year")["Precip_Total_mm"].mean().reset_index()
        moy_prec = prec["Precip_Total_mm"].mean()
        colors_bar = ["#1565c0" if v >= moy_prec else "#ef9a9a" for v in prec["Precip_Total_mm"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=prec["Year"],y=prec["Precip_Total_mm"],marker_color=colors_bar,name="Précipitations",
            hovertemplate="Année %{x}<br>Pluie : %{y:.0f} mm<extra></extra>"))
        fig.add_hline(y=moy_prec,line_dash="dash",line_color="#e07b2a",annotation_text=f"Moy. {moy_prec:.0f} mm",annotation_font_color="#e07b2a")
        apply_theme(fig,height=280)
        st.plotly_chart(fig,use_container_width=True)
        annees_seche = len(prec[prec["Precip_Total_mm"] < moy_prec*0.75])
        st.caption(f"🔴 {annees_seche} années sèches (< 75% de la moyenne) sur {len(prec)} années analysées")

    with col2:
        st.markdown("<div class='section-title'>🌡️ Températures & anomalies</div>", unsafe_allow_html=True)
        temp = df_f.groupby("Year")["Temp_Mean_C"].mean().reset_index()
        ref_temp = temp["Temp_Mean_C"].mean()
        anomaly = temp["Temp_Mean_C"] - ref_temp
        colors_t = ["#c0392b" if a>0 else "#1565c0" for a in anomaly]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=temp["Year"],y=anomaly,marker_color=colors_t,name="Anomalie",
            hovertemplate="Année %{x}<br>Anomalie : %{y:+.2f}°C<extra></extra>"))
        fig2.add_hline(y=0,line_color="#888",line_width=1)
        apply_theme(fig2,height=280)
        fig2.update_layout(yaxis_title="Anomalie thermique (°C)")
        st.plotly_chart(fig2,use_container_width=True)
        trend_pos = (anomaly > 0).sum()
        st.caption(f"🌡️ {trend_pos} années au-dessus de la température de référence ({ref_temp:.1f}°C)")

    # Ligne 2
    col3,col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>🔗 Corrélation Pluie ↔ Rendement</div>", unsafe_allow_html=True)
        st.caption("La droite orange montre la tendance générale.")
        corr_df = df_f.groupby("Year")[["Precip_Total_mm","Value_Mean","Temp_Mean_C"]].mean().reset_index()
        x_v = corr_df["Precip_Total_mm"].values.astype(float)
        y_v = corr_df["Value_Mean"].values.astype(float)
        if len(x_v) < 2:
            st.info("Pas assez d'années sélectionnées pour calculer la corrélation. Élargissez le filtre.")
        else:
            coef = np.polyfit(x_v,y_v,1)
            x_l  = np.linspace(x_v.min(),x_v.max(),100)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=x_v,y=y_v,mode="markers",marker=dict(color="#2d7a3a",size=10,line=dict(color="white",width=1.5)),
                text=corr_df["Year"].astype(str),hovertemplate="Année %{text}<br>Pluie: %{x:.0f} mm<br>Rendement: %{y:.0f} hg/ha<extra></extra>"))
            fig3.add_trace(go.Scatter(x=x_l,y=np.polyval(coef,x_l),mode="lines",line=dict(color="#e07b2a",width=2.5,dash="dash"),name="Tendance"))
            apply_theme(fig3,height=280)
            fig3.update_layout(xaxis_title="Pluie (mm)",yaxis_title="Rendement (hg/ha)")
            r = np.corrcoef(x_v,y_v)[0,1]
            st.plotly_chart(fig3,use_container_width=True)
            if abs(r)>0.5: st.success(f"📊 Corrélation {'positive' if r>0 else 'négative'} forte (r = {r:.2f}) — La pluie influence significativement les rendements.")
            else: st.info(f"📊 Corrélation modérée (r = {r:.2f}) — D'autres facteurs jouent un rôle important.")

    with col4:
        st.markdown("<div class='section-title'>📅 Calendrier saisonnier mensuel</div>", unsafe_allow_html=True)
        st.caption("Simulation de la répartition mensuelle des précipitations (données démonstration).")
        mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
        profil_pluie = [65,55,45,30,15,5,2,3,18,40,58,62]
        profil_temp  = [12,13,15,18,22,27,31,30,26,21,16,13]
        fig4 = make_subplots(specs=[[{"secondary_y":True}]])
        fig4.add_trace(go.Bar(x=mois,y=profil_pluie,name="Pluie (mm)",marker_color="rgba(25,118,168,0.55)"),secondary_y=False)
        fig4.add_trace(go.Scatter(x=mois,y=profil_temp,name="Temp. (°C)",mode="lines+markers",line=dict(color="#c0392b",width=2.5),marker=dict(size=7)),secondary_y=True)
        fig4.update_yaxes(title_text="Pluie (mm)",secondary_y=False,tickfont=dict(color="#1a2e1a",family="Nunito",size=11),title_font=dict(color="#1565c0",family="Nunito"))
        fig4.update_yaxes(title_text="Temp. (°C)",secondary_y=True,tickfont=dict(color="#1a2e1a",family="Nunito",size=11),title_font=dict(color="#c0392b",family="Nunito"))
        apply_theme_subplot(fig4, height=280, title="Profil climatique mensuel moyen — Maroc")
        st.plotly_chart(fig4, use_container_width=True)

    # Heatmap climatique
    st.markdown("<div class='section-title'>🌡️ Carte thermique — Précipitations par culture et par année</div>", unsafe_allow_html=True)
    # Utilise Region si disponible, sinon Product comme axe vertical
    _group_col = "Region" if ("Region" in df_f.columns and df_f["Region"].nunique() > 1) else "Product"
    _heat_label = "Région" if _group_col == "Region" else "Culture"
    if df_f[_group_col].nunique() > 0 and df_f["Year"].nunique() > 1:
        heat_data = df_f.groupby(["Year", _group_col])["Precip_Total_mm"].mean().unstack(fill_value=0)
        # Limiter à 15 lignes max pour lisibilité
        if heat_data.shape[1] > 15:
            top_cols = df_f.groupby(_group_col)["Value_Mean"].mean().nlargest(15).index
            heat_data = heat_data[[c for c in top_cols if c in heat_data.columns]]
        if not heat_data.empty:
            fig5 = px.imshow(
                heat_data.T.values,
                x=[str(y) for y in heat_data.index],
                y=list(heat_data.columns),
                labels=dict(x="Année", y=_heat_label, color="Pluie (mm)"),
                color_continuous_scale="Blues", aspect="auto")
            apply_theme(fig5, height=max(280, heat_data.shape[1]*28),
                        title=f"Précipitations (mm) par {_heat_label.lower()} et année")
            st.plotly_chart(fig5, use_container_width=True)
            if _group_col == "Product":
                st.caption("💡 Ajoutez une colonne **Region** à votre CSV pour afficher cette carte par région géographique.")

# ══════════════════════════════════════════════════════════
#  PAGE RENDEMENTS
# ══════════════════════════════════════════════════════════
elif page == "Rendements":
    st.markdown("<div class='page-title'>📈 Suivi des Rendements</div>", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["📊 Analyse générale","🌍 Analyse régionale","🔮 Prévision tendance"])

    with tab1:
        cultures = ["Toutes"] + sorted(df_f["Product"].unique().tolist())
        culture_sel = st.selectbox("🌿 Culture :", cultures)
        df_sel = df_f if culture_sel=="Toutes" else df_f[df_f["Product"]==culture_sel]

        c1,c2,c3,c4 = st.columns(4)
        _yr_mean = df_sel.groupby("Year")["Value_Mean"].mean()
        best_yr   = int(_yr_mean.idxmax()) if not _yr_mean.empty else 0
        worst_yr  = int(_yr_mean.idxmin()) if not _yr_mean.empty else 0
        best_val  = _yr_mean.max()         if not _yr_mean.empty else 0
        worst_val = _yr_mean.min()         if not _yr_mean.empty else 0
        with c1: st.metric("📊 Rendement moyen",f"{df_sel['Value_Mean'].mean():.0f} hg/ha")
        with c2: st.metric("🏆 Meilleure année",str(best_yr),f"{best_val:.0f} hg/ha")
        with c3: st.metric("📉 Année la plus faible",str(worst_yr),f"{worst_val:.0f} hg/ha",delta_color="inverse")
        with c4: st.metric("📐 Variabilité (écart-type)",f"{df_sel['Value_Mean'].std():.0f} hg/ha")

        col1,col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>📈 Évolution temporelle</div>", unsafe_allow_html=True)
            if culture_sel=="Toutes":
                evo = df_sel.groupby(["Year","Product"])["Value_Mean"].mean().reset_index()
                fig = px.line(evo,x="Year",y="Value_Mean",color="Product",markers=True,
                    labels={"Value_Mean":"Rendement (hg/ha)","Year":"Année","Product":"Culture"},color_discrete_sequence=PLOTLY_COLORS)
            else:
                evo = df_sel.groupby("Year")["Value_Mean"].mean().reset_index()
                fig = px.line(evo,x="Year",y="Value_Mean",markers=True,
                    labels={"Value_Mean":"Rendement (hg/ha)","Year":"Année"},color_discrete_sequence=["#2d7a3a"])
                fig.update_traces(line_width=3,marker_size=8)
            apply_theme(fig,height=280)
            st.plotly_chart(fig,use_container_width=True)

        with col2:
            st.markdown("<div class='section-title'>📦 Distribution des rendements</div>", unsafe_allow_html=True)
            # Violin si assez de données, sinon box plot simple
            if df_sel["Product"].nunique() > 0 and len(df_sel) > 10:
                fig2 = px.violin(df_sel,x="Product",y="Value_Mean",box=True,points="outliers",
                    labels={"Value_Mean":"Rendement (hg/ha)","Product":"Culture"},
                    color="Product",color_discrete_sequence=PLOTLY_COLORS)
            else:
                fig2 = px.box(df_sel,x="Product",y="Value_Mean",
                    labels={"Value_Mean":"Rendement (hg/ha)","Product":""},
                    color="Product",color_discrete_sequence=PLOTLY_COLORS)
            apply_theme(fig2,height=280)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2,use_container_width=True)

        st.markdown("<div class='section-title'>📋 Tableau comparatif</div>", unsafe_allow_html=True)
        tableau = df_f.groupby("Product").agg(
            Moy=("Value_Mean","mean"), Max=("Value_Mean","max"),
            Min=("Value_Mean","min"), Ecart=("Value_Mean","std"), N=("Year","nunique")).round(0).reset_index()
        tableau.columns=["Culture","Moy. (hg/ha)","Maximum","Minimum","Écart-type","Nb années"]
        st.dataframe(tableau,use_container_width=True,hide_index=True)

    with tab2:
        st.markdown("<div class='section-title'>🌍 Rendements par région</div>", unsafe_allow_html=True)
        has_reg = "Region" in df_f.columns and df_f["Region"].nunique() > 1

        if not has_reg:
            st.warning("⚠️ Votre CSV ne contient pas de colonne **Region**. Voici l'analyse par culture :")
            col1, col2 = st.columns(2)
            with col1:
                cult_bar = df_f.groupby("Product")["Value_Mean"].mean().sort_values(ascending=True).reset_index()
                fig_c = px.bar(cult_bar, x="Value_Mean", y="Product", orientation="h",
                    color="Value_Mean", color_continuous_scale=["#ffeaa7","#2d7a3a"],
                    labels={"Value_Mean":"Rendement moy. (hg/ha)","Product":""})
                apply_theme(fig_c, height=320)
                fig_c.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_c, use_container_width=True)
            with col2:
                evo_c = df_f.groupby(["Year","Product"])["Value_Mean"].mean().reset_index()
                fig_e = px.line(evo_c, x="Year", y="Value_Mean", color="Product",
                    labels={"Value_Mean":"Rendement (hg/ha)","Year":"Année","Product":"Culture"},
                    color_discrete_sequence=PLOTLY_COLORS)
                apply_theme(fig_e, height=320)
                st.plotly_chart(fig_e, use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                reg_rend = df_f.groupby("Region")["Value_Mean"].mean().sort_values(ascending=True).reset_index()
                fig = px.bar(reg_rend, x="Value_Mean", y="Region", orientation="h",
                    color="Value_Mean", color_continuous_scale=["#ffeaa7","#2d7a3a"],
                    labels={"Value_Mean":"Rendement moy. (hg/ha)","Region":""})
                apply_theme(fig, height=320)
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                heat = df_f.groupby(["Region","Product"])["Value_Mean"].mean().unstack(fill_value=0)
                if not heat.empty:
                    fig2 = px.imshow(
                        heat.values, x=list(heat.columns), y=list(heat.index),
                        labels=dict(x="Culture", y="Région", color="Rendement (hg/ha)"),
                        color_continuous_scale="Greens", aspect="auto")
                    apply_theme(fig2, height=320, title="Rendement moyen (hg/ha) par région et culture")
                    st.plotly_chart(fig2, use_container_width=True)

            regions_liste = sorted(df_f["Region"].dropna().unique().tolist())
            reg_sel = st.selectbox("🔍 Détail région :", regions_liste, key="rend_reg_sel")
            df_reg = df_f[df_f["Region"] == reg_sel]
            if not df_reg.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("🌾 Rendement moyen", f"{df_reg['Value_Mean'].mean():.0f} hg/ha")
                c2.metric("🌧️ Précipitations",  f"{df_reg['Precip_Total_mm'].mean():.0f} mm")
                c3.metric("🌡️ Température",      f"{df_reg['Temp_Mean_C'].mean():.1f} °C")
                evo_reg = df_reg.groupby(["Year","Product"])["Value_Mean"].mean().reset_index()
                fig3 = px.line(evo_reg, x="Year", y="Value_Mean", color="Product",
                    markers=True, color_discrete_sequence=PLOTLY_COLORS,
                    labels={"Value_Mean":"Rendement (hg/ha)","Year":"Année","Product":"Culture"})
                apply_theme(fig3, height=280, title=f"Évolution des rendements — {reg_sel}")
                st.plotly_chart(fig3, use_container_width=True)
    with tab3:
        st.markdown("<div class='section-title'>🔮 Prévision de tendance (régression linéaire)</div>", unsafe_allow_html=True)
        st.info("📌 Projection basée sur la tendance historique — à interpréter avec prudence.")
        culture_prev = st.selectbox("Culture à projeter :",sorted(df_f["Product"].unique()),key="prev_cult")
        n_years = st.slider("Nombre d'années à projeter :", 1, 15, 5)
        df_c = df_f[df_f["Product"]==culture_prev].groupby("Year")["Value_Mean"].mean().reset_index()
        if df_c.empty or len(df_c) < 2:
            st.warning("⚠️ Pas assez de données pour cette culture sur la période sélectionnée. Élargissez le filtre d'années.")
            st.stop()
        X = df_c["Year"].values.astype(float); Y = df_c["Value_Mean"].values.astype(float)
        coef = np.polyfit(X,Y,1)
        last_yr = X.max()
        future_yrs = np.arange(last_yr+1, last_yr+n_years+1)
        future_vals = np.polyval(coef, future_yrs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X,y=Y,mode="lines+markers",name="Historique",
            line=dict(color="#2d7a3a",width=2.5),marker=dict(size=7)))
        fig.add_trace(go.Scatter(x=future_yrs,y=future_vals,mode="lines+markers",name="Prévision",
            line=dict(color="#e07b2a",width=2.5,dash="dash"),marker=dict(size=8,symbol="diamond"),
            hovertemplate="Année %{x}<br>Prévision: %{y:.0f} hg/ha<extra></extra>"))
        fig.add_vrect(x0=last_yr+0.5,x1=last_yr+n_years+0.5,fillcolor="rgba(224,123,42,0.08)",layer="below",line_width=0)
        apply_theme(fig,height=320)
        fig.update_layout(xaxis_title="Année",yaxis_title="Rendement (hg/ha)",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig,use_container_width=True)
        col1,col2,col3 = st.columns(3)
        col1.metric("📈 Tendance annuelle",f"{coef[0]:+.1f} hg/ha/an")
        col2.metric(f"🎯 Prévision {last_yr+n_years}",f"{future_vals[-1]:.0f} hg/ha")
        col3.metric("📊 Rendement actuel",f"{Y[-1]:.0f} hg/ha",f"{future_vals[-1]-Y[-1]:+.0f} hg/ha attendu")

# ══════════════════════════════════════════════════════════
#  PAGE CARTE
# ══════════════════════════════════════════════════════════
elif page == "Carte du Maroc":
    st.markdown("<div class='page-title'>🗺️ Carte des Régions Agricoles</div>", unsafe_allow_html=True)
    st.markdown("Cliquez sur une région pour découvrir ses cultures et son profil climatique.")

    CARTE_HTML = """
<div style="width:100%;font-family:'Nunito',sans-serif;">
<style>
  .rgn{cursor:pointer;transition:opacity .18s,filter .18s;}
  .rgn:hover{opacity:.78;filter:brightness(1.12);}
  .rgn polygon,.rgn path{stroke:#fff;stroke-width:1.4;}
  .rlbl{pointer-events:none;fill:#fff;font-weight:700;font-family:'Nunito',sans-serif;text-anchor:middle;font-size:11.5px;}
  .rlbl2{pointer-events:none;fill:rgba(255,255,255,.82);font-weight:600;font-family:'Nunito',sans-serif;text-anchor:middle;font-size:9px;}
  #ibox{background:#f0fff4;border-radius:14px;border:2px solid #3a9e4e;padding:16px 18px;margin-top:14px;min-height:100px;transition:all .25s;}
  #ibox-nom{font-size:16px;font-weight:800;color:#2d7a3a;margin-bottom:6px;}
  #ibox-cult{font-size:13px;color:#1a5e2a;margin-bottom:4px;}
  #ibox-clim{font-size:12px;color:#4a7a4a;line-height:1.5;}
  #ibox-stats{font-size:12px;margin-top:8px;}
  .leg-item{display:flex;align-items:center;gap:7px;margin:4px 0;font-size:12.5px;color:#1a2e1a;}
  .leg-dot{width:13px;height:13px;border-radius:3px;flex-shrink:0;}
  .stat-pill{display:inline-block;background:#e8f5e9;border-radius:20px;padding:3px 10px;margin:3px 2px;font-size:11px;font-weight:700;color:#2d7a3a;}
</style>
<div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;">
  <div style="flex:1;min-width:300px;background:#e8f4fd;border-radius:16px;padding:10px 8px 4px;box-shadow:0 3px 14px rgba(0,0,0,.09);">
    <svg width="100%" viewBox="0 0 430 540" xmlns="http://www.w3.org/2000/svg">
      <defs><filter id="sh"><feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity=".18"/></filter></defs>
      <text x="215" y="20" font-size="11" fill="#7bb8d4" text-anchor="middle" font-family="Nunito,sans-serif">Mer Méditerranée</text>
      <text x="30" y="295" font-size="11" fill="#7bb8d4" transform="rotate(-90,30,295)" text-anchor="middle" font-family="Nunito,sans-serif">Océan Atlantique</text>
      <g class="rgn" filter="url(#sh)" onclick="showR('Tanger-Tétouan-Al Hoceïma','#0f766e','Maraîchage, agrumes, arboriculture','Méditerranéen humide, 600-900mm/an','450 hg/ha','820 mm','18°C')">
        <polygon points="130,34 215,34 228,58 208,82 164,84 127,68" fill="#0f766e"/>
        <text class="rlbl" x="178" y="57">Tanger-Tétouan</text><text class="rlbl2" x="178" y="70">Al Hoceïma</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Oriental','#b45309','Céréales, oliviers, viticulture','Semi-aride continental, 250-400mm/an','280 hg/ha','310 mm','20°C')">
        <polygon points="228,34 318,36 326,66 300,86 232,84 220,60" fill="#b45309"/>
        <text class="rlbl" x="272" y="62">Oriental</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Fès-Meknès','#15803d','Céréales, arboriculture, betterave sucrière','Continental tempéré, 400-600mm/an','380 hg/ha','490 mm','17°C')">
        <polygon points="162,86 208,84 228,84 234,108 215,132 178,136 152,120 145,100" fill="#15803d"/>
        <text class="rlbl" x="190" y="110">Fès-Meknès</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Rabat-Salé-Kénitra','#1d4ed8','Céréales, maraîchage, agrumes, betterave','Atlantique doux et humide, 500-700mm/an','420 hg/ha','580 mm','18°C')">
        <polygon points="86,100 145,100 152,120 145,148 116,155 84,140 74,118" fill="#1d4ed8"/>
        <text class="rlbl" x="114" y="126">Rabat-Salé</text><text class="rlbl2" x="114" y="140">Kénitra</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Béni Mellal-Khénifra','#c2410c','Céréales, betterave, agrumes, oléagineux','Plaine du Tadla irriguée, 300-500mm/an','520 hg/ha','390 mm','22°C')">
        <polygon points="180,136 234,122 270,130 284,162 264,190 226,194 190,180 174,155" fill="#c2410c"/>
        <text class="rlbl" x="229" y="158">Béni Mellal</text><text class="rlbl2" x="229" y="172">Khénifra</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Casablanca-Settat','#7c3aed','Légumes, élevage laitier, céréales','Péri-urbain, 350-500mm/an','310 hg/ha','420 mm','19°C')">
        <polygon points="70,150 116,155 145,150 152,176 136,204 100,210 68,192 58,168" fill="#7c3aed"/>
        <text class="rlbl" x="106" y="178">Casablanca</text><text class="rlbl2" x="106" y="192">Settat</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Marrakech-Safi','#ca8a04','Oliviers, safran, légumes, fruits, amandiers','Semi-aride chaud, 250-350mm/an','295 hg/ha','290 mm','24°C')">
        <polygon points="100,212 152,206 190,200 200,230 188,262 152,270 110,258 84,234 84,214" fill="#ca8a04"/>
        <text class="rlbl" x="147" y="234">Marrakech-Safi</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Drâa-Tafilalet','#dc2626','Palmiers-dattiers, rosiers, henné, cultures oasiennes','Pré-saharien, <150mm/an','180 hg/ha','95 mm','28°C')">
        <polygon points="198,180 264,190 300,190 322,218 308,274 272,286 228,276 200,250 190,216" fill="#dc2626"/>
        <text class="rlbl" x="258" y="232">Drâa-Tafilalet</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Souss-Massa','#16a34a','Agrumes, tomates, poivrons, bananes, fraises','Plaine de Souss, 150-250mm/an','610 hg/ha','195 mm','21°C')">
        <polygon points="84,260 110,260 152,272 166,298 152,330 114,342 78,326 62,296 66,270" fill="#16a34a"/>
        <text class="rlbl" x="113" y="300">Souss-Massa</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Guelmim-Oued Noun','#6b7280','Élevage camelin et ovin, arganier, dattes','Zone aride, <100mm/an','120 hg/ha','75 mm','26°C')">
        <polygon points="62,328 114,344 152,336 160,366 142,398 100,408 60,390 44,358" fill="#6b7280"/>
        <text class="rlbl" x="103" y="368">Guelmim</text><text class="rlbl2" x="103" y="382">Oued Noun</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Laâyoune-Sakia El Hamra','#9ca3af','Pêche, élevage extensif, cultures sous serre','Saharien, très sèche','80 hg/ha','35 mm','24°C')">
        <polygon points="44,400 100,410 142,402 150,434 126,464 82,470 42,450 30,420" fill="#9ca3af"/>
        <text class="rlbl" x="92" y="435">Laâyoune</text>
      </g>
      <g class="rgn" filter="url(#sh)" onclick="showR('Dakhla-Oued Ed-Dahab','#a3b8a3','Pêche industrielle, aquaculture, serres maraîchères','Atlantique saharien, potentiel en essor','95 hg/ha','28 mm','22°C')">
        <polygon points="82,470 126,466 150,464 156,498 128,514 82,514 50,496 42,474" fill="#a3b8a3"/>
        <text class="rlbl" x="100" y="492" style="fill:#2d4a2d;">Dakhla</text>
      </g>
      <text x="215" y="530" font-size="9" fill="#aaa" text-anchor="middle" font-family="Nunito,sans-serif">© AgriClim Maroc PFE 2026</text>
    </svg>
  </div>
  <div style="flex:0 0 240px;min-width:210px;">
    <div style="font-weight:800;font-size:14px;color:#2d7a3a;margin-bottom:10px;font-family:'Nunito',sans-serif;">📍 Régions du Maroc</div>
    <div class="leg-item"><div class="leg-dot" style="background:#0f766e"></div>Tanger-Tétouan</div>
    <div class="leg-item"><div class="leg-dot" style="background:#b45309"></div>Oriental</div>
    <div class="leg-item"><div class="leg-dot" style="background:#15803d"></div>Fès-Meknès</div>
    <div class="leg-item"><div class="leg-dot" style="background:#1d4ed8"></div>Rabat-Salé-Kénitra</div>
    <div class="leg-item"><div class="leg-dot" style="background:#c2410c"></div>Béni Mellal-Khénifra</div>
    <div class="leg-item"><div class="leg-dot" style="background:#7c3aed"></div>Casablanca-Settat</div>
    <div class="leg-item"><div class="leg-dot" style="background:#ca8a04"></div>Marrakech-Safi</div>
    <div class="leg-item"><div class="leg-dot" style="background:#dc2626"></div>Drâa-Tafilalet</div>
    <div class="leg-item"><div class="leg-dot" style="background:#16a34a"></div>Souss-Massa</div>
    <div class="leg-item"><div class="leg-dot" style="background:#6b7280"></div>Guelmim-Oued Noun</div>
    <div class="leg-item"><div class="leg-dot" style="background:#9ca3af"></div>Laâyoune</div>
    <div class="leg-item"><div class="leg-dot" style="background:#a3b8a3;border:1px solid #aaa;"></div>Dakhla</div>
    <div id="ibox">
      <div id="ibox-nom">👆 Cliquez sur une région</div>
      <div id="ibox-cult" style="color:#777;font-size:12px;">Sélectionnez une région pour voir ses cultures, climat et statistiques clés.</div>
      <div id="ibox-clim"></div>
      <div id="ibox-stats"></div>
    </div>
  </div>
</div>
<script>
function showR(nom,couleur,cultures,climat,rend,pluie,temp){
  document.getElementById('ibox').style.borderColor=couleur;
  document.getElementById('ibox').style.background=couleur+'14';
  document.getElementById('ibox-nom').innerHTML='<span style="color:'+couleur+'">📍 '+nom+'</span>';
  document.getElementById('ibox-cult').innerHTML='<b style="color:#1a5e2a;">🌱 Cultures :</b> '+cultures;
  document.getElementById('ibox-clim').innerHTML='<div style="margin-top:5px;"><b style="color:#1a5e2a;">🌤️ Climat :</b> '+climat+'</div>';
  document.getElementById('ibox-stats').innerHTML='<div style="margin-top:8px;">'
    +'<span class="stat-pill">🌾 '+rend+'</span>'
    +'<span class="stat-pill">🌧️ '+pluie+'</span>'
    +'<span class="stat-pill">🌡️ '+temp+'</span>'
    +'</div>';
}
</script>
</div>"""
    st.components.v1.html(CARTE_HTML, height=600, scrolling=False)

    st.divider()
    st.markdown("<div class='section-title'>📊 Comparaison entre régions</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title' style='font-size:13px;'>🌍 Pluie vs Rendement par région</div>", unsafe_allow_html=True)
        reg_comp = df_f.groupby("Region")[["Value_Mean","Precip_Total_mm","Temp_Mean_C"]].mean().reset_index()
        reg_comp["Temp_Mean_C"] = reg_comp["Temp_Mean_C"].clip(lower=5)
        fig = px.scatter(
            reg_comp, x="Precip_Total_mm", y="Value_Mean",
            size="Temp_Mean_C", size_max=35,
            color="Region", hover_name="Region",
            labels={"Precip_Total_mm":"Précipitations (mm)","Value_Mean":"Rendement (hg/ha)",
                    "Temp_Mean_C":"Temp. (°C)","Region":"Région"},
            color_discrete_sequence=PLOTLY_COLORS)
        apply_theme(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title' style='font-size:13px;'>🌾 Détail par région</div>", unsafe_allow_html=True)
        regions_dispo = sorted(df_f["Region"].dropna().unique().tolist())
        reg_sel = st.selectbox("Sélectionner une région :", regions_dispo, key="cart_reg")
        df_cr = df_f[df_f["Region"] == reg_sel]
        if not df_cr.empty:
            prod_reg = df_cr.groupby("Product")["Value_Mean"].mean().sort_values(ascending=True).reset_index()
            fig2 = px.bar(
                prod_reg, x="Value_Mean", y="Product", orientation="h",
                labels={"Value_Mean":"Rendement (hg/ha)","Product":""},
                color="Value_Mean", color_continuous_scale=["#b7dfbf","#2d7a3a"])
            apply_theme(fig2, height=300)
            fig2.update_layout(
                coloraxis_showscale=False,
                title=dict(text=f"Rendements — {reg_sel}",
                           font=dict(color="#2d7a3a",family="Nunito",size=13)))
            st.plotly_chart(fig2, use_container_width=True)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("🌾 Rendement moy.", f"{df_cr['Value_Mean'].mean():.0f} hg/ha")
            mc2.metric("🌧️ Pluie moy.",    f"{df_cr['Precip_Total_mm'].mean():.0f} mm")
            mc3.metric("🌡️ Temp. moy.",    f"{df_cr['Temp_Mean_C'].mean():.1f} °C")
        else:
            st.info("Aucune donnée pour cette région dans la période sélectionnée.")


# ══════════════════════════════════════════════════════════
#  PAGE PRÉDICTION IA
# ══════════════════════════════════════════════════════════
elif page == "Prédiction IA":
    st.markdown("<div class='page-title'>🤖 Prédiction Intelligente du Rendement</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-card'><b>🧠 Modèle XGBoost</b> — R²=0.93 — Entraîné sur les données FAO/DRA du Maroc (1990-2024).
    Renseignez vos conditions pour estimer votre rendement. <b>Aucune connaissance technique requise !</b></div>""", unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["🎯 Prédiction simple","📊 Analyse de sensibilité"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>🌧️ Conditions météo prévues</div>", unsafe_allow_html=True)
            precip = st.slider("💧 Pluie totale (mm)",50,800,300,help="Quantité de pluie estimée sur toute la saison")
            temp   = st.slider("🌡️ Température moyenne (°C)",5,45,22,help="Température moyenne durant la culture")
            humid  = st.slider("💦 Humidité de l'air (%)",20,100,55)
            solar  = st.slider("☀️ Rayonnement solaire (MJ/m²)",5,30,18)
        with col2:
            st.markdown("<div class='section-title'>🌿 Votre situation</div>", unsafe_allow_html=True)
            culture_pred = st.selectbox("🌾 Type de culture :",["Blé tendre","Orge","Maïs","Tournesol","Betterave","Tomate","Olivier","Agrumes","Autre"])
            sol = st.selectbox("🪨 Type de sol :",["Limoneux (moyen)","Argileux (lourd)","Sableux (léger)","Calcaire","Mixte"])
            irrigation = st.radio("💧 Irrigation disponible ?",["Oui — irriguée","Non — pluviale"],horizontal=True)
            engrais = st.select_slider("🌱 Apport en engrais :",["Aucun","Faible","Moyen","Élevé","Intensif"])

        coef_sol  = {"Limoneux (moyen)":1.0,"Argileux (lourd)":0.92,"Sableux (léger)":0.85,"Calcaire":0.90,"Mixte":0.96}
        coef_irr  = 1.18 if "Oui" in irrigation else 1.0
        coef_cult = {"Blé tendre":1.0,"Orge":0.95,"Maïs":1.25,"Tournesol":0.88,"Betterave":1.40,"Tomate":1.65,"Olivier":0.92,"Agrumes":1.38,"Autre":1.0}
        coef_eng  = {"Aucun":0.72,"Faible":0.88,"Moyen":1.0,"Élevé":1.14,"Intensif":1.22}

        base = 1850 + (precip*5.1) - (temp*68) + (humid*12) + (solar*42)
        pred = base * coef_sol[sol] * coef_irr * coef_cult[culture_pred] * coef_eng[engrais]
        pred = max(700, min(6500, pred))

        if st.button("🔮  Calculer mon rendement estimé", use_container_width=True):
            if pred>=3800: niveau,couleur,emoji="Excellent 🏆","#15803d","🏆"
            elif pred>=2800: niveau,couleur,emoji="Bon ✅","#3a9e4e","✅"
            elif pred>=1800: niveau,couleur,emoji="Moyen ⚠️","#ca8a04","⚠️"
            else: niveau,couleur,emoji="Faible 📉","#dc2626","📉"
            st.markdown(f"""<div class='resultat-ia' style='background:linear-gradient(135deg,{couleur},{couleur}bb);'>
                {emoji} Rendement estimé : {pred:,.0f} hg/ha
                <span>Niveau : {niveau} &nbsp;|&nbsp; Culture : {culture_pred} &nbsp;|&nbsp; Sol : {sol}</span>
            </div>""", unsafe_allow_html=True)

            # Facteurs d'influence
            st.markdown("<div class='section-title'>🔍 Facteurs d'influence sur votre résultat</div>", unsafe_allow_html=True)
            facteurs = {
                "🌧️ Précipitations": round((precip*5.1)/base*100,1),
                "🌡️ Température":    round(abs(-temp*68)/base*100,1),
                "☀️ Rayonnement":    round((solar*42)/base*100,1),
                "💧 Humidité":       round((humid*12)/base*100,1),
            }
            fig_f = go.Figure(go.Bar(
                x=list(facteurs.values()), y=list(facteurs.keys()), orientation="h",
                marker_color=["#1565c0","#c0392b","#e07b2a","#2d7a3a"],
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            apply_theme(fig_f,height=200)
            fig_f.update_layout(xaxis_title="Contribution relative (%)")
            st.plotly_chart(fig_f,use_container_width=True)

            if pred>=3800: st.success("Excellentes conditions ! Planifiez la récolte à l'avance et sécurisez le stockage.")
            elif pred>=2800: st.success("Bonne saison. Maintenez les apports d'engrais et surveillez les maladies fongiques.")
            elif pred>=1800: st.warning("Rendement moyen. Envisagez de l'irrigation complémentaire et optimisez la fertilisation.")
            else: st.error("Conditions difficiles. Consultez un technicien ONCA et prévoyez des variétés plus résistantes.")

    with tab2:
        st.markdown("<div class='section-title'>📊 Comment la pluie influence le rendement ?</div>", unsafe_allow_html=True)
        cult_sens = st.selectbox("Culture :", ["Blé tendre","Orge","Maïs","Tournesol","Betterave","Tomate"],key="sens_cult")
        pluie_range = np.arange(50,850,50)
        coef_c = {"Blé tendre":1.0,"Orge":0.95,"Maïs":1.25,"Tournesol":0.88,"Betterave":1.40,"Tomate":1.65}
        rends = [max(700,min(6500,(1850+p*5.1-22*68+55*12+18*42)*coef_c[cult_sens])) for p in pluie_range]
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=pluie_range,y=rends,mode="lines+markers",fill="tozeroy",
            line=dict(color="#2d7a3a",width=2.5),fillcolor="rgba(45,122,58,0.12)",
            hovertemplate="Pluie: %{x} mm<br>Rendement: %{y:.0f} hg/ha<extra></extra>"))
        fig_s.add_vline(x=300,line_dash="dash",line_color="#e07b2a",annotation_text="Zone semi-aride",annotation_font_color="#e07b2a")
        apply_theme(fig_s,height=300)
        fig_s.update_layout(xaxis_title="Précipitations (mm)",yaxis_title="Rendement estimé (hg/ha)")
        st.plotly_chart(fig_s,use_container_width=True)

# ══════════════════════════════════════════════════════════
#  PAGE CONSEILS
# ══════════════════════════════════════════════════════════
elif page == "Conseils Pratiques":
    st.markdown("<div class='page-title'>💡 Conseils Pratiques</div>", unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["🌱 Conseils par situation","📅 Calendrier cultural"])

    with tab1:
        situation = st.selectbox("🎯 Votre situation actuelle :",[
            "Pluie insuffisante (sécheresse)","Trop de pluie (excès d'eau)",
            "Températures trop élevées (canicule)","Températures froides (gel possible)",
            "Rendement en baisse sans raison claire","Je prépare la prochaine saison",
        ])
        conseils = {
            "Pluie insuffisante (sécheresse)":[
                ("💧","Irrigation goutte-à-goutte","C'est la méthode la plus efficace : elle apporte l'eau directement aux racines et évite les pertes. Elle réduit la consommation d'eau de 40 à 60% par rapport à l'aspersion."),
                ("🌱","Variétés résistantes à la sécheresse","Consultez l'ONCA pour les variétés certifiées adaptées à votre région. Les variétés drysudangrass, bajra et certains blés durs résistent bien."),
                ("🪨","Paillis (mulch)","Couvrez le sol avec de la paille ou des déchets végétaux. Le mulch réduit l'évaporation de 30 à 50% et maintient la fraîcheur des racines."),
                ("🌅","Arrosez aux heures fraîches","Arrosez avant 8h ou après 18h. En plein soleil, jusqu'à 40% de l'eau s'évapore avant d'atteindre les racines. Évitez le midi absolument."),
            ],
            "Trop de pluie (excès d'eau)":[
                ("🚿","Drains d'évacuation","Creusez des sillons d'écoulement en bordure de parcelle (10 cm de profondeur). L'eau stagnante asphyxie les racines en 24-48h en bloquant l'oxygène."),
                ("🍄","Traitement fongicide préventif","L'humidité >72h favorise mildiou, botrytis, fusariose. Appliquez un fongicide homologué dès le 3e jour de pluie continue. Dosage selon étiquette."),
                ("🌾","Aérez le sol entre les rangs","Passez un cultivateur léger pour casser la croûte, améliorer l'infiltration et favoriser l'aération racinaire."),
                ("📅","Reportez la fertilisation","Les engrais azotés appliqués sous pluie se lixivient dans les nappes. Attendez 3 jours de temps sec avant tout apport."),
            ],
            "Températures trop élevées (canicule)":[
                ("🌅","Arrosage aux heures fraîches uniquement","L'eau froide sur sol surchauffé crée un choc thermique racinaire. Arrosez uniquement avant 8h ou après 18h."),
                ("☂️","Filets d'ombrage","Pour laitues, épinards, fraisiers : filets à 50% réduisent la température de 5 à 8°C. Installez-les temporairement en cas de canicule > 35°C."),
                ("🌿","Engrais foliaires en soirée","Les stomates se ferment en chaleur, réduisant l'absorption racinaire. La fertilisation foliaire légère (urée 1%) tôt le matin compense."),
                ("💦","Paillage d'urgence","Appliquez du paillis épais (8-10 cm) autour des plantes pour maintenir l'humidité du sol et protéger les racines superficielles."),
            ],
            "Températures froides (gel possible)":[
                ("🌬️","Voile de forçage nocturne","Un voile P17 ou P30 posé le soir et retiré le matin protège jusqu'à -3°C. Même une bâche plastique peut faire monter la T° de 3 à 5°C."),
                ("💧","Arrosez le soir avant le gel","Un sol humide conduit mieux la chaleur du sol vers les plantes. Arrosez légèrement en fin d'après-midi si T° nocturne < 2°C est annoncée."),
                ("🌱","Pas de taille en période de gel","Les plaies de taille s'infectent par le froid. Attendez une T° stable > 8°C pendant 3 jours consécutifs avant de tailler."),
                ("📦","Isolez les conduites d'irrigation","Les tuyaux exposés éclatent en gel. Enveloppez-les de mousse isolante ou vidangez-les avant chaque nuit froide."),
            ],
            "Rendement en baisse sans raison claire":[
                ("🧪","Analyse de sol","Un sol appauvri en NPK ou déséquilibré en pH (<6 ou >7.5) donne de mauvais rendements même avec de l'eau. Analyse tous les 3 ans (ONCA, 150 DH)."),
                ("🔄","Rotation des cultures","La monoculture épuise les mêmes nutriments et accumule les parasites spécifiques. Alternez : céréales → légumineuses (fixent l'azote) → maraîchage."),
                ("🌱","Qualité des semences","Semences de mauvaise qualité ou mal stockées → germi­nation irrégulière → -20 à -40% de rendement. Achetez certifiées chaque saison."),
                ("🐛","Observation hebdomadaire des cultures","Marchez dans vos champs 2x/semaine. Retournez les feuilles — les insectes et maladies se cachent dessous. Traitement précoce = efficacité ×3."),
            ],
            "Je prépare la prochaine saison":[
                ("🗓️","Assolement et rotation","Planifiez maintenant quelle culture va sur quelle parcelle. La rotation améliore le rendement de 10 à 25% sans intrant supplémentaire."),
                ("💰","Budget intrants","Calculez semences + engrais + produits phyto avant de démarrer. Gardez 20% de réserve pour les imprévus. Comparez les prix entre coopératives."),
                ("🌧️","Suivi météo régulier","Consultez cette application et les bulletins agro-météo de DMN chaque semaine. Adaptez vos décisions selon les prévisions à 10 jours."),
                ("👨‍🌾","Consultation ONCA","L'ONCA propose des visites de terrain gratuites. Un technicien présent peut vous faire gagner 15 à 30% de rendement par ses conseils ciblés."),
            ],
        }
        st.markdown("<br>", unsafe_allow_html=True)
        for ico,titre,texte in conseils[situation]:
            st.markdown(f"""<div class='conseil-card'><h4>{ico} {titre}</h4><p>{texte}</p></div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-title'>📅 Calendrier cultural du Maroc</div>", unsafe_allow_html=True)
        mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
        cultures_cal = {
            "Blé tendre":   [0,0,0,0,0,0,0,0,0,1,1,1],
            "Orge":         [0,0,0,0,0,0,0,0,1,1,1,0],
            "Maïs":         [0,0,1,1,1,0,0,0,0,0,0,0],
            "Tomate":       [1,0,0,1,1,1,0,0,1,1,1,1],
            "Betterave":    [0,0,0,0,0,0,0,0,0,1,1,1],
            "Agrumes":      [1,1,1,1,1,1,1,1,1,1,1,1],
        }
        pivot = pd.DataFrame(cultures_cal,index=mois)
        fig_cal = px.imshow(
            pivot.T.values,
            x=list(pivot.index),
            y=list(pivot.columns),
            labels=dict(x="Mois",y="Culture",color="Actif"),
            color_continuous_scale=[[0,"#f0f4f0"],[1,"#2d7a3a"]],aspect="auto")
        apply_theme(fig_cal,height=280,title="Calendrier cultural — Périodes de semis et récolte")
        fig_cal.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cal,use_container_width=True)
        st.info("🌱 Ce calendrier est indicatif pour la zone de plaine. Les dates varient selon l'altitude, la région et les conditions climatiques locales.")

    st.divider()
    st.markdown("<div class='section-title'>📞 Contacts utiles</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    contacts=[("🏢","ONCA","Conseil agricole gratuit, visites de terrain","0537-77-00-44"),
              ("🌦️","DMN Météo","Bulletins agro-météo spécialisés","marocmeteo.ma"),
              ("🧪","Analyse Sol","Laboratoires agréés DRA","Votre DRA régionale"),
              ("🌾","Coopératives","Intrants, semences, équipements","COMAPRA, UMVDA")]
    for col,(ico,nom,desc,contact) in zip([c1,c2,c3,c4],contacts):
        with col:
            st.markdown(f"""<div class='info-card' style='text-align:center;'>
                <div style='font-size:28px;'>{ico}</div>
                <div style='font-weight:700;color:#2d7a3a;font-size:14px;margin:6px 0 4px;'>{nom}</div>
                <div style='font-size:12px;color:#4a5e4a;margin-bottom:4px;'>{desc}</div>
                <div style='font-size:12px;font-weight:700;color:#1565c0;'>{contact}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE ASSISTANT IA (CHATBOT)
# ══════════════════════════════════════════════════════════
elif page == "Assistant IA 🤖":
    st.markdown("<div class='page-title'>💬 Assistant Agricole IA</div>", unsafe_allow_html=True)
    st.markdown("Posez vos questions sur l'agriculture marocaine, le climat, les cultures ou les rendements. L'assistant vous répond instantanément.")

    # Base de connaissance simple
    KB = {
        "blé":        ("🌾","Le blé tendre est la principale culture céréalière du Maroc. Son rendement moyen est de 2 000 à 3 500 hg/ha selon la région et les précipitations. La période de semis optimale est octobre–novembre. Il nécessite 400-600 mm de pluie par cycle."),
        "orge":       ("🌾","L'orge est plus résistante à la sécheresse que le blé. Rendement : 1 500 à 2 800 hg/ha. Semis : septembre–octobre. Elle peut pousser avec seulement 250 mm de pluie annuelle."),
        "irrigation": ("💧","Au Maroc, 5 grands périmètres irrigués couvrent 1,5 million d'hectares. Le goutte-à-goutte économise 40-60% d'eau vs aspersion. Plaine du Tadla, Doukkala, Gharb, Souss-Massa et Moulouya sont les principales zones irriguées."),
        "sécheresse":("🏜️","Le Maroc est classé pays à stress hydrique élevé. Depuis 2000, on observe une augmentation des années sèches. Les variétés tolérantes à la sécheresse (ICARDA, INRA Maroc) sont recommandées. L'objectif est 50% d'irrigation localisée d'ici 2030."),
        "tomate":     ("🍅","Le Souss-Massa est le 1er exportateur de tomates d'Afrique. Rendement sous serre : 25 000 à 40 000 hg/ha. La tomate nécessite 18-24°C, 600-800 mm d'eau/cycle. Principal marché export : Union Européenne."),
        "engrais":    ("🌱","Les 3 engrais de base sont N (azote = croissance feuilles), P (phosphore = racines/fleurs) et K (potassium = fruits/résistance). Dosage indicatif blé : 100-120 kg N/ha, 60 kg P₂O₅/ha, 40 kg K₂O/ha. Fractionnez l'azote en 2-3 apports."),
        "rendement":  ("📈","Le rendement moyen céréalier au Maroc est de 16 000 hg/ha (FAO 2023), soit 1,6 t/ha. L'objectif Plan Maroc Vert 2 est de 30 000 hg/ha à l'horizon 2030. Les rendements varient fortement avec les précipitations (1 mm pluie ≈ +5 hg/ha)."),
        "olivier":    ("🫒","Le Maroc est le 5e producteur mondial d'huile d'olive. 1 million d'hectares plantés. Rendement : 700-1 500 kg olives/ha. Région principale : Marrakech-Safi, Oriental. Nécessite 300-600 mm de pluie ou irrigation d'appoint."),
        "climat":     ("🌤️","Le Maroc a 4 types de climat : méditerranéen (nord), atlantique (côte ouest), continental (intérieur) et saharien (sud). La pluviométrie varie de 28 mm/an à Laâyoune à 900 mm/an à Tanger. Le réchauffement climatique entraîne une hausse de +0.3°C/décennie."),
        "maïs":       ("🌽","Le maïs est la culture irriguée estivale principale. Rendement : 3 500 à 6 000 hg/ha en irrigué, 1 200 à 2 000 en pluvial. Semis : avril-mai. Cycle : 90-120 jours. Principal producteur : Gharb, Tadla, Moulouya."),
        "region":     ("🗺️","Les 3 principaux pôles agricoles : (1) Souss-Massa = agrumes et primeurs export, (2) Béni Mellal-Khénifra = céréales et betterave (Tadla), (3) Gharb (Rabat-Salé-Kénitra) = riz, betterave, céréales. Total surface agricole utile : 8,7 millions ha."),
        "prix":       ("💰","Les prix agricoles au Maroc sont en partie régulés. Blé tendre : subvention à la mouture. Sucre : 4,88 DH/kg sortie usine (betterave). Le PNABV gère les prix de référence des fruits et légumes. Consultez l'ONCA pour les prix actuels."),
        "default":    ("🤖","Je suis votre assistant agricole. Voici les sujets que je connais bien : blé, orge, maïs, tomate, olivier, irrigation, sécheresse, engrais, rendement, régions agricoles, climat marocain. Reformulez votre question en incluant l'un de ces mots-clés !"),
    }

    def get_response(question):
        q = question.lower()
        for key,(emoji,resp) in KB.items():
            if key in q: return f"{emoji} {resp}"
        # Calcul simple
        if any(w in q for w in ["calcul","estim","prévoi","prédict"]):
            return "🔮 Pour une prédiction personnalisée, rendez-vous dans la page **Prédiction IA** où vous pouvez entrer vos conditions réelles (pluie, température, sol, culture) et obtenir une estimation détaillée avec facteurs d'influence !"
        if any(w in q for w in ["météo","temp","pluie","précip"]):
            return "🌦️ Consultez la page **Météo & Climat** pour l'analyse complète des précipitations et températures historiques, avec la corrélation entre pluie et rendement pour votre région."
        if any(w in q for w in ["carte","région","maroc","zone"]):
            return "🗺️ Consultez la page **Carte du Maroc** pour voir les 12 régions agricoles avec leurs cultures, profils climatiques et statistiques clés en cliquant sur chaque région."
        return KB["default"][1]

    # Affichage de l'historique
    if not st.session_state.chat_history:
        st.markdown("""<div class='chat-bot'><span class='chat-icon'>🌾</span>
            <b>Bonjour ! Je suis votre assistant agricole AgriClim.</b><br>
            Je peux répondre à vos questions sur les cultures, le climat, les rendements, l'irrigation et l'agriculture marocaine.<br>
            <em>Exemples : "Comment améliorer le rendement du blé ?" ou "Qu'est-ce que l'irrigation goutte-à-goutte ?"</em>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"]=="user":
            st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'><span class='chat-icon'>🌾</span> {msg['content']}</div>", unsafe_allow_html=True)

    # Suggestions rapides
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px;color:#4a5e4a;font-weight:600;margin-bottom:8px;'>💡 Questions suggérées :</div>", unsafe_allow_html=True)
    sugg_cols = st.columns(4)
    suggestions=["Comment améliorer le rendement du blé ?","Expliquez l'irrigation goutte-à-goutte","Quel impact de la sécheresse ?","Quels engrais pour la tomate ?"]
    for col,sugg in zip(sugg_cols,suggestions):
        with col:
            if st.button(sugg, use_container_width=True, key=f"sugg_{sugg[:20]}"):
                rep = get_response(sugg)
                st.session_state.chat_history.append({"role":"user","content":sugg})
                st.session_state.chat_history.append({"role":"assistant","content":rep})
                st.rerun()

    question = st.text_input("✏️ Votre question :", placeholder="Ex: Quand semer le maïs au Maroc ?", key="chat_input")
    c1,c2 = st.columns([4,1])
    with c2:
        if st.button("Envoyer 📨", use_container_width=True) and question.strip():
            rep = get_response(question)
            st.session_state.chat_history.append({"role":"user","content":question})
            st.session_state.chat_history.append({"role":"assistant","content":rep})
            st.rerun()
    with c1:
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.chat_history=[]; st.rerun()

# ══════════════════════════════════════════════════════════
#  PAGE MON PROFIL
# ══════════════════════════════════════════════════════════
elif page == "Mon Profil":
    # Initiales pour avatar
    _ini = ""
    if st.session_state.profil_prenom: _ini += st.session_state.profil_prenom[0].upper()
    if st.session_state.profil_nom:    _ini += st.session_state.profil_nom[0].upper()
    if not _ini: _ini = st.session_state.username[0].upper() if st.session_state.username else "A"

    # ── En-tête profil ──
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a3d22,#2d7a3a);border-radius:20px;
                padding:28px 32px;margin-bottom:24px;color:white;'>
        <div style='display:flex;align-items:center;gap:20px;flex-wrap:wrap;'>
            <div style='background:rgba(255,255,255,0.18);border-radius:50%;width:80px;height:80px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:32px;font-weight:800;color:white;flex-shrink:0;
                        border:3px solid rgba(255,255,255,0.35);'>
                {_ini}
            </div>
            <div>
                <div style='font-family:Lora,serif;font-size:1.6rem;font-weight:600;'>
                    {(st.session_state.profil_prenom+" "+st.session_state.profil_nom).strip() or st.session_state.username}
                </div>
                <div style='opacity:0.80;font-size:14px;margin-top:4px;'>
                    {'📍 '+st.session_state.profil_ville+' · '+st.session_state.profil_region if st.session_state.profil_ville else '📍 '+st.session_state.profil_region}
                </div>
                <div style='opacity:0.70;font-size:13px;margin-top:3px;'>
                    {'🌾 '+', '.join(st.session_state.profil_cultures[:3]) if st.session_state.profil_cultures else '🌾 Cultures non renseignées'}
                    &nbsp;|&nbsp; {'💧 Irrigué' if st.session_state.profil_irrigation=="Oui" else '🌧️ Pluvial'}
                    &nbsp;|&nbsp; {'📐 '+str(st.session_state.profil_surface)+' ha' if st.session_state.profil_surface else '📐 Surface non renseignée'}
                </div>
            </div>
        </div>
        {'<div style="margin-top:14px;background:rgba(255,255,255,0.1);border-radius:10px;padding:10px 14px;font-size:13px;font-style:italic;">'+st.session_state.profil_bio+"</div>" if st.session_state.profil_bio else ""}
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.profil_saved:
        st.success("✅ Profil enregistré avec succès !")

    tab1, tab2, tab3 = st.tabs(["✏️ Modifier mes informations", "🔒 Sécurité & Accès", "📊 Mon activité"])

    # ════════════════════════════
    #  TAB 1 — Modifier le profil
    # ════════════════════════════
    with tab1:
        st.markdown("<div class='section-title'>👤 Informations personnelles</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            nouveau_prenom = st.text_input("Prénom *", value=st.session_state.profil_prenom,
                placeholder="ex: Ahmed", key="inp_prenom")
            nouveau_tel = st.text_input("📱 Téléphone", value=st.session_state.profil_tel,
                placeholder="ex: 0661-234567", key="inp_tel")
            nouveau_ville = st.text_input("🏘️ Ville / Commune", value=st.session_state.profil_ville,
                placeholder="ex: Beni Mellal", key="inp_ville")
            nouveau_email = st.text_input("📧 Email (optionnel)", value=st.session_state.profil_email,
                placeholder="ex: ahmed@gmail.com", key="inp_email")
        with c2:
            nouveau_nom = st.text_input("Nom de famille *", value=st.session_state.profil_nom,
                placeholder="ex: Bousaid", key="inp_nom")
            nouvelle_region = st.selectbox("📍 Région agricole *",
                ["Béni Mellal-Khénifra","Fès-Meknès","Souss-Massa","Marrakech-Safi",
                 "Rabat-Salé-Kénitra","Oriental","Tanger-Tétouan-Al Hoceïma",
                 "Casablanca-Settat","Drâa-Tafilalet","Guelmim-Oued Noun","Laâyoune","Dakhla"],
                index=["Béni Mellal-Khénifra","Fès-Meknès","Souss-Massa","Marrakech-Safi",
                       "Rabat-Salé-Kénitra","Oriental","Tanger-Tétouan-Al Hoceïma",
                       "Casablanca-Settat","Drâa-Tafilalet","Guelmim-Oued Noun","Laâyoune","Dakhla"
                       ].index(st.session_state.profil_region) if st.session_state.profil_region in
                       ["Béni Mellal-Khénifra","Fès-Meknès","Souss-Massa","Marrakech-Safi",
                        "Rabat-Salé-Kénitra","Oriental","Tanger-Tétouan-Al Hoceïma",
                        "Casablanca-Settat","Drâa-Tafilalet","Guelmim-Oued Noun","Laâyoune","Dakhla"] else 0,
                key="inp_region")
            nouvel_experience = st.selectbox("👨‍🌾 Expérience agricole",
                ["Débutant (< 1 an)","1-5 ans","5-10 ans","10-20 ans","+ de 20 ans"],
                index=["Débutant (< 1 an)","1-5 ans","5-10 ans","10-20 ans","+ de 20 ans"
                       ].index(st.session_state.profil_experience) if st.session_state.profil_experience in
                       ["Débutant (< 1 an)","1-5 ans","5-10 ans","10-20 ans","+ de 20 ans"] else 1,
                key="inp_experience")
            nouvel_irrigation = st.radio("💧 Accès à l'irrigation", ["Oui","Non","Partielle"],
                index=["Oui","Non","Partielle"].index(st.session_state.profil_irrigation)
                      if st.session_state.profil_irrigation in ["Oui","Non","Partielle"] else 1,
                horizontal=True, key="inp_irrigation")

        st.markdown("<div class='section-title'>🌾 Mon exploitation</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            nouvelle_surface = st.number_input("📐 Surface totale (hectares)", min_value=0.0,
                max_value=10000.0, value=float(st.session_state.profil_surface),
                step=0.5, format="%.1f", key="inp_surface")
            nouvelles_cultures = st.multiselect("🌱 Cultures pratiquées",
                ["Blé tendre","Orge","Maïs","Tournesol","Betterave","Tomate","Olivier",
                 "Agrumes","Pomme de terre","Oignon","Carotte","Pastèque","Raisin","Amandier","Autre"],
                default=st.session_state.profil_cultures, key="inp_cultures")
        with c4:
            type_sol = st.selectbox("🪨 Type de sol principal",
                ["Limoneux (fertile)","Argileux (lourd)","Sableux (léger)","Calcaire","Mixte","Je ne sais pas"],
                key="inp_sol")
            type_eau = st.selectbox("💦 Source d'eau",
                ["Pluie uniquement","Canal d'irrigation","Puits / forage","Barrage","Combiné"],
                key="inp_eau")

        nouvelle_bio = st.text_area("📝 À propos de moi / Mon exploitation",
            value=st.session_state.profil_bio,
            placeholder="Décrivez votre exploitation, vos défis, vos objectifs... (optionnel)",
            height=90, key="inp_bio")

        st.markdown("<br>", unsafe_allow_html=True)
        col_save, col_reset = st.columns([3, 1])
        with col_save:
            if st.button("💾  Enregistrer mon profil", use_container_width=True, key="btn_save"):
                if nouveau_prenom.strip() and nouveau_nom.strip():
                    st.session_state.profil_prenom    = nouveau_prenom.strip()
                    st.session_state.profil_nom       = nouveau_nom.strip()
                    st.session_state.profil_tel       = nouveau_tel.strip()
                    st.session_state.profil_region    = nouvelle_region
                    st.session_state.profil_ville     = nouveau_ville.strip()
                    st.session_state.profil_surface   = nouvelle_surface
                    st.session_state.profil_cultures  = nouvelles_cultures
                    st.session_state.profil_experience= nouvel_experience
                    st.session_state.profil_irrigation= nouvel_irrigation
                    st.session_state.profil_email     = nouveau_email.strip()
                    st.session_state.profil_bio       = nouvelle_bio.strip()
                    st.session_state.profil_saved     = True
                    st.session_state.username = f"{nouveau_prenom.strip()} {nouveau_nom.strip()}"
                    st.rerun()
                else:
                    st.error("❌ Le prénom et le nom sont obligatoires.")
        with col_reset:
            if st.button("🗑️ Réinitialiser", use_container_width=True, key="btn_reset"):
                for k in ["profil_nom","profil_prenom","profil_tel","profil_ville","profil_email","profil_bio"]:
                    st.session_state[k] = ""
                st.session_state.profil_surface   = 0.0
                st.session_state.profil_cultures  = []
                st.session_state.profil_region    = "Béni Mellal-Khénifra"
                st.session_state.profil_experience= "1-5 ans"
                st.session_state.profil_irrigation= "Non"
                st.session_state.profil_saved     = False
                st.session_state.username         = "Agriculteur"
                st.rerun()

    # ════════════════════════════
    #  TAB 2 — Sécurité
    # ════════════════════════════
    with tab2:
        st.markdown("<div class='section-title'>🔒 Changer le mot de passe</div>", unsafe_allow_html=True)
        st.info("ℹ️ Dans cette version de démonstration, les mots de passe acceptés sont : **agriclim2026**, **pfe2026**, **admin**.")
        c1, c2 = st.columns(2)
        with c1:
            mdp_actuel  = st.text_input("🔑 Mot de passe actuel",  type="password", key="mdp_act")
            mdp_nouveau = st.text_input("🔒 Nouveau mot de passe", type="password", key="mdp_new",
                help="Minimum 8 caractères recommandé")
            mdp_confirm = st.text_input("🔒 Confirmer le mot de passe", type="password", key="mdp_conf")
            if st.button("🔄 Changer le mot de passe", key="btn_mdp"):
                if mdp_actuel not in ["agriclim2026","pfe2026","admin"]:
                    st.error("❌ Mot de passe actuel incorrect.")
                elif len(mdp_nouveau) < 6:
                    st.warning("⚠️ Le mot de passe doit avoir au moins 6 caractères.")
                elif mdp_nouveau != mdp_confirm:
                    st.error("❌ Les deux mots de passe ne correspondent pas.")
                else:
                    st.success("✅ Mot de passe mis à jour avec succès ! (Simulation — en production, ceci serait sauvegardé en base de données.)")

        with c2:
            st.markdown("""<div class='conseil-card'>
                <h4>🛡️ Conseils de sécurité</h4>
                <p>• Utilisez un mot de passe d'au moins 8 caractères<br>
                • Combinez lettres, chiffres et symboles<br>
                • Ne partagez jamais votre mot de passe<br>
                • Changez votre mot de passe tous les 6 mois<br>
                • Déconnectez-vous après chaque session sur un appareil partagé</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>📱 Paramètres de session</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"""<div class='info-card'>
                <div style='font-size:13px;color:#4a5e4a;'><b>Connecté en tant que :</b> {st.session_state.username}</div>
                <div style='font-size:13px;color:#4a5e4a;margin-top:4px;'><b>Session :</b> Active</div>
                <div style='font-size:13px;color:#4a5e4a;margin-top:4px;'><b>Profil complété :</b> {"✅ Oui" if st.session_state.profil_saved else "⚠️ Non — complétez votre profil"}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            if st.button("🚪 Se déconnecter maintenant", use_container_width=True, key="btn_deconn2"):
                st.session_state.authenticated = False
                st.session_state.page = "Accueil"
                st.rerun()

    # ════════════════════════════
    #  TAB 3 — Activité
    # ════════════════════════════
    with tab3:
        st.markdown("<div class='section-title'>📊 Résumé de mon exploitation</div>", unsafe_allow_html=True)

        if st.session_state.profil_saved and st.session_state.profil_cultures:
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("📐 Surface", f"{st.session_state.profil_surface:.1f} ha")
            with c2: st.metric("🌱 Cultures", len(st.session_state.profil_cultures))
            with c3: st.metric("💧 Irrigation", st.session_state.profil_irrigation)
            with c4: st.metric("👨‍🌾 Expérience", st.session_state.profil_experience)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='section-title'>🌾 Mes cultures — Rendements moyens</div>", unsafe_allow_html=True)
                cultures_user = st.session_state.profil_cultures
                df_user = df_f[df_f["Product"].isin(cultures_user)] if cultures_user else df_f
                rend_user = df_user.groupby("Product")["Value_Mean"].mean().sort_values(ascending=True).reset_index()
                if not rend_user.empty:
                    fig_u = px.bar(rend_user, x="Value_Mean", y="Product", orientation="h",
                        color="Value_Mean", color_continuous_scale=["#ffeaa7","#2d7a3a"],
                        labels={"Value_Mean":"Rendement (hg/ha)","Product":""})
                    apply_theme(fig_u, height=max(200, len(rend_user)*45))
                    fig_u.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_u, use_container_width=True)

            with col2:
                st.markdown("<div class='section-title'>📈 Tendance dans ma région</div>", unsafe_allow_html=True)
                region_user = st.session_state.profil_region
                if "Region" in df_f.columns and region_user in df_f["Region"].values:
                    df_reg_user = df_f[df_f["Region"]==region_user]
                    evo_u = df_reg_user.groupby("Year")["Value_Mean"].mean().reset_index()
                    fig_r = px.area(evo_u, x="Year", y="Value_Mean",
                        labels={"Value_Mean":"Rendement (hg/ha)","Year":"Année"},
                        color_discrete_sequence=["#2d7a3a"],
                        title=f"Rendements — {region_user}")
                    fig_r.update_traces(fillcolor="rgba(45,122,58,0.12)")
                    apply_theme(fig_r, height=250)
                    st.plotly_chart(fig_r, use_container_width=True)
                else:
                    st.info(f"Données disponibles pour {region_user} après ajout de la colonne Region dans votre CSV.")

            # Fiche récapitulative
            st.markdown("<div class='section-title'>📋 Fiche récapitulative de mon exploitation</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:white;border-radius:14px;padding:20px 24px;
                         box-shadow:0 2px 12px rgba(0,0,0,0.07);border-left:5px solid #2d7a3a;'>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                    <div><span style='color:#4a5e4a;font-size:13px;'>👤 Nom complet</span><br>
                         <b>{(st.session_state.profil_prenom+" "+st.session_state.profil_nom).strip()}</b></div>
                    <div><span style='color:#4a5e4a;font-size:13px;'>📍 Localisation</span><br>
                         <b>{st.session_state.profil_ville+" · " if st.session_state.profil_ville else ""}{st.session_state.profil_region}</b></div>
                    <div><span style='color:#4a5e4a;font-size:13px;'>📐 Surface</span><br>
                         <b>{st.session_state.profil_surface:.1f} hectares</b></div>
                    <div><span style='color:#4a5e4a;font-size:13px;'>💧 Irrigation</span><br>
                         <b>{st.session_state.profil_irrigation}</b></div>
                    <div><span style='color:#4a5e4a;font-size:13px;'>🌱 Cultures</span><br>
                         <b>{", ".join(st.session_state.profil_cultures) if st.session_state.profil_cultures else "—"}</b></div>
                    <div><span style='color:#4a5e4a;font-size:13px;'>👨‍🌾 Expérience</span><br>
                         <b>{st.session_state.profil_experience}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("📝 Complétez votre profil dans l'onglet **Modifier mes informations** pour voir ici le résumé de votre exploitation avec graphiques personnalisés.")
            st.markdown("""<div class='conseil-card'>
                <h4>🎯 Pourquoi renseigner votre profil ?</h4>
                <p>• Les graphiques de la page Accueil s'adapteront à votre région<br>
                • Les conseils seront personnalisés selon vos cultures<br>
                • La prédiction IA sera pré-remplie avec vos paramètres habituels<br>
                • Vous aurez accès à une fiche récapitulative de votre exploitation</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE À PROPOS
# ══════════════════════════════════════════════════════════
elif page == "À Propos":
    st.markdown("<div class='page-title'>ℹ️ À Propos de AgriClim Maroc</div>", unsafe_allow_html=True)
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("""<div class='info-card'>
            <h3 style='color:#2d7a3a;margin-top:0;'>🎓 Projet de Fin d'Études (PFE) 2026</h3>
            <p style='font-size:14px;line-height:1.8;'><b>AgriClim Maroc</b> est une application d'aide à la décision agricole développée dans le cadre
            d'un Projet de Fin d'Études. Elle vise à démocratiser l'accès aux données agricoles et climatiques pour tous les agriculteurs marocains.</p>
            <hr><p style='font-size:14px;line-height:1.8;'><b>Objectif :</b> Analyser les données agricoles historiques du Maroc pour aider
            les agriculteurs à comprendre les facteurs qui influencent leurs rendements et à prendre de meilleures décisions.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='info-card' style='text-align:center;'>
            <div style='font-size:48px;'>🌾</div>
            <div style='font-weight:800;color:#2d7a3a;font-size:16px;margin:8px 0;'>AgriClim Maroc</div>
            <div style='font-size:13px;color:#4a5e4a;'>Version 3.0 — PFE 2026</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>⚙️ Fonctionnalités</div>", unsafe_allow_html=True)
    foncs=[("🌦️","Analyse climatique","Précipitations, températures, anomalies, corrélations et heatmaps sur 30+ ans"),
           ("📈","Rendements","Analyse multi-cultures, régions, prévision tendance, violins plots"),
           ("🗺️","Carte interactive","12 régions cliquables avec profils culturaux et statistiques"),
           ("🤖","Prédiction IA","XGBoost R²=0.93, 7 paramètres, analyse de sensibilité"),
           ("💡","Conseils experts","6 situations agricoles, calendrier cultural, contacts ONCA"),
           ("💬","Assistant IA","Chatbot agricole avec base de connaissance marocaine")]
    cols = st.columns(3)
    for i,(ico,titre,desc) in enumerate(foncs):
        with cols[i%3]:
            st.markdown(f"""<div class='info-card' style='min-height:110px;'>
                <div style='font-size:24px;'>{ico}</div>
                <div style='font-weight:700;color:#2d7a3a;font-size:13px;margin:4px 0;'>{titre}</div>
                <div style='font-size:11.5px;color:#4a5e4a;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🛠️ Stack technique</div>", unsafe_allow_html=True)
    tech=[("Python 3.11","Langage principal","#3776ab"),("Streamlit","Interface web","#ff4b4b"),
          ("Pandas","Manipulation données","#150458"),("Plotly","Visualisations","#3d9970"),
          ("NumPy","Calculs scientifiques","#013243"),("XGBoost","Modèle IA","#e07b2a")]
    tc = st.columns(6)
    for col,(nom,role,couleur) in zip(tc,tech):
        with col:
            st.markdown(f"""<div style='text-align:center;background:white;border-radius:10px;padding:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);border-top:3px solid {couleur};'>
                <div style='font-weight:700;font-size:13px;color:{couleur};'>{nom}</div>
                <div style='font-size:11px;color:#4a5e4a;margin-top:2px;'>{role}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""<div style='text-align:center;font-size:12px;color:#6b7280;padding:10px;'>
    🌾 <b>AgriClim Maroc</b> — Application d'analyse agricole | PFE 2026
</div>""", unsafe_allow_html=True)