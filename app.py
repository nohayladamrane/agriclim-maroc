import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import json
import hashlib
import base64
from datetime import datetime, date
import requests
import joblib
import time

st.set_page_config(page_title="AgriClim Maroc", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════
#  TRADUCTIONS (FR / AR)
# ═══════════════════════════════════════════════
TRANSLATIONS = {
    "fr": {
        "app_name": "AgriClim Maroc",
        "app_subtitle": "Assistant intelligent pour les agriculteurs",
        "login_title": "Connexion",
        "login_username": "👤 Votre nom d'utilisateur",
        "login_password": "🔒 Mot de passe",
        "login_btn": "🚪 Se Connecter",
        "login_error": "❌ Identifiants incorrects.",
        "register_btn": "📝 Créer un compte",
        "register_title": "Créer un compte",
        "reg_username": "Nom d'utilisateur *",
        "reg_password": "Mot de passe *",
        "reg_confirm": "Confirmer le mot de passe *",
        "reg_nom": "Nom de famille *",
        "reg_prenom": "Prénom *",
        "reg_region": "Région *",
        "reg_submit": "✅ Créer mon compte",
        "reg_success": "✅ Compte créé ! Vous pouvez vous connecter.",
        "reg_error_exist": "❌ Ce nom d'utilisateur existe déjà.",
        "reg_error_match": "❌ Les mots de passe ne correspondent pas.",
        "reg_error_fields": "❌ Remplissez tous les champs obligatoires.",
        "logout": "Déconnexion",
        "nav_home": "Accueil",
        "nav_meteo": "Météo & Climat",
        "nav_rend": "Rendements",
        "nav_carte": "Carte du Maroc",
        "nav_pred": "Prédiction IA",
        "nav_conseils": "Conseils Pratiques",
        "nav_chat": "Assistant IA 🤖",
        "nav_profil": "Mon Profil",
        "nav_history": "Historique",
        "nav_about": "Guide & À Propos",
        "welcome": "Bienvenue",
        "period": "Période",
        "years_analyzed": "années analysées",
        "avg_yield": "🌾 Rendement moyen",
        "avg_rain": "🌧️ Pluie moyenne",
        "avg_temp": "🌡️ Température moy.",
        "crops_followed": "🌿 Cultures suivies",
        "regions": "régions",
        "yield_evolution": "📈 Évolution des rendements",
        "top_crops": "🌿 Top cultures par rendement",
        "rain_vs_yield": "🌧️ Pluie vs Rendement",
        "yield_by_region": "📊 Rendement par région",
        "lang_label": "🌐 Langue",
        "photo_upload": "📸 Photo de profil",
        "save_profile": "💾 Enregistrer",
        "history_title": "📋 Historique des prédictions",
        "history_empty": "Aucune prédiction enregistrée.",
        "pred_result": "Rendement estimé",
        "pred_save": "💾 Sauvegarder cette prédiction",
        "pred_saved": "✅ Prédiction sauvegardée !",
        "current_weather": "☁️ Météo actuelle",
        "filter_label": "Filtres",
        "period_label": "Période",
        "search_crop": "🔍 Rechercher une culture",
        "all_crops": "Toutes les cultures",
        "crop_label": "Culture",
        "region_label": "Région",
        "back_login": "← Retour à la connexion",
    },
    "ar": {
        "app_name": "أجريكليم المغرب",
        "app_subtitle": "مساعد ذكي للمزارعين",
        "login_title": "تسجيل الدخول",
        "login_username": "👤 اسم المستخدم",
        "login_password": "🔒 كلمة المرور",
        "login_btn": "🚪 تسجيل الدخول",
        "login_error": "❌ بيانات الدخول غير صحيحة.",
        "register_btn": "📝 إنشاء حساب",
        "register_title": "إنشاء حساب جديد",
        "reg_username": "اسم المستخدم *",
        "reg_password": "كلمة المرور *",
        "reg_confirm": "تأكيد كلمة المرور *",
        "reg_nom": "اللقب *",
        "reg_prenom": "الاسم *",
        "reg_region": "الجهة *",
        "reg_submit": "✅ إنشاء الحساب",
        "reg_success": "✅ تم إنشاء الحساب! يمكنك تسجيل الدخول الآن.",
        "reg_error_exist": "❌ اسم المستخدم موجود مسبقاً.",
        "reg_error_match": "❌ كلمتا المرور غير متطابقتين.",
        "reg_error_fields": "❌ يرجى ملء جميع الحقول الإلزامية.",
        "logout": "تسجيل الخروج",
        "nav_home": "الرئيسية",
        "nav_meteo": "الطقس والمناخ",
        "nav_rend": "المردودية",
        "nav_carte": "خريطة المغرب",
        "nav_pred": "التنبؤ بالذكاء الاصطناعي",
        "nav_conseils": "نصائح عملية",
        "nav_chat": "المساعد الذكي 🤖",
        "nav_profil": "ملفي",
        "nav_history": "السجل",
        "nav_about": "الدليل وحول التطبيق",
        "welcome": "مرحباً",
        "period": "الفترة",
        "years_analyzed": "سنوات محللة",
        "avg_yield": "🌾 متوسط المردودية",
        "avg_rain": "🌧️ متوسط الأمطار",
        "avg_temp": "🌡️ متوسط الحرارة",
        "crops_followed": "🌿 المحاصيل المتابعة",
        "regions": "جهات",
        "yield_evolution": "📈 تطور المردودية",
        "top_crops": "🌿 أفضل المحاصيل",
        "rain_vs_yield": "🌧️ الأمطار مقابل المردودية",
        "yield_by_region": "📊 المردودية حسب الجهة",
        "lang_label": "🌐 اللغة",
        "photo_upload": "📸 صورة الملف الشخصي",
        "save_profile": "💾 حفظ",
        "history_title": "📋 سجل التنبؤات",
        "history_empty": "لا توجد تنبؤات محفوظة.",
        "pred_result": "المردودية المقدرة",
        "pred_save": "💾 حفظ هذا التنبؤ",
        "pred_saved": "✅ تم حفظ التنبؤ!",
        "current_weather": "☁️ الطقس الحالي",
        "filter_label": "المرشحات",
        "period_label": "الفترة",
        "search_crop": "🔍 البحث عن محصول",
        "all_crops": "جميع المحاصيل",
        "crop_label": "المحصول",
        "region_label": "الجهة",
        "back_login": "← العودة إلى تسجيل الدخول",
    }
}

def T(key):
    lang = st.session_state.get("lang", "fr")
    return TRANSLATIONS.get(lang, TRANSLATIONS["fr"]).get(key, key)

# ═══════════════════════════════════════════════
#  USER DATABASE (JSON-based)
# ═══════════════════════════════════════════════
USERS_FILE = "agriclim_users.json"
HISTORY_FILE = "agriclim_history.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    default = {
        "agriclim": {
            "password": hash_password("agriclim2026"),
            "nom": "Admin", "prenom": "AgriClim",
            "region": "Béni Mellal-Khénifra", "ville": "",
            "surface": 0.0, "cultures": [], "experience": "1-5 ans",
            "irrigation": "Non", "email": "", "bio": "",
            "tel": "", "photo": None, "created": str(date.today())
        },
        "admin": {
            "password": hash_password("admin"),
            "nom": "Administrateur", "prenom": "",
            "region": "Rabat-Salé-Kénitra", "ville": "",
            "surface": 0.0, "cultures": [], "experience": "10-20 ans",
            "irrigation": "Oui", "email": "admin@agriclim.ma", "bio": "",
            "tel": "", "photo": None, "created": str(date.today())
        }
    }
    save_users(default)
    return default

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ═══════════════════════════════════════════════
#  CSS GLOBAL
# ═══════════════════════════════════════════════
def apply_css(lang="fr"):
    rtl = "direction: rtl; text-align: right;" if lang == "ar" else ""
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Lora:wght@600&family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

html, body, .stApp {{
    background: #f0f2f0 !important;
    font-family: {'Noto Naskh Arabic' if lang == 'ar' else 'Nunito'}, sans-serif !important;
    color: #1a1a1a !important;
    {rtl}
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1a472a 0%, #0d2818 100%) !important;
    border-right: none !important;
}}

.main .block-container {{
    background: #ffffff !important;
    padding: 1.5rem 2rem 2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}}

.page-title {{
    font-family: {'Noto Naskh Arabic' if lang == 'ar' else 'Lora'}, serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: #1a472a !important;
    border-bottom: 2px solid #e0e8e0;
    padding-bottom: 8px;
    margin-bottom: 1.2rem;
    {rtl}
}}

.section-title {{
    font-size: 1rem;
    font-weight: 600;
    color: #1a472a !important;
    margin: 1.2rem 0 0.75rem;
}}

[data-testid="stMetric"] {{
    background: #f8faf8;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #e0e8e0;
}}

[data-testid="stMetricValue"] > div {{ font-size: 24px !important; color: #1a472a !important; font-weight: 700 !important; }}
[data-testid="stMetricLabel"] > div {{ font-size: 12px !important; color: #5a6e5a !important; font-weight: 500 !important; }}

.stButton > button {{
    background: #1a472a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
}}
.stButton > button:hover {{ background: #2d5a3a !important; }}

.info-card, .conseil-card, .history-item {{
    background: #f8faf8;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
    border: 1px solid #e0e8e0;
    color: #1a1a1a !important;
}}
.conseil-card {{ border-left: 3px solid #6ab04c; }}
.conseil-card h4 {{ color: #1a472a !important; margin: 0 0 6px; font-size: 14px; }}

.resultat-ia {{
    background: linear-gradient(135deg, #1a472a, #2d5a3a);
    color: white;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin: 14px 0;
}}

.chat-user {{
    background: #e8f0e8;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    margin: 8px 0 8px auto;
    max-width: 80%;
    color: #1a1a1a !important;
}}
.chat-bot {{
    background: #f0f4f0;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px;
    margin: 8px auto 8px 0;
    max-width: 85%;
    color: #1a1a1a !important;
    border-left: 3px solid #6ab04c;
}}

.weather-card {{
    background: linear-gradient(135deg, #1565c0, #1976d2);
    color: white;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
}}

hr {{ border-color: #e0e8e0 !important; margin: 16px 0 !important; }}

.avatar-circle {{
    width: 80px; height: 80px; border-radius: 50%;
    background: linear-gradient(135deg, #6ab04c, #1a472a);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; font-weight: 700; color: white;
}}

[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] div, [data-testid="stSidebar"] label {{ color: rgba(255,255,255,0.88) !important; }}

[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    color: rgba(255,255,255,0.68) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{ background: rgba(255,255,255,0.09) !important; color: white !important; }}

[data-testid="stSidebar"] .stSlider label {{ color: white !important; font-weight: 500 !important; }}
[data-testid="stSidebar"] .stSlider div[data-testid="stMarkdownContainer"] p {{ color: white !important; font-size: 12px !important; }}
[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{ background-color: #6ab04c !important; }}
[data-testid="stSidebar"] div[style*="text-transform:uppercase"] {{ color: rgba(255,255,255,0.6) !important; }}

.stSlider label {{ color: #1a1a1a !important; }}
.stSlider .stMarkdown p {{ color: #1a1a1a !important; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 0.5rem; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px 8px 0 0; padding: 0.5rem 1rem; font-weight: 500; color: #1a1a1a !important; }}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════
defaults = {
    "authenticated": False, "username": "", "lang": "fr",
    "chat_history": [], "page": "Accueil",
    "show_register": False,
    "users": None, "pred_history": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.users is None:
    st.session_state.users = load_users()
if st.session_state.pred_history is None:
    st.session_state.pred_history = load_history()

lang = st.session_state.lang
apply_css(lang)

# ═══════════════════════════════════════════════
#  FONCTIONS DE CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════
@st.cache_data
def load_data():
    for file in ["dataset_powerbi_regional.csv", "dataset_analysis.csv", "dataset_ml.csv", "dataset_powerbi.csv"]:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, encoding='utf-8')
                df = standardize_columns(df)
                if 'Region' in df.columns and df['Region'].nunique() > 1:
                    return df
            except:
                try:
                    df = pd.read_csv(file, encoding='latin-1')
                    df = standardize_columns(df)
                    if 'Region' in df.columns and df['Region'].nunique() > 1:
                        return df
                except:
                    continue
    return generate_demo_data()

def standardize_columns(df):
    col_mapping = {
        'Year': ['Year', 'year', 'Année', 'annee'],
        'Product': ['Product', 'product', 'Item', 'item', 'Culture', 'culture'],
        'Value_Mean': ['Value_Mean', 'value_mean', 'Rendement', 'Yield', 'yield'],
        'Precip_Total_mm': ['Precip_Total_mm', 'precip_total_mm', 'Precipitation', 'Pluie', 'Rain'],
        'Temp_Mean_C': ['Temp_Mean_C', 'temp_mean_c', 'Temperature', 'Temp', 'temp'],
        'Region': ['Region', 'region', 'Région', 'Area', 'area'],
        'Humidity_Mean': ['Humidity_Mean', 'humidity_mean', 'Humidity', 'humidity'],
        'Solar_Rad': ['Solar_Rad', 'solar_rad', 'Solar', 'Radiation']
    }
    for target, aliases in col_mapping.items():
        for alias in aliases:
            if alias in df.columns and target not in df.columns:
                df = df.rename(columns={alias: target})
                break
    if 'Year' not in df.columns:
        df['Year'] = 2020
    if 'Product' not in df.columns:
        df['Product'] = 'Culture'
    if 'Value_Mean' not in df.columns:
        df['Value_Mean'] = 2000
    if 'Precip_Total_mm' not in df.columns:
        df['Precip_Total_mm'] = 400
    if 'Temp_Mean_C' not in df.columns:
        df['Temp_Mean_C'] = 20
    if 'Region' not in df.columns:
        df['Region'] = 'Maroc'
    if 'Humidity_Mean' not in df.columns:
        df['Humidity_Mean'] = 55
    if 'Solar_Rad' not in df.columns:
        df['Solar_Rad'] = 18
    return df

def generate_demo_data():
    np.random.seed(42)
    years = list(range(1990, 2025))
    products = ["Blé tendre", "Orge", "Maïs", "Tomate", "Olivier", "Agrumes", "Pomme de terre", "Dattier"]
    regions = [
        "Béni Mellal-Khénifra", "Fès-Meknès", "Souss-Massa", "Marrakech-Safi", 
        "Rabat-Salé-Kénitra", "Oriental", "Tanger-Tétouan-Al Hoceïma", 
        "Casablanca-Settat", "Drâa-Tafilalet"
    ]
    rows = []
    for y in years:
        base_precip = 350 + 30 * np.sin((y - 1990) / 5) + np.random.normal(0, 50)
        base_temp = 19 + (y - 1990) * 0.02 + np.random.normal(0, 1)
        for p in products:
            for r in regions:
                if r == "Souss-Massa" and p in ["Tomate", "Agrumes"]:
                    rend = 5000 + base_precip * 5 - (base_temp - 20) * 30 + np.random.normal(0, 200)
                elif r == "Drâa-Tafilalet" and p == "Dattier":
                    rend = 4000 + base_precip * 3 - (base_temp - 20) * 20 + np.random.normal(0, 150)
                elif r == "Béni Mellal-Khénifra" and p in ["Blé tendre", "Orge"]:
                    rend = 2500 + base_precip * 4 - (base_temp - 20) * 40 + np.random.normal(0, 120)
                else:
                    rend = 1800 + base_precip * 4 - (base_temp - 20) * 50 + np.random.normal(0, 150)
                rows.append({
                    "Year": y, "Product": p, "Region": r,
                    "Value_Mean": max(300, round(rend)),
                    "Precip_Total_mm": round(max(50, base_precip + np.random.normal(0, 35)), 1),
                    "Temp_Mean_C": round(base_temp + np.random.normal(0, 0.7), 1),
                    "Humidity_Mean": round(np.random.uniform(40, 70), 1),
                    "Solar_Rad": round(np.random.uniform(12, 26), 1)
                })
    return pd.DataFrame(rows)

df = load_data()

# ═══════════════════════════════════════════════
#  WEATHER API AVEC DONNÉES DE DÉMONSTRATION
# ═══════════════════════════════════════════════
REGION_COORDS = {
    "Béni Mellal-Khénifra": (32.34, -6.36),
    "Fès-Meknès": (34.03, -4.99),
    "Souss-Massa": (30.42, -9.60),
    "Marrakech-Safi": (31.63, -8.01),
    "Rabat-Salé-Kénitra": (34.01, -6.83),
    "Oriental": (34.68, -1.90),
    "Tanger-Tétouan-Al Hoceïma": (35.77, -5.80),
    "Casablanca-Settat": (33.59, -7.62),
    "Drâa-Tafilalet": (31.93, -4.43),
}

DEMO_WEATHER = {
    "Béni Mellal-Khénifra": {"temp": 22, "humidity": 55, "precip": 15, "wind": 12, "icon": "⛅", "description": "Partiellement nuageux"},
    "Fès-Meknès": {"temp": 20, "humidity": 60, "precip": 20, "wind": 10, "icon": "🌧️", "description": "Pluies éparses"},
    "Souss-Massa": {"temp": 24, "humidity": 50, "precip": 5, "wind": 15, "icon": "☀️", "description": "Ensoleillé"},
    "Marrakech-Safi": {"temp": 26, "humidity": 45, "precip": 8, "wind": 14, "icon": "☀️", "description": "Beau temps"},
    "Rabat-Salé-Kénitra": {"temp": 19, "humidity": 65, "precip": 25, "wind": 18, "icon": "⛅", "description": "Nuageux"},
    "Oriental": {"temp": 21, "humidity": 52, "precip": 12, "wind": 16, "icon": "⛅", "description": "Ciel variable"},
    "Tanger-Tétouan-Al Hoceïma": {"temp": 18, "humidity": 70, "precip": 30, "wind": 20, "icon": "🌧️", "description": "Pluie"},
    "Casablanca-Settat": {"temp": 20, "humidity": 62, "precip": 18, "wind": 17, "icon": "⛅", "description": "Partiellement nuageux"},
    "Drâa-Tafilalet": {"temp": 28, "humidity": 40, "precip": 2, "wind": 13, "icon": "☀️", "description": "Sec et chaud"},
}

def get_weather_demo(region_name):
    return DEMO_WEATHER.get(region_name, DEMO_WEATHER["Béni Mellal-Khénifra"])

# ═══════════════════════════════════════════════
#  CHARGEMENT DU MODÈLE XGBOOST
# ═══════════════════════════════════════════════
xgb_model = None
model_loaded = False
try:
    if os.path.exists("models/xgboost_model.pkl"):
        xgb_model = joblib.load("models/xgboost_model.pkl")
        model_loaded = True
except:
    pass

# ═══════════════════════════════════════════════
#  LANDING PAGE
# ═══════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    col_lang = st.columns([4, 1])
    with col_lang[1]:
        lang_choice = st.selectbox("", ["Français 🇫🇷", "العربية 🇲🇦"], key="landing_lang",
                                   index=0 if st.session_state.lang == "fr" else 1, label_visibility="collapsed")
        new_lang = "fr" if "Français" in lang_choice else "ar"
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()

    if not st.session_state.show_register:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 30px;'>
            <div style='font-size:80px; margin-bottom:12px;'>🌾</div>
            <div style='font-family:Lora,serif; font-size:2.5rem; color:#1a472a; font-weight:600; margin-bottom:8px;'>AgriClim Maroc</div>
            <div style='color:#5a6e5a; font-size:14px;'>🌍 Plateforme intelligente d'analyse agricole et climatique</div>
        </div>""", unsafe_allow_html=True)

        cols = st.columns(4)
        features = [("🌦️", "Météo Temps Réel", "Données météo par région"), ("🤖", "Prédiction IA", "XGBoost R²=0.93"), ("📈", "Rendements", "Analyse multi-cultures"), ("💬", "Assistant IA", "Chat agricole intelligent")]
        for col, (ico, titre, desc) in zip(cols, features):
            with col:
                st.markdown(f"""<div class='info-card' style='text-align:center;'><div style='font-size:28px;'>{ico}</div><div style='font-weight:600; color:#1a472a; font-size:13px; margin:4px 0;'>{titre}</div><div style='font-size:11px; color:#5a6e5a;'>{desc}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            st.markdown(f"""<div style='background:white; border-radius:16px; padding:28px; box-shadow:0 2px 12px rgba(0,0,0,0.05); border-top:3px solid #6ab04c;'><div style='font-family:Lora,serif; font-size:1.3rem; color:#1a472a; text-align:center; margin-bottom:20px;'>🔐 {T('login_title')}</div>""", unsafe_allow_html=True)
            username_in = st.text_input(T("login_username"), placeholder="ex: agriclim", key="login_user_in")
            password_in = st.text_input(T("login_password"), type="password", key="login_pass_in")
            if st.button(T("login_btn"), use_container_width=True, key="btn_login"):
                users = st.session_state.users
                if username_in in users and users[username_in]["password"] == hash_password(password_in):
                    st.session_state.authenticated = True
                    st.session_state.username = username_in
                    st.session_state.page = "Accueil"
                    st.rerun()
                else:
                    st.error(T("login_error"))
            st.markdown("<div style='text-align:center; color:#aaa; font-size:12px; margin:12px 0 8px;'>— ou —</div>", unsafe_allow_html=True)
            if st.button(T("register_btn"), use_container_width=True, key="btn_goto_register"):
                st.session_state.show_register = True
                st.rerun()
            st.markdown("<div style='text-align:center; color:#aaa; font-size:11px; margin-top:14px;'>Demo: <b>agriclim</b> / <b>agriclim2026</b></div></div>", unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns([1, 1.4, 1])
        with c2:
            st.markdown(f"""<div style='background:white; border-radius:16px; padding:28px; box-shadow:0 2px 12px rgba(0,0,0,0.05); border-top:3px solid #6ab04c;'><div style='font-family:Lora,serif; font-size:1.3rem; color:#1a472a; text-align:center; margin-bottom:20px;'>📝 {T('register_title')}</div>""", unsafe_allow_html=True)
            r_user = st.text_input(T("reg_username"), placeholder="ex: ahmed_bousaid", key="reg_user")
            col_r1, col_r2 = st.columns(2)
            with col_r1: r_prenom = st.text_input(T("reg_prenom"), placeholder="Ahmed", key="reg_prenom")
            with col_r2: r_nom = st.text_input(T("reg_nom"), placeholder="Bousaid", key="reg_nom")
            r_region = st.selectbox(T("reg_region"), ["Béni Mellal-Khénifra", "Fès-Meknès", "Souss-Massa", "Marrakech-Safi", "Rabat-Salé-Kénitra", "Oriental", "Tanger-Tétouan-Al Hoceïma", "Casablanca-Settat", "Drâa-Tafilalet"], key="reg_region_sel")
            r_pw = st.text_input(T("reg_password"), type="password", key="reg_pw")
            r_pw2 = st.text_input(T("reg_confirm"), type="password", key="reg_pw2")
            if st.button(T("reg_submit"), use_container_width=True, key="btn_register"):
                users = st.session_state.users
                if not all([r_user.strip(), r_prenom.strip(), r_nom.strip(), r_pw]):
                    st.error(T("reg_error_fields"))
                elif r_user.strip() in users:
                    st.error(T("reg_error_exist"))
                elif r_pw != r_pw2:
                    st.error(T("reg_error_match"))
                else:
                    users[r_user.strip()] = {"password": hash_password(r_pw), "nom": r_nom.strip(), "prenom": r_prenom.strip(), "region": r_region, "ville": "", "surface": 0.0, "cultures": [], "experience": "1-5 ans", "irrigation": "Non", "email": "", "bio": "", "tel": "", "photo": None, "created": str(date.today())}
                    save_users(users)
                    st.session_state.users = users
                    st.success(T("reg_success"))
            if st.button(T("back_login"), use_container_width=True, key="btn_back"):
                st.session_state.show_register = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#aaa;font-size:11px;margin-top:20px;'>© AgriClim Maroc PFE 2026</div>", unsafe_allow_html=True)
    st.stop()

st.markdown("""
<style>
    [data-testid="stSidebar"] { display: block; }
    [data-testid="collapsedControl"] { display: block; }
</style>
""", unsafe_allow_html=True)

user_data = st.session_state.users.get(st.session_state.username, {})
NAV_ITEMS = [(T("nav_home"), "🏠"), (T("nav_meteo"), "🌦️"), (T("nav_rend"), "📈"), (T("nav_carte"), "🗺️"), (T("nav_pred"), "🤖"), (T("nav_conseils"), "💡"), (T("nav_chat"), "💬"), (T("nav_profil"), "👤"), (T("nav_history"), "📋"), (T("nav_about"), "ℹ️")]

with st.sidebar:
    st.markdown(f"""<div style='padding:20px 16px 14px;'><div style='display:flex; align-items:center; gap:10px;'><div style='background:rgba(255,255,255,.12); border-radius:10px; width:40px; height:40px; display:flex; align-items:center; justify-content:center; font-size:20px;'>🌾</div><div><div style='font-weight:700; color:white; font-size:1rem;'>{T("app_name")}</div><div style='font-size:10px; color:rgba(255,255,255,.40);'>{T("app_subtitle")}</div></div></div></div>""", unsafe_allow_html=True)
    lang_sel = st.selectbox(T("lang_label"), ["Français 🇫🇷", "العربية 🇲🇦"], index=0 if lang == "fr" else 1, key="lang_selector")
    new_lang = "fr" if "Français" in lang_sel else "ar"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    for label, ico in NAV_ITEMS:
        if st.button(f"{ico} {label}", key=f"nav_{label.replace(' ', '_')}", use_container_width=True):
            st.session_state.page = label
            st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""<div style='padding:0 8px 4px; font-size:10px; font-weight:600; letter-spacing:.10em; color:white; text-transform:uppercase;'>{T("filter_label")}</div>""", unsafe_allow_html=True)
    years_range = st.slider(T("period_label"), int(df["Year"].min()), int(df["Year"].max()), (int(df["Year"].min()), int(df["Year"].max())), label_visibility="collapsed")
    _ini = ""
    if user_data.get("prenom"): _ini += user_data["prenom"][0].upper()
    if user_data.get("nom"): _ini += user_data["nom"][0].upper()
    if not _ini: _ini = st.session_state.username[0].upper() if st.session_state.username else "A"
    _display = f"{user_data.get('prenom', '')} {user_data.get('nom', '')}".strip() or st.session_state.username
    st.markdown(f"""<div style='margin:0 8px 8px; background:rgba(255,255,255,.07); border-radius:8px; padding:8px 10px;'><div style='display:flex; align-items:center; gap:8px;'><div style='background:linear-gradient(135deg,#6ab04c,#1a472a); border-radius:50%; width:36px; height:36px; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; color:white;'>{_ini}</div><div><div style='font-size:12px; font-weight:600; color:white;'>{_display}</div><div style='font-size:9px; color:rgba(255,255,255,.40);'>{user_data.get("region", "")[:20]}</div></div></div></div>""", unsafe_allow_html=True)
    if st.button(f"🚪 {T('logout')}", use_container_width=True, key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.page = "Accueil"
        st.rerun()

page = st.session_state.page
df_f = df[df["Year"].between(years_range[0], years_range[1])].copy()
COLORS = ["#1a472a", "#2d5a3a", "#6ab04c", "#d4a056", "#1565c0", "#8e44ad", "#c0392b", "#16a085"]

def apply_theme(fig, height=300, title=None):
    fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(family="Nunito", size=12, color="#1a1a1a"), xaxis=dict(tickfont=dict(color="#1a1a1a"), gridcolor="#e0e8e0"), yaxis=dict(tickfont=dict(color="#1a1a1a"), gridcolor="#e0e8e0"), legend=dict(font=dict(color="#1a1a1a")), margin=dict(l=40, r=40, t=50 if title else 30, b=40), height=height)
    if title: fig.update_layout(title=dict(text=title, font=dict(size=14, color="#1a472a"), x=0))
    return fig

ALL_CROPS = sorted(df["Product"].unique().tolist())
REGIONS_LIST = sorted(df["Region"].unique().tolist())

# ══════════════════════════════════════════════════════════
#  PAGE ACCUEIL
# ══════════════════════════════════════════════════════════
if page == T("nav_home"):
    st.markdown("<div class='page-title'>🏠 Accueil</div>", unsafe_allow_html=True)
    
    user_region = user_data.get("region", "Béni Mellal-Khénifra")
    demo = get_weather_demo(user_region)
    st.markdown(f"""<div class='weather-card'><div style='display:flex; align-items:center; gap:16px;'><div style='font-size:40px;'>{demo['icon']}</div><div><div style='font-weight:600;'>{T("current_weather")} — {user_region}</div><div style='font-size:1.5rem; font-weight:700;'>{demo['temp']}°C</div><div>💧 {demo['humidity']}% · 🌬️ {demo['wind']} km/h · {demo['description']}</div></div></div></div>""", unsafe_allow_html=True)
    
    st.markdown(f"""<div style='background:linear-gradient(135deg,#1a472a,#2d5a3a); border-radius:16px; padding:20px; margin-bottom:20px; color:white;'><div style='font-size:1.5rem; font-weight:600;'>🌱 {T('welcome')}, {user_data.get('prenom', '') or st.session_state.username} !</div><div>{T('period')} {years_range[0]}–{years_range[1]} · {years_range[1]-years_range[0]+1} {T('years_analyzed')}</div></div>""", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric(T("avg_yield"), f"{df_f['Value_Mean'].mean():.0f} hg/ha")
    with c2: st.metric(T("avg_rain"), f"{df_f['Precip_Total_mm'].mean():.0f} mm")
    with c3: st.metric(T("avg_temp"), f"{df_f['Temp_Mean_C'].mean():.1f} °C")
    with c4: st.metric(T("crops_followed"), df_f["Product"].nunique(), f"{df_f['Region'].nunique()} {T('regions')}")
    
    rend_yr = df_f.groupby("Year")["Value_Mean"].mean().reset_index()
    fig = px.area(rend_yr, x="Year", y="Value_Mean", color_discrete_sequence=["#1a472a"])
    apply_theme(fig, height=300, title="📈 Évolution des rendements")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        top_crops = df_f.groupby("Product")["Value_Mean"].mean().sort_values(ascending=True).tail(10).reset_index()
        fig2 = px.bar(top_crops, x="Value_Mean", y="Product", orientation="h", title="🏆 Top 10 cultures", color="Value_Mean", color_continuous_scale=["#b7dfbf", "#1a472a"])
        apply_theme(fig2, height=350)
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        top_regions = df_f.groupby("Region")["Value_Mean"].mean().sort_values(ascending=True).reset_index()
        fig3 = px.bar(top_regions, x="Value_Mean", y="Region", orientation="h", title="🏆 Top régions", color="Value_Mean", color_continuous_scale=["#b7dfbf", "#1a472a"])
        apply_theme(fig3, height=350)
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
    
    corr_data = df_f.groupby("Year")[["Precip_Total_mm", "Value_Mean"]].mean().reset_index()
    fig4 = px.scatter(corr_data, x="Precip_Total_mm", y="Value_Mean", title="🔗 Corrélation pluie-rendement", labels={"Precip_Total_mm": "Pluie (mm)", "Value_Mean": "Rendement (hg/ha)"})
    apply_theme(fig4, height=350)
    st.plotly_chart(fig4, use_container_width=True)
    corr_value = corr_data["Precip_Total_mm"].corr(corr_data["Value_Mean"])
    st.info(f"📊 Coefficient de corrélation : {corr_value:.3f}")
    
    st.markdown("<div class='section-title'>🌡️ Matrice des rendements</div>", unsafe_allow_html=True)
    top_8_crops = df_f.groupby("Product")["Value_Mean"].mean().nlargest(8).index.tolist()
    heat_data = df_f[df_f["Product"].isin(top_8_crops)].groupby(["Region", "Product"])["Value_Mean"].mean().unstack().fillna(0)
    if not heat_data.empty:
        fig5 = px.imshow(heat_data.values, x=list(heat_data.columns), y=list(heat_data.index), labels=dict(x="Culture", y="Région", color="Rendement"), color_continuous_scale="Greens", aspect="auto", title="🎯 Rendement par région et culture")
        apply_theme(fig5, height=400)
        fig5.update_layout(xaxis=dict(tickangle=45))
        st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  PAGE MÉTÉO & CLIMAT
# ══════════════════════════════════════════════════════════
elif page == T("nav_meteo"):
    st.markdown("<div class='page-title'>🌦️ Météo & Climat</div>", unsafe_allow_html=True)
    
    regions_list = list(REGION_COORDS.keys())
    selected_region = st.selectbox("📍 Sélectionnez une région :", regions_list, index=0)
    demo = get_weather_demo(selected_region)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(f"{demo['icon']} Température", f"{demo['temp']}°C")
    with col2: st.metric("💧 Humidité", f"{demo['humidity']}%")
    with col3: st.metric("🌧️ Précipitations", f"{demo['precip']} mm")
    with col4: st.metric("🌬️ Vent", f"{demo['wind']} km/h")
    st.caption(f"📡 Conditions météo pour {selected_region} : {demo['description']}")
    
    st.markdown("<div class='section-title'>📊 Analyses climatiques historiques</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        prec = df_f.groupby("Year")["Precip_Total_mm"].mean().reset_index()
        fig = px.bar(prec, x="Year", y="Precip_Total_mm", title="🌧️ Précipitations annuelles (mm)", color_discrete_sequence=["#1565c0"])
        apply_theme(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        temp = df_f.groupby("Year")["Temp_Mean_C"].mean().reset_index()
        fig = px.line(temp, x="Year", y="Temp_Mean_C", markers=True, title="🌡️ Températures annuelles (°C)", color_discrete_sequence=["#c0392b"])
        apply_theme(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    corr_data = df_f.groupby("Year")[["Precip_Total_mm", "Value_Mean"]].mean().reset_index()
    fig3 = px.scatter(corr_data, x="Precip_Total_mm", y="Value_Mean", title="🔗 Corrélation pluie-rendement")
    apply_theme(fig3, height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  PAGE RENDEMENTS
# ══════════════════════════════════════════════════════════
elif page == T("nav_rend"):
    st.markdown("<div class='page-title'>📈 Suivi des Rendements</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Analyse générale", "🌍 Analyse régionale", "📈 Prévision tendance"])
    
    with tab1:
        search_crop = st.text_input(T("search_crop"), placeholder="ex: blé, tomate...", key="search_rend")
        filtered = [c for c in ALL_CROPS if search_crop.lower() in c.lower()] if search_crop else ALL_CROPS
        culture = st.selectbox("Culture", [T("all_crops")] + filtered)
        df_sel = df_f if culture == T("all_crops") else df_f[df_f["Product"] == culture]
        
        if not df_sel.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Rendement moyen", f"{df_sel['Value_Mean'].mean():.0f} hg/ha")
            c2.metric("🏆 Maximum", f"{df_sel['Value_Mean'].max():.0f} hg/ha")
            c3.metric("📉 Minimum", f"{df_sel['Value_Mean'].min():.0f} hg/ha")
            c4.metric("📐 Variabilité", f"{df_sel['Value_Mean'].std():.0f} hg/ha")
            
            evo = df_sel.groupby("Year")["Value_Mean"].mean().reset_index()
            fig = px.line(evo, x="Year", y="Value_Mean", markers=True, title="📈 Évolution temporelle", color_discrete_sequence=["#1a472a"])
            apply_theme(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.box(df_sel, x="Product", y="Value_Mean", title="📦 Distribution des rendements")
                apply_theme(fig2, height=350)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                rend_hist = df_sel["Value_Mean"].value_counts().sort_index().reset_index()
                rend_hist.columns = ["Rendement", "Fréquence"]
                fig3 = px.bar(rend_hist.head(30), x="Rendement", y="Fréquence", title="📊 Distribution des rendements", color_discrete_sequence=["#1a472a"])
                apply_theme(fig3, height=350)
                st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        reg_sel = st.selectbox("Sélectionnez une région", REGIONS_LIST)
        df_reg = df_f[df_f["Region"] == reg_sel]
        if not df_reg.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("🌾 Rendement moyen", f"{df_reg['Value_Mean'].mean():.0f} hg/ha")
            c2.metric("🌧️ Pluie", f"{df_reg['Precip_Total_mm'].mean():.0f} mm")
            c3.metric("🌡️ Température", f"{df_reg['Temp_Mean_C'].mean():.1f} °C")
            
            top_crops_reg = df_reg.groupby("Product")["Value_Mean"].mean().sort_values(ascending=True).tail(10).reset_index()
            fig = px.bar(top_crops_reg, x="Value_Mean", y="Product", orientation="h", title=f"🏆 Top cultures - {reg_sel}", color="Value_Mean", color_continuous_scale=["#b7dfbf", "#1a472a"])
            apply_theme(fig, height=350)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        culture_prev = st.selectbox("Culture à analyser", ALL_CROPS)
        df_c = df_f[df_f["Product"] == culture_prev].groupby("Year")["Value_Mean"].mean().reset_index()
        if len(df_c) >= 3:
            x = df_c["Year"].values
            y = df_c["Value_Mean"].values
            slope, intercept = np.polyfit(x, y, 1)
            fig = px.scatter(df_c, x="Year", y="Value_Mean", title=f"📈 Tendance - {culture_prev}", labels={"Value_Mean": "Rendement (hg/ha)"})
            fig.add_trace(go.Scatter(x=x, y=slope*x+intercept, mode="lines", name=f"Tendance: +{slope:.1f} hg/ha/an", line=dict(color="#d4a056", dash="dash")))
            apply_theme(fig, height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"📊 Tendance: +{slope:.1f} hg/ha par an")

# ══════════════════════════════════════════════════════════
#  PAGE CARTE DU MAROC
# ══════════════════════════════════════════════════════════
elif page == T("nav_carte"):
    st.markdown("<div class='page-title'>🗺️ Carte des Régions Agricoles</div>", unsafe_allow_html=True)
    
    CARTE_HTML = '''
<div style="width:100%;font-family:Nunito,sans-serif;">
<style>
  .rgn{cursor:pointer;transition:opacity .18s;}
  .rgn:hover{opacity:.78;}
  .rgn polygon,.rgn path{stroke:#fff;stroke-width:1.4;}
  .rlbl{pointer-events:none;fill:#fff;font-weight:700;font-size:11.5px;text-anchor:middle;}
  #ibox{background:#f0fff4;border-radius:14px;border:2px solid #6ab04c;padding:16px;margin-top:14px;}
  #ibox-nom{font-size:16px;font-weight:800;color:#1a472a;margin-bottom:6px;}
  .stat-pill{display:inline-block;background:#e8f5e9;border-radius:20px;padding:3px 10px;margin:3px 2px;font-size:11px;font-weight:700;color:#1a472a;}
</style>
<div style="display:flex;gap:18px;flex-wrap:wrap;">
  <div style="flex:1;min-width:280px;background:#e8f4fd;border-radius:16px;padding:8px;">
    <svg width="100%" viewBox="0 0 430 540">
      <g class="rgn" onclick="showR('Tanger-Tétouan','#0f766e','Maraîchage, agrumes','Méditerranéen','450 hg/ha','820mm','18°C')"><polygon points="130,34 215,34 228,58 208,82 164,84 127,68" fill="#0f766e"/><text class="rlbl" x="178" y="57">Tanger-Tétouan</text></g>
      <g class="rgn" onclick="showR('Oriental','#b45309','Céréales, oliviers','Semi-aride','280 hg/ha','310mm','20°C')"><polygon points="228,34 318,36 326,66 300,86 232,84 220,60" fill="#b45309"/><text class="rlbl" x="272" y="62">Oriental</text></g>
      <g class="rgn" onclick="showR('Fès-Meknès','#15803d','Céréales, arboriculture','Continental','380 hg/ha','490mm','17°C')"><polygon points="162,86 208,84 228,84 234,108 215,132 178,136 152,120 145,100" fill="#15803d"/><text class="rlbl" x="190" y="110">Fès-Meknès</text></g>
      <g class="rgn" onclick="showR('Rabat-Salé-Kénitra','#1d4ed8','Céréales, maraîchage','Atlantique','420 hg/ha','580mm','18°C')"><polygon points="86,100 145,100 152,120 145,148 116,155 84,140 74,118" fill="#1d4ed8"/><text class="rlbl" x="114" y="126">Rabat-Salé</text></g>
      <g class="rgn" onclick="showR('Béni Mellal-Khénifra','#c2410c','Céréales, betterave','Plaine Tadla','520 hg/ha','390mm','22°C')"><polygon points="180,136 234,122 270,130 284,162 264,190 226,194 190,180 174,155" fill="#c2410c"/><text class="rlbl" x="229" y="158">Béni Mellal</text></g>
      <g class="rgn" onclick="showR('Casablanca-Settat','#7c3aed','Légumes, élevage','Péri-urbain','310 hg/ha','420mm','19°C')"><polygon points="70,150 116,155 145,150 152,176 136,204 100,210 68,192 58,168" fill="#7c3aed"/><text class="rlbl" x="106" y="178">Casablanca</text></g>
      <g class="rgn" onclick="showR('Marrakech-Safi','#ca8a04','Oliviers, safran','Semi-aride','295 hg/ha','290mm','24°C')"><polygon points="100,212 152,206 190,200 200,230 188,262 152,270 110,258 84,234 84,214" fill="#ca8a04"/><text class="rlbl" x="147" y="234">Marrakech-Safi</text></g>
      <g class="rgn" onclick="showR('Drâa-Tafilalet','#dc2626','Dattiers, oasis','Pré-saharien','180 hg/ha','95mm','28°C')"><polygon points="198,180 264,190 300,190 322,218 308,274 272,286 228,276 200,250 190,216" fill="#dc2626"/><text class="rlbl" x="258" y="232">Drâa-Tafilalet</text></g>
      <g class="rgn" onclick="showR('Souss-Massa','#16a34a','Agrumes, tomates','Plaine Souss','610 hg/ha','195mm','21°C')"><polygon points="84,260 110,260 152,272 166,298 152,330 114,342 78,326 62,296 66,270" fill="#16a34a"/><text class="rlbl" x="113" y="300">Souss-Massa</text></g>
      <g class="rgn" onclick="showR('Guelmim-Oued Noun','#6b7280','Élevage, arganier','Zone aride','120 hg/ha','75mm','26°C')"><polygon points="62,328 114,344 152,336 160,366 142,398 100,408 60,390 44,358" fill="#6b7280"/><text class="rlbl" x="103" y="368">Guelmim</text></g>
      <g class="rgn" onclick="showR('Laâyoune','#9ca3af','Pêche, serres','Saharien','80 hg/ha','35mm','24°C')"><polygon points="44,400 100,410 142,402 150,434 126,464 82,470 42,450 30,420" fill="#9ca3af"/><text class="rlbl" x="92" y="435">Laâyoune</text></g>
      <g class="rgn" onclick="showR('Dakhla','#a3b8a3','Pêche, aquaculture','Atlantique saharien','95 hg/ha','28mm','22°C')"><polygon points="82,470 126,466 150,464 156,498 128,514 82,514 50,496 42,474" fill="#a3b8a3"/><text class="rlbl" x="100" y="492" style="fill:#2d4a2d;">Dakhla</text></g>
    </svg>
  </div>
  <div style="flex:0 0 230px;">
    <div style="font-weight:800;font-size:14px;color:#1a472a;">📍 Régions du Maroc</div>
    <div id="ibox"><div id="ibox-nom">👆 Cliquez sur une région</div><div id="ibox-cult" style="color:#777;">Sélectionnez une région.</div><div id="ibox-stats"></div></div>
  </div>
</div>
<script>
function showR(nom,couleur,cultures,climat,rend,pluie,temp){
  document.getElementById('ibox').style.borderColor=couleur;
  document.getElementById('ibox').style.background=couleur+'14';
  document.getElementById('ibox-nom').innerHTML='📍 '+nom;
  document.getElementById('ibox-cult').innerHTML='🌱 Cultures : '+cultures+'<br>🌤️ Climat : '+climat;
  document.getElementById('ibox-stats').innerHTML='<div style="margin-top:8px;"><span class="stat-pill">🌾 '+rend+'</span><span class="stat-pill">🌧️ '+pluie+'</span><span class="stat-pill">🌡️ '+temp+'</span></div>';
}
</script></div>'''
    st.components.v1.html(CARTE_HTML, height=550)
    
    st.markdown("<div class='section-title'>📊 Rendement moyen par région</div>", unsafe_allow_html=True)
    rend_reg = df_f.groupby("Region")["Value_Mean"].mean().sort_values(ascending=True).reset_index()
    fig1 = px.bar(rend_reg, x="Value_Mean", y="Region", orientation="h", color="Value_Mean", color_continuous_scale=["#b7dfbf", "#1a472a"], title="🏆 Classement des régions")
    apply_theme(fig1, height=400)
    fig1.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("<div class='section-title'>📈 Évolution temporelle par région</div>", unsafe_allow_html=True)
    selected_regions = st.multiselect("Sélectionnez les régions à comparer", REGIONS_LIST, default=REGIONS_LIST[:3])
    if selected_regions:
        df_compare = df_f[df_f["Region"].isin(selected_regions)]
        evo_reg = df_compare.groupby(["Year", "Region"])["Value_Mean"].mean().reset_index()
        fig2 = px.line(evo_reg, x="Year", y="Value_Mean", color="Region", markers=True, title="Évolution des rendements")
        apply_theme(fig2, height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("<div class='section-title'>🔗 Corrélation pluie-rendement par région</div>", unsafe_allow_html=True)
    reg_comp = df_f.groupby("Region")[["Value_Mean", "Precip_Total_mm"]].mean().reset_index()
    fig3 = px.scatter(reg_comp, x="Precip_Total_mm", y="Value_Mean", size="Value_Mean", color="Region", title="Relation pluie-rendement")
    apply_theme(fig3, height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("<div class='section-title'>🌡️ Heatmap des rendements</div>", unsafe_allow_html=True)
    top_crops_heat = df_f.groupby("Product")["Value_Mean"].mean().nlargest(8).index.tolist()
    heat_data = df_f[df_f["Product"].isin(top_crops_heat)].groupby(["Region", "Product"])["Value_Mean"].mean().unstack().fillna(0)
    if not heat_data.empty:
        fig4 = px.imshow(heat_data.values, x=list(heat_data.columns), y=list(heat_data.index), labels=dict(x="Culture", y="Région", color="Rendement"), color_continuous_scale="Greens", aspect="auto", title="Matrice des rendements")
        apply_theme(fig4, height=450)
        fig4.update_layout(xaxis=dict(tickangle=45))
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  PAGE PRÉDICTION IA (AMÉLIORÉE AVEC GRAPHIQUE DES FACTEURS)
# ══════════════════════════════════════════════════════════
elif page == T("nav_pred"):
    st.markdown("<div class='page-title'>🤖 Prédiction du Rendement</div>", unsafe_allow_html=True)
    
    if model_loaded:
        st.success("✅ Modèle XGBoost chargé (R²=0.9259)")
    
    # Section d'entrée des informations
    st.markdown("<div class='section-title'>📝 Entrez vos informations</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌧️ Conditions climatiques")
        precip = st.slider("💧 Précipitations (mm)", 50, 800, 350, help="Quantité totale de pluie sur la saison")
        temp = st.slider("🌡️ Température moyenne (°C)", 10, 35, 20, help="Température moyenne pendant la saison de croissance")
        humid = st.slider("💦 Humidité relative (%)", 30, 90, 55, help="Humidité moyenne de l'air")
        solar = st.slider("☀️ Rayonnement solaire (MJ/m²)", 5, 30, 18, help="Énergie solaire reçue")
    
    with col2:
        st.markdown("#### 🌿 Paramètres culturaux")
        search_pred = st.text_input(T("search_crop"), placeholder="ex: blé, tomate, orge...", key="search_pred", help="Recherchez votre culture")
        filtered_pred = [c for c in ALL_CROPS if search_pred.lower() in c.lower()] if search_pred else ALL_CROPS
        culture_pred = st.selectbox("🌾 Culture", filtered_pred if filtered_pred else ALL_CROPS, help="Choisissez la culture que vous voulez analyser")
        
        sol = st.selectbox("🪨 Type de sol", ["Limoneux", "Argileux", "Sableux", "Calcaire", "Mixte"], 
                          help="Le type de sol influence la rétention d'eau et les nutriments")
        irrigation = st.radio("💧 Irrigation", ["Oui", "Non"], horizontal=True, 
                             help="L'irrigation peut augmenter significativement le rendement")
        engrais = st.select_slider("🌱 Niveau d'engrais", ["Aucun", "Faible", "Moyen", "Élevé", "Intensif"],
                                  help="La fertilisation améliore la nutrition des plantes")
    
    # Coefficients pour la prédiction
    coef_sol = {"Limoneux": 1.0, "Argileux": 0.92, "Sableux": 0.85, "Calcaire": 0.90, "Mixte": 0.96}
    coef_irr = 1.18 if irrigation == "Oui" else 1.0
    coef_eng = {"Aucun": 0.72, "Faible": 0.88, "Moyen": 1.0, "Élevé": 1.14, "Intensif": 1.22}
    
    # Coefficient culture
    if culture_pred in df["Product"].values:
        crop_mean = df[df["Product"] == culture_pred]["Value_Mean"].mean()
        coef_cult = crop_mean / df["Value_Mean"].mean() if df["Value_Mean"].mean() > 0 else 1.0
    else:
        coef_cult = 1.0
    
    # Calcul de la prédiction
    base = 1850 + (precip * 5.1) - (temp * 68) + (humid * 12) + (solar * 42)
    pred = base * coef_sol[sol] * coef_irr * coef_cult * coef_eng[engrais]
    pred = max(700, min(6500, pred))
    
    # Bouton de prédiction
    if st.button("🔮 Calculer mon rendement estimé", use_container_width=True):
        # Détermination du niveau
        if pred >= 3800:
            niveau, emoji, color = "Excellent 🏆", "🏆", "#16a34a"
        elif pred >= 2800:
            niveau, emoji, color = "Bon ✅", "✅", "#6ab04c"
        elif pred >= 1800:
            niveau, emoji, color = "Moyen ⚠️", "⚠️", "#d4a056"
        else:
            niveau, emoji, color = "Faible 📉", "📉", "#dc2626"
        
        # Affichage du résultat
        st.markdown(f"""
        <div class='resultat-ia'>
            <div style='font-size: 1.8rem;'>{emoji}</div>
            <div style='font-size: 1.2rem; font-weight: 600; margin-top: 8px;'>{T("pred_result")}</div>
            <div style='font-size: 2rem; font-weight: 800;'>{pred:,.0f} hg/ha</div>
            <div style='font-size: 1rem; margin-top: 8px; background: {color}22; display: inline-block; padding: 4px 16px; border-radius: 20px;'>
                {niveau}
            </div>
            <div style='font-size: 0.85rem; margin-top: 12px; opacity: 0.8;'>
                🌾 {culture_pred} · 🪨 {sol} · 💧 {irrigation}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphique des facteurs d'influence
        st.markdown("<div class='section-title'>📊 Facteurs d'influence sur le rendement</div>", unsafe_allow_html=True)
        
        # Calcul des contributions relatives
        contributions = {
            "🌧️ Précipitations": abs(precip * 5.1),
            "🌡️ Température": abs(temp * 68),
            "☀️ Rayonnement": abs(solar * 42),
            "💧 Humidité": abs(humid * 12)
        }
        
        # Normalisation
        total = sum(contributions.values())
        if total > 0:
            for k in contributions:
                contributions[k] = (contributions[k] / total) * 100
        
        # Création du graphique à barres horizontales
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(
            x=list(contributions.values()),
            y=list(contributions.keys()),
            orientation='h',
            marker_color=["#1565c0", "#c0392b", "#d4a056", "#1a472a"],
            text=[f"{v:.1f}%" for v in contributions.values()],
            textposition='outside',
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
        ))
        
        fig_f.update_layout(
            title="🔍 Contribution des facteurs climatiques",
            xaxis_title="Contribution relative (%)",
            yaxis_title="",
            height=300,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Nunito", size=12, color="#1a1a1a"),
            xaxis=dict(
                tickfont=dict(color="#1a1a1a"),
                gridcolor="#e0e8e0",
                range=[0, 100]
            ),
            yaxis=dict(
                tickfont=dict(color="#1a1a1a"),
                gridcolor="#e0e8e0"
            ),
            margin=dict(l=10, r=100, t=50, b=10)
        )
        st.plotly_chart(fig_f, use_container_width=True)
        
        # Explication des facteurs
        with st.expander("ℹ️ Comprendre les facteurs d'influence"):
            st.markdown("""
            <div style='background:#f8faf8; border-radius:10px; padding:15px;'>
                <p><b>🌧️ Précipitations :</b> Représente <b>{:.1f}%</b> de l'influence. L'eau est essentielle pour la croissance des plantes.</p>
                <p><b>🌡️ Température :</b> Représente <b>{:.1f}%</b> de l'influence. Les températures extrêmes peuvent stresser les cultures.</p>
                <p><b>☀️ Rayonnement solaire :</b> Représente <b>{:.1f}%</b> de l'influence. La photosynthèse dépend de la lumière.</p>
                <p><b>💧 Humidité :</b> Représente <b>{:.1f}%</b> de l'influence. L'humidité affecte la transpiration et les maladies.</p>
            </div>
            """.format(contributions["🌧️ Précipitations"], contributions["🌡️ Température"], 
                      contributions["☀️ Rayonnement"], contributions["💧 Humidité"]), unsafe_allow_html=True)
        
        # Sauvegarde de la prédiction
        col_sv1, col_sv2 = st.columns([3, 1])
        with col_sv2:
            if st.button(T("pred_save"), key="btn_save_pred", help="Sauvegarder cette prédiction dans votre historique"):
                history = st.session_state.pred_history
                username = st.session_state.username
                if username not in history:
                    history[username] = []
                history[username].append({
                    "date": str(datetime.now().strftime("%Y-%m-%d %H:%M")),
                    "culture": culture_pred,
                    "rendement": round(pred),
                    "niveau": niveau,
                    "precip": precip, 
                    "temp": temp, 
                    "sol": sol,
                    "irrigation": irrigation, 
                    "engrais": engrais
                })
                save_history(history)
                st.session_state.pred_history = history
                st.success(T("pred_saved"))

# ══════════════════════════════════════════════════════════
#  PAGE CONSEILS PRATIQUES
# ══════════════════════════════════════════════════════════
elif page == T("nav_conseils"):
    st.markdown("<div class='page-title'>💡 Conseils Pratiques</div>", unsafe_allow_html=True)
    situation = st.selectbox("Votre situation", ["Pluie insuffisante (sécheresse)", "Températures trop élevées", "Rendement en baisse", "Je prépare la prochaine saison"])
    conseils_data = {
        "Pluie insuffisante (sécheresse)": [("💧", "Irrigation goutte-à-goutte", "Économise 40-60% d'eau"), ("🌱", "Variétés résistantes", "Utilisez des variétés certifiées"), ("🪨", "Paillage", "Réduit l'évaporation")],
        "Températures trop élevées": [("🌅", "Arrosage matinal", "Arrosez avant 8h"), ("☂️", "Filets d'ombrage", "Réduisent la température"), ("💦", "Paillage", "Maintient l'humidité")],
        "Rendement en baisse": [("🧪", "Analyse de sol", "Faites analyser votre sol"), ("🔄", "Rotation des cultures", "Améliore la fertilité"), ("🌱", "Semences certifiées", "Utilisez des semences de qualité")],
        "Je prépare la prochaine saison": [("🗓️", "Planification", "Établissez un calendrier"), ("💰", "Budget prévisionnel", "Prévoyez vos intrants"), ("👨‍🌾", "Conseil agricole", "Consultez un technicien")]
    }
    for ico, titre, texte in conseils_data[situation]:
        st.markdown(f"""<div class='conseil-card'><div style='font-weight:600;'>{ico} {titre}</div><div>{texte}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE ASSISTANT IA
# ══════════════════════════════════════════════════════════
elif page == T("nav_chat"):
    st.markdown("<div class='page-title'>💬 Assistant Agricole</div>", unsafe_allow_html=True)
    def get_response(q):
        if "blé" in q.lower(): return "Le blé tendre est la céréale principale au Maroc. Rendement: 2000-3500 hg/ha."
        elif "orge" in q.lower(): return "L'orge résiste mieux à la sécheresse. Rendement: 1500-2800 hg/ha."
        elif "tomate" in q.lower(): return "La tomate est cultivée dans le Souss-Massa. Rendement serre: 25000-40000 hg/ha."
        else: return "Posez une question sur: blé, orge, tomate, irrigation, engrais."
    for msg in st.session_state.chat_history:
        if msg["role"] == "user": st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else: st.markdown(f"<div class='chat-bot'>🌾 {msg['content']}</div>", unsafe_allow_html=True)
    question = st.text_input("Votre question:", placeholder="ex: Parle-moi du blé")
    if st.button("Envoyer") and question.strip():
        rep = get_response(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": rep})
        st.rerun()

# ══════════════════════════════════════════════════════════
#  PAGE MON PROFIL
# ══════════════════════════════════════════════════════════
elif page == T("nav_profil"):
    user_data = st.session_state.users.get(st.session_state.username, {})
    _ini = ""
    if user_data.get("prenom"): _ini += user_data["prenom"][0].upper()
    if user_data.get("nom"): _ini += user_data["nom"][0].upper()
    if not _ini: _ini = st.session_state.username[0].upper() if st.session_state.username else "A"
    st.markdown(f"""<div style='background:linear-gradient(135deg,#1a472a,#2d5a3a); border-radius:16px; padding:20px; margin-bottom:20px; color:white;'><div style='display:flex; align-items:center; gap:18px;'><div class='avatar-circle'>{_ini}</div><div><div style='font-size:1.3rem; font-weight:600;'>{user_data.get('prenom', '')} {user_data.get('nom', '')}</div><div>@{st.session_state.username} · 📍 {user_data.get('region', '')}</div></div></div></div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: np_prenom = st.text_input("Prénom", value=user_data.get("prenom", ""))
    with c2: np_nom = st.text_input("Nom", value=user_data.get("nom", ""))
    regions_list = REGIONS_LIST if REGIONS_LIST else ["Béni Mellal-Khénifra", "Fès-Meknès", "Souss-Massa", "Marrakech-Safi", "Rabat-Salé-Kénitra"]
    cur_reg = user_data.get("region", regions_list[0])
    np_region = st.selectbox("Région", regions_list, index=regions_list.index(cur_reg) if cur_reg in regions_list else 0)
    np_cultures = st.multiselect("Cultures pratiquées", ALL_CROPS, default=user_data.get("cultures", []))
    if st.button(T("save_profile")):
        st.session_state.users[st.session_state.username].update({"prenom": np_prenom.strip(), "nom": np_nom.strip(), "region": np_region, "cultures": np_cultures})
        save_users(st.session_state.users)
        st.success("✅ Profil mis à jour !")
        st.rerun()

# ══════════════════════════════════════════════════════════
#  PAGE HISTORIQUE
# ══════════════════════════════════════════════════════════
elif page == T("nav_history"):
    st.markdown(f"<div class='page-title'>{T('history_title')}</div>", unsafe_allow_html=True)
    history = st.session_state.pred_history.get(st.session_state.username, [])
    if not history:
        st.info(T("history_empty"))
    else:
        for pred in reversed(history[-10:]):
            st.markdown(f"""<div class='history-item'><div><strong>🌾 {pred.get('culture', '—')}</strong> - {pred.get('rendement', 0)} hg/ha</div><div style='font-size:0.7rem;'>📅 {pred.get('date', '—')[:16]}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE GUIDE & À PROPOS
# ══════════════════════════════════════════════════════════
elif page == T("nav_about"):
    st.markdown("<div class='page-title'>📖 Guide d'utilisation & À Propos</div>", unsafe_allow_html=True)
    
    guide_tabs = st.tabs(["🚀 Guide de démarrage", "📊 Navigation", "🎯 Fonctionnalités", "ℹ️ À Propos"])
    
    with guide_tabs[0]:
        st.markdown("""
        <div style='background:#1a472a; border-radius:12px; padding:20px; border:1px solid #6ab04c;'>
            <h3 style='color:#6ab04c;'>🚀 Bienvenue sur AgriClim Maroc !</h3>
            <p style='color:white;'>Cette application vous aide à suivre, analyser et prédire les rendements agricoles au Maroc.</p>
            <hr style='border-color:#6ab04c; margin:16px 0;'>
            <h4 style='color:#6ab04c;'>📝 Étape 1 : Créer un compte ou vous connecter</h4>
            <ul style='color:white;'><li>Sur la page d'accueil, cliquez sur "Créer un compte"</li><li>Remplissez vos informations (nom, prénom, région, mot de passe)</li><li>Compte démo : <b style='color:#6ab04c;'>agriclim</b> / <b style='color:#6ab04c;'>agriclim2026</b></li></ul>
            <h4 style='color:#6ab04c;'>🎯 Étape 2 : Définir votre période d'analyse</h4>
            <ul style='color:white;'><li>Dans le menu latéral, utilisez le curseur "Période"</li><li>Sélectionnez la plage d'années à analyser</li><li>Toutes les données s'adapteront automatiquement</li></ul>
            <h4 style='color:#6ab04c;'>🌍 Étape 3 : Choisir votre langue</h4>
            <ul style='color:white;'><li>En haut du menu latéral, sélectionnez Français 🇫🇷 ou العربية 🇲🇦</li><li>L'interface se traduit automatiquement</li></ul>
            <div style='margin-top:16px; background:#2d5a3a; border-radius:8px; padding:12px;'>
                <p style='color:#6ab04c;'>💡 <b>Astuce :</b> Connectez-vous pour sauvegarder vos prédictions et personnaliser votre profil !</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with guide_tabs[1]:
        st.markdown("""
        <div style='background:#1a472a; border-radius:12px; padding:20px; border:1px solid #6ab04c;'>
            <h3 style='color:#6ab04c;'>📊 Les différentes sections de l'application</h3>
            <table style='width:100%; border-collapse:collapse;'>
                <tr style='background:#2d5a3a;'>
                    <th style='padding:10px; text-align:left; color:#6ab04c;'>Section</th>
                    <th style='padding:10px; text-align:left; color:#6ab04c;'>Fonctionnalités</th>
                </tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>🏠 Accueil</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Météo actuelle, indicateurs clés, graphiques</td></tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>🌦️ Météo & Climat</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Météo temps réel + analyses climatiques</td></tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>📈 Rendements</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Analyse des rendements par culture</td></tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>🗺️ Carte du Maroc</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Carte interactive + étude comparative</td></tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>🤖 Prédiction IA</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Estimation du rendement avec analyse des facteurs</td></tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>💡 Conseils Pratiques</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Recommandations par situation</td></tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>💬 Assistant IA</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Chatbot agricole</td></tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>👤 Mon Profil</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Gérez vos informations</td></tr>
                <tr><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>📋 Historique</td><td style='padding:8px; border-bottom:1px solid #2d5a3a; color:white;'>Vos prédictions sauvegardées</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with guide_tabs[2]:
        st.markdown("""
        <div style='background:#1a472a; border-radius:12px; padding:20px; border:1px solid #6ab04c;'>
            <h3 style='color:#6ab04c;'>🎯 Détail des fonctionnalités principales</h3>
            <h4 style='color:#6ab04c;'>🔍 Prédiction IA</h4>
            <ul style='color:white;'><li>Ajustez les conditions météo (pluie, température, humidité, rayonnement)</li><li>Choisissez votre culture, type de sol, irrigation et niveau d'engrais</li><li>Obtenez un rendement estimé avec un niveau d'appréciation</li><li>Visualisez les facteurs d'influence (graphique des contributions)</li><li>Sauvegardez vos prédictions dans l'historique</li></ul>
            <h4 style='color:#6ab04c;'>🗺️ Étude comparative</h4>
            <ul style='color:white;'><li>Comparaison des rendements par région</li><li>Sélection multiple de régions pour analyse temporelle</li><li>Corrélation pluie-rendement</li><li>Matrice des rendements par région et culture</li></ul>
            <div style='margin-top:16px; background:#2d5a3a; border-radius:8px; padding:12px;'>
                <p style='color:#6ab04c;'>⚠️ <b>Note :</b> Modèle XGBoost avec R²=0.9259</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with guide_tabs[3]:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div style='background:#1a472a; border-radius:12px; padding:20px; border:1px solid #6ab04c;'>
                <h3 style='color:#6ab04c;'>🎓 Projet de Fin d'Études (PFE) 2026</h3>
                <p style='color:white;'><b>AgriClim Maroc</b> - Application d'aide à la décision agricole<br>EST Fkih Ben Salah</p>
                <hr style='border-color:#6ab04c; margin:16px 0;'>
                <h4 style='color:#6ab04c;'>📊 Performances du modèle :</h4>
                <ul style='color:white;'><li>R² = 0,9259</li><li>RMSE = 3 780 hg/ha</li><li>MAPE = 21,36%</li></ul>
                <h4 style='color:#6ab04c;'>🗺️ Régions couvertes :</h4>
                <p style='color:white;'>12 régions agricoles marocaines</p>
                <h4 style='color:#6ab04c;'>👨‍🌾 Encadrement :</h4>
                <p style='color:white;'><b>Mme Ichrak Khoulqi</b><br>EST Fkih Ben Salah</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background:#1a472a; border-radius:12px; padding:20px; text-align:center; border:1px solid #6ab04c;'>
                <div style='font-size:48px;'>🌾</div>
                <div style='font-weight:700; color:#6ab04c; font-size:18px;'>AgriClim Maroc</div>
                <div style='font-size:13px; color:white;'>Version 3.0 — PFE 2026</div>
                <hr style='border-color:#6ab04c; margin:16px 0;'>
                <div style='text-align:left;'><h4 style='color:#6ab04c;'>🛠️ Stack technique :</h4><ul style='color:white;'><li>🐍 Python + Streamlit</li><li>📊 Plotly + Pandas</li><li>🤖 XGBoost</li><li>🌐 Open-Meteo API</li></ul></div>
                <hr style='border-color:#6ab04c; margin:16px 0;'>
                <div style='text-align:left;'><h4 style='color:#6ab04c;'>📚 Sources :</h4><ul style='color:white;'><li>FAOSTAT</li><li>NASA POWER</li><li>Open-Meteo</li></ul></div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center; font-size:11px; color:#8a9a8a; padding:10px; border-top:1px solid #e0e8e0; margin-top:20px;'>
    🌾 <b>AgriClim Maroc v3.0</b> — Application d'analyse agricole intelligente | PFE 2026<br>© EST Fkih Ben Salah
</div>
""", unsafe_allow_html=True)
