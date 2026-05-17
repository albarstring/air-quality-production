"""
app.py  –  Dashboard Prediksi Kualitas Udara
Jalankan: streamlit run app.py
"""

# ─── Standard imports ──────────────────────────────────────────────────────────
import json, math, warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AirSense – Prediksi Kualitas Udara",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  –  dark industrial dashboard aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root tokens ─────────────────────────────────────────────────── */
:root {
    --bg-0:    #0d1117;
    --bg-1:    #161b22;
    --bg-2:    #1c2230;
    --bg-3:    #21293b;
    --border:  #30384a;
    --text-1:  #e6edf3;
    --text-2:  #8b949e;
    --accent:  #38bdf8;
    --accent2: #818cf8;
    --good:    #22c55e;
    --warn:    #eab308;
    --danger1: #f97316;
    --danger2: #ef4444;
    --danger3: #a855f7;
    --dead:    #6b7280;
}

/* ── Global ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background: var(--bg-0) !important;
    color: var(--text-1) !important;
}
.main .block-container { padding: 1.5rem 2rem 4rem; max-width: 1400px; }

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--accent) !important;
}

/* ── Sliders ──────────────────────────────────────────────────────── */
.stSlider [data-testid="stSlider"] { accent-color: var(--accent); }

/* ── KPI card ─────────────────────────────────────────────────────── */
.kpi-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    display: flex; flex-direction: column; gap: 4px;
    transition: border-color .2s;
}
.kpi-card:hover { border-color: var(--accent); }
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    color: var(--text-2);
}
.kpi-value { font-size: 28px; font-weight: 800; line-height: 1; }
.kpi-sub   { font-size: 11px; color: var(--text-2); margin-top: 2px; }

/* ── AQI gauge wrapper ───────────────────────────────────────────── */
.gauge-wrap {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}

/* ── Alert box ───────────────────────────────────────────────────── */
.alert-box {
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border-left: 5px solid;
    margin: 1rem 0;
    font-size: 15px;
}
.alert-good    { background: #052e1680; border-color: var(--good);    color: #86efac; }
.alert-moderate{ background: #42300580; border-color: var(--warn);    color: #fde047; }
.alert-usg     { background: #431d0580; border-color: var(--danger1); color: #fdba74; }
.alert-us      { background: #450a0a80; border-color: var(--danger2); color: #fca5a5; }
.alert-vus     { background: #2e0a5480; border-color: var(--danger3); color: #d8b4fe; }
.alert-haz     { background: #1a1f2880; border-color: var(--dead);    color: #9ca3af; }

/* ── Section heading ─────────────────────────────────────────────── */
.section-head {
    font-size: 13px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: var(--text-2);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px; margin: 1.5rem 0 1rem;
}

/* ── Metric model badge ──────────────────────────────────────────── */
.badge {
    display: inline-block;
    background: #0ea5e920;
    border: 1px solid #0ea5e940;
    color: var(--accent);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
}

/* ── Button ──────────────────────────────────────────────────────── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #818cf8);
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.75rem !important;
    letter-spacing: 1px;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Tab strip ───────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-1);
    border-radius: 10px 10px 0 0;
    border-bottom: 1px solid var(--border);
    gap: 4px; padding: 4px 8px 0;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--text-2) !important;
    padding: 8px 18px !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: var(--bg-2) !important;
    color: var(--text-1) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Expander ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Dataframe ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Scroll bar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ── Hide streamlit branding ─────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR  = Path("model_output")
FEATURES   = [
    "pm25","pm10","co","no2","o3",
    "temperature","humidity","wind_speed","precipitation",
    "pm25_pm10_ratio","heat_index","total_polutan",
    "is_rush_hour","is_daytime","is_weekend",
    "sin_hour","cos_hour","sin_weekday",
    "wind_humidity_interact",
]
LATITUDE   = -7.797068
LONGITUDE  = 110.370529

AQI_LEVELS = [
    (  0,  50, "Baik",                   "good",    "#22c55e", "✅",
       "Kualitas udara memuaskan dan risiko polusi udara minim."),
    ( 51, 100, "Sedang",                 "moderate","#eab308", "🟡",
       "Kualitas udara dapat diterima. Kelompok sensitif mungkin terpengaruh."),
    (101, 150, "Tidak Sehat (Sensitif)", "usg",     "#f97316", "🟠",
       "Kelompok sensitif (anak-anak, lansia, penderita asma) berisiko."),
    (151, 200, "Tidak Sehat",            "us",      "#ef4444", "🔴",
       "Seluruh masyarakat mungkin merasakan efek kesehatan."),
    (201, 300, "Sangat Tidak Sehat",     "vus",     "#a855f7", "🟣",
       "Peringatan darurat kesehatan! Seluruh populasi berisiko."),
    (301, 500, "Berbahaya",              "haz",     "#6b7280", "☠️",
       "BAHAYA SERIUS – hindari semua aktivitas luar ruang."),
]

ACTIONS = {
    "Baik":                   ["Aktivitas luar ruang bebas dilakukan","Ventilasi rumah dapat dibuka lebar","Olahraga di luar aman untuk semua"],
    "Sedang":                 ["Batasi aktivitas fisik berat di luar","Penderita asma siapkan inhaler","Pantau kondisi jika terasa sesak"],
    "Tidak Sehat (Sensitif)": ["Kurangi waktu di luar ruang","Gunakan masker jika aktivitas luar","Tutup jendela di rumah","Hindari olahraga intensitas tinggi di luar"],
    "Tidak Sehat":            ["Gunakan masker N95 di luar ruang","Batasi waktu di luar semaksimal mungkin","Nyalakan air purifier di dalam ruang","Segera masuk ke ruangan ber-AC"],
    "Sangat Tidak Sehat":     ["Tetap di dalam ruangan","Masker N95 wajib jika keluar","Siapkan obat pernapasan","Evakuasi kelompok rentan ke lokasi lebih aman"],
    "Berbahaya":              ["EVAKUASI segera jika memungkinkan","Hubungi layanan darurat (119)","Segel semua ventilasi dengan isolasi","Jangan keluar dalam kondisi apapun"],
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model  = joblib.load(MODEL_DIR / "LightGBM.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    meta   = json.loads((MODEL_DIR / "metadata.json").read_text())
    return model, scaler, meta


def get_aqi_info(aqi_val: float):
    for lo, hi, label, cls, color, icon, desc in AQI_LEVELS:
        if lo <= aqi_val <= hi:
            return label, cls, color, icon, desc
    return "Berbahaya", "haz", "#6b7280", "☠️", "BAHAYA SERIUS!"


def engineer_features(raw: dict, hour: int, weekday: int, is_weekend: int) -> np.ndarray:
    pm25, pm10 = raw["pm25"], raw["pm10"]
    temp       = raw["temperature"]
    hum        = raw["humidity"]

    feat = {
        "pm25":        pm25,
        "pm10":        pm10,
        "co":          raw["co"],
        "no2":         raw["no2"],
        "o3":          raw["o3"],
        "temperature": temp,
        "humidity":    hum,
        "wind_speed":  raw["wind_speed"],
        "precipitation": raw["precipitation"],
        "pm25_pm10_ratio": pm25 / (pm10 + 1e-6),
        "heat_index":  temp + 0.33*(hum/100)*6.105*math.exp(17.27*temp/(237.7+temp)) - 4,
        "total_polutan": pm25 + pm10 + raw["no2"]*0.1 + raw["co"]*10,
        "is_rush_hour": int((7<=hour<=9) or (17<=hour<=19)),
        "is_daytime":   int(6<=hour<=18),
        "is_weekend":   is_weekend,
        "sin_hour":     math.sin(2*math.pi*hour/24),
        "cos_hour":     math.cos(2*math.pi*hour/24),
        "sin_weekday":  math.sin(2*math.pi*weekday/7),
        "wind_humidity_interact": raw["wind_speed"] * hum,
    }
    return np.array([[feat[f] for f in FEATURES]])


def predict_aqi(raw: dict, model, scaler, hour: int, weekday: int, is_weekend: int) -> float:
    X = engineer_features(raw, hour, weekday, is_weekend)
    return float(model.predict(scaler.transform(X))[0])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather():
    try:
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={LATITUDE}&longitude={LONGITUDE}"
               f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
               f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
               f"&forecast_days=3")
        r = requests.get(url, timeout=6)
        d = r.json()
        cur = d.get("current", {})
        hourly = d.get("hourly", {})
        return {
            "temperature":   cur.get("temperature_2m", 28.0),
            "humidity":      cur.get("relative_humidity_2m", 72.0),
            "wind_speed":    cur.get("wind_speed_10m", 4.0),
            "precipitation": cur.get("precipitation", 0.0),
            "hourly":        hourly,
        }
    except Exception:
        return None


def forecast_24h(base: dict, model, scaler, hour_now: int, weekday: int, is_weekend: int):
    rows = []
    for h in range(24):
        raw = base.copy()
        raw["temperature"] = base["temperature"] + 3*math.sin(2*math.pi*h/24 - math.pi/2)
        if (7 <= h <= 9) or (17 <= h <= 19):
            raw["pm25"] = base["pm25"] * 1.4
            raw["co"]   = base["co"] * 1.3
        elif 0 <= h <= 5:
            raw["pm25"] = base["pm25"] * 0.7
        aqi = predict_aqi(raw, model, scaler, h, weekday, is_weekend)
        label, cls, color, icon, _ = get_aqi_info(aqi)
        rows.append({"hour": h, "aqi": aqi, "label": label, "color": color, "icon": icon})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB HELPERS  (dark theme)
# ══════════════════════════════════════════════════════════════════════════════
DARK = dict(facecolor="#1c2230", edgecolor="none")
GRID = dict(color="#30384a", linewidth=0.5)

def dark_fig(*args, **kw):
    fig, ax = plt.subplots(*args, **kw)
    fig.patch.set_facecolor("#1c2230")
    if isinstance(ax, np.ndarray):
        for a in ax.flat:
            a.set_facecolor("#21293b")
    else:
        ax.set_facecolor("#21293b")
    return fig, ax


def style_ax(ax, *, xlabel="", ylabel="", title=""):
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.set_xlabel(xlabel, color="#8b949e", fontsize=9)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
    ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=10)
    ax.spines[:].set_color("#30384a")
    ax.grid(**GRID)


def gauge_chart(aqi_val: float, color: str):
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    theta_start, theta_end = math.pi, 0   # 180° → 0° (left to right)
    max_aqi = 300
    val = min(aqi_val, max_aqi)
    frac = val / max_aqi

    # Gradient arc segments
    cmap_colors = ["#22c55e","#eab308","#f97316","#ef4444","#a855f7","#6b7280"]
    n_seg = 120
    for i in range(n_seg):
        t0 = math.pi - (i/n_seg)*math.pi
        t1 = math.pi - ((i+1)/n_seg)*math.pi
        seg_frac = i/n_seg
        seg_idx = min(int(seg_frac * len(cmap_colors)), len(cmap_colors)-1)
        c = cmap_colors[seg_idx]
        ax.plot([t0, t1], [0.85, 0.85], lw=14, color=c, alpha=0.35,
                solid_capstyle="butt", transform=ax.transData)

    # Value arc
    theta_val = math.pi - frac*math.pi
    ax.plot([math.pi, theta_val], [0.85, 0.85], lw=14, color=color, alpha=0.95,
            solid_capstyle="butt")

    # Needle
    ax.plot([0, theta_val], [0, 0.78], lw=2.5, color="white", alpha=0.9)
    ax.plot(0, 0, "o", color="white", markersize=7)

    # AQI text
    ax.text(0, -0.25, f"{aqi_val:.1f}", ha="center", va="center",
            fontsize=28, fontweight="800", color=color,
            fontfamily="Syne", transform=ax.transData)

    ax.set_rlim(-0.3, 1.1)
    ax.set_thetamin(0); ax.set_thetamax(180)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
try:
    model, scaler, meta = load_model_and_scaler()
    model_ok = True
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}  \nPastikan `train_model.py` sudah dijalankan terlebih dahulu.")
    st.stop()
    model_ok = False

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("""
    <div style="padding: 0.25rem 0 0.5rem">
        <div style="font-size:11px;letter-spacing:4px;color:#8b949e;text-transform:uppercase;
                    font-family:'JetBrains Mono',monospace">EARLY WARNING SYSTEM</div>
        <h1 style="margin:4px 0 0;font-size:36px;font-weight:800;
                   background:linear-gradient(135deg,#38bdf8,#818cf8);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-family:'Syne',sans-serif">
            🌫️ AirSense Dashboard
        </h1>
        <div style="color:#8b949e;font-size:14px;margin-top:4px">
            Prediksi Kualitas Udara · Yogyakarta, Indonesia · LightGBM Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    now = datetime.now()
    st.markdown(f"""
    <div style="text-align:right;padding-top:1rem">
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#8b949e">
            {now.strftime('%A, %d %B %Y')}
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                    color:#38bdf8;font-weight:600">
            {now.strftime('%H:%M')} WIB
        </div>
        <span class="badge">LightGBM · R²={meta['metrics']['R2']}</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR – INPUT PARAMETER
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="font-size:28px">🎛️</div>
        <div style="font-weight:700;font-size:16px">Parameter Input</div>
        <div style="font-size:11px;color:#8b949e;margin-top:2px">
            Sesuaikan nilai polutan & cuaca
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Ambil cuaca realtime ────────────────────────────────────────────────
    with st.spinner("Mengambil data cuaca realtime…"):
        weather = fetch_weather()

    if weather:
        st.success("☁️  Cuaca realtime berhasil diambil", icon="✅")
        default_temp  = weather["temperature"]
        default_hum   = weather["humidity"]
        default_wind  = weather["wind_speed"]
        default_prec  = weather["precipitation"]
    else:
        st.warning("Cuaca API offline – menggunakan nilai default")
        default_temp, default_hum, default_wind, default_prec = 28.0, 72.0, 4.0, 0.0

    # ── Jam & hari ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>⏱ Waktu</div>", unsafe_allow_html=True)
    col_h, col_d = st.columns(2)
    with col_h:
        hour    = st.slider("Jam", 0, 23, now.hour)
    with col_d:
        weekday = st.slider("Hari (0=Sen)", 0, 6, now.weekday())
    is_weekend = int(weekday >= 5)

    # ── Polutan ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>💨 Polutan Udara</div>", unsafe_allow_html=True)
    pm25 = st.slider("PM2.5  (µg/m³)", 0.0, 350.0, 45.0, step=0.5)
    pm10 = st.slider("PM10  (µg/m³)",  0.0, 450.0, 70.0, step=0.5)
    co   = st.slider("CO  (mg/m³)",    0.1, 10.0,   1.0, step=0.05)
    no2  = st.slider("NO₂  (µg/m³)",   0.0, 200.0, 35.0, step=0.5)
    o3   = st.slider("O₃  (µg/m³)",    0.0, 180.0, 40.0, step=0.5)

    # ── Cuaca ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>🌡️ Cuaca</div>", unsafe_allow_html=True)
    temperature  = st.slider("Suhu  (°C)",          15.0, 45.0, float(round(default_temp,1)), step=0.1)
    humidity     = st.slider("Kelembapan  (%)",      20.0, 99.0, float(round(default_hum,1)),  step=0.5)
    wind_speed   = st.slider("Kecepatan Angin (m/s)", 0.0, 20.0, float(round(default_wind,1)), step=0.1)
    precipitation= st.slider("Curah Hujan  (mm)",    0.0, 50.0, float(round(default_prec,1)),  step=0.1)

    st.markdown("---")
    predict_btn = st.button("🔮  PREDIKSI AQI", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
raw_input = dict(pm25=pm25, pm10=pm10, co=co, no2=no2, o3=o3,
                 temperature=temperature, humidity=humidity,
                 wind_speed=wind_speed, precipitation=precipitation)

# Auto-predict on every slider change
aqi_val = predict_aqi(raw_input, model, scaler, hour, weekday, is_weekend)
label, cls, color, icon, desc = get_aqi_info(aqi_val)

# ── TOP KPI ROW ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("PM2.5",  f"{pm25:.1f}",  "µg/m³",  "#38bdf8"),
    ("PM10",   f"{pm10:.1f}",  "µg/m³",  "#818cf8"),
    ("CO",     f"{co:.2f}",    "mg/m³",  "#34d399"),
    ("NO₂",    f"{no2:.1f}",   "µg/m³",  "#fb923c"),
    ("O₃",     f"{o3:.1f}",    "µg/m³",  "#f472b6"),
]
for col, (lbl, val, unit, clr) in zip([k1,k2,k3,k4,k5], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value" style="color:{clr}">{val}</div>
            <div class="kpi-sub">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN SPLIT: Gauge + Alert + Forecast / Chart ──────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    # Gauge
    st.markdown("<div class='gauge-wrap'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px;letter-spacing:3px;color:#8b949e;
                text-transform:uppercase;margin-bottom:0.5rem;
                font-family:'JetBrains Mono',monospace">
        Indeks Kualitas Udara
    </div>""", unsafe_allow_html=True)
    st.pyplot(gauge_chart(aqi_val, color), use_container_width=True)
    st.markdown(f"""
    <div style="margin-top:-0.5rem;text-align:center">
        <span style="font-size:20px">{icon}</span>
        <span style="font-size:18px;font-weight:700;color:{color};margin-left:8px">{label}</span>
        <div style="font-size:12px;color:#8b949e;margin-top:4px">{desc}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alert + actions
    st.markdown(f"""
    <div class="alert-box alert-{cls}">
        <strong>{icon} {label}</strong><br>
        <span style="font-size:13px">{desc}</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Rekomendasi Tindakan", expanded=True):
        for act in ACTIONS.get(label, []):
            st.markdown(f"• {act}")

    # Model metrics
    st.markdown("<div class='section-head'>🤖 Performa Model</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{meta['metrics']['MAE']}")
    m2.metric("RMSE", f"{meta['metrics']['RMSE']}")
    m3.metric("R²",   f"{meta['metrics']['R2']}")

with right:
    tab1, tab2, tab3 = st.tabs(["📅 Prakiraan 24 Jam", "📊 Analisis Input", "🌡️ Cuaca Realtime"])

    # ── TAB 1 – 24h forecast ─────────────────────────────────────────────────
    with tab1:
        fc_df = forecast_24h(raw_input, model, scaler, hour, weekday, is_weekend)

        fig, ax = dark_fig(figsize=(9, 3.8))
        bars = ax.bar(fc_df["hour"], fc_df["aqi"],
                      color=fc_df["color"].tolist(), alpha=0.88, width=0.75,
                      edgecolor="none")

        for threshold, c, lbl in [(50,"#22c55e","Baik 50"),(100,"#eab308","Sedang 100"),(150,"#ef4444","Tidak Sehat 150")]:
            ax.axhline(threshold, color=c, linestyle="--", linewidth=1, alpha=0.6)
            ax.text(23.6, threshold+1, lbl, color=c, fontsize=7, va="bottom")

        # Current hour marker
        ax.axvline(hour, color="white", linewidth=1.5, linestyle=":", alpha=0.5)
        ax.text(hour, fc_df["aqi"].max()*1.04, "▼ Sekarang", color="white",
                fontsize=8, ha="center", va="bottom", alpha=0.7)

        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
        style_ax(ax, xlabel="Jam (WIB)", ylabel="Prediksi AQI",
                 title="Prakiraan AQI 24 Jam ke Depan")
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)

        # Summary table
        summary = fc_df.groupby("label")["hour"].apply(list).reset_index()
        summary.columns = ["Kategori","Jam-jam"]
        summary["Jumlah Jam"] = summary["Jam-jam"].apply(len)
        st.dataframe(summary[["Kategori","Jumlah Jam"]].sort_values("Jumlah Jam", ascending=False),
                     use_container_width=True, hide_index=True)

    # ── TAB 2 – Input analysis ───────────────────────────────────────────────
    with tab2:
        fig2, axes = plt.subplots(2, 2, figsize=(9, 6))
        fig2.patch.set_facecolor("#1c2230")

        # ① Radar / bar feature contribution
        feature_vals = [pm25/350, pm10/450, co/10, no2/200, o3/180,
                        temperature/45, humidity/100, wind_speed/20]
        feat_labels  = ["PM2.5","PM10","CO","NO₂","O₃","Suhu","Lembap","Angin"]
        bar_colors   = ["#38bdf8","#818cf8","#34d399","#fb923c","#f472b6",
                        "#fbbf24","#60a5fa","#a3e635"]

        ax = axes[0, 0]; ax.set_facecolor("#21293b")
        bars = ax.barh(feat_labels, feature_vals, color=bar_colors, alpha=0.85, height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Nilai Ternormalisasi (0–1)", color="#8b949e", fontsize=8)
        ax.set_title("Nilai Relatif Parameter Input", color="#e6edf3", fontsize=10, fontweight="bold")
        ax.spines[:].set_color("#30384a"); ax.tick_params(colors="#8b949e", labelsize=8)
        ax.grid(axis="x", **GRID)
        for bar, v in zip(bars, feature_vals):
            ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.2f}", va="center", fontsize=7, color="#8b949e")

        # ② PM2.5 vs AQI sensitivity
        ax = axes[0, 1]; ax.set_facecolor("#21293b")
        pm25_range = np.linspace(5, 250, 80)
        aqi_sens   = []
        for p in pm25_range:
            tmp = raw_input.copy(); tmp["pm25"] = p
            aqi_sens.append(predict_aqi(tmp, model, scaler, hour, weekday, is_weekend))
        ax.plot(pm25_range, aqi_sens, color="#38bdf8", linewidth=2)
        ax.axvline(pm25, color="#f97316", linestyle="--", linewidth=1.5, label=f"PM2.5={pm25:.0f}")
        ax.axhline(aqi_val, color="#818cf8", linestyle="--", linewidth=1, alpha=0.7)
        ax.scatter([pm25],[aqi_val], color="#f97316", s=60, zorder=5)
        style_ax(ax, xlabel="PM2.5 (µg/m³)", ylabel="AQI", title="Sensitivitas PM2.5 → AQI")
        ax.legend(fontsize=8, labelcolor="#8b949e", framealpha=0)

        # ③ Wind speed vs AQI
        ax = axes[1, 0]; ax.set_facecolor("#21293b")
        wind_range = np.linspace(0.1, 15, 60)
        aqi_wind   = []
        for w in wind_range:
            tmp = raw_input.copy(); tmp["wind_speed"] = w
            aqi_wind.append(predict_aqi(tmp, model, scaler, hour, weekday, is_weekend))
        ax.plot(wind_range, aqi_wind, color="#34d399", linewidth=2)
        ax.axvline(wind_speed, color="#f97316", linestyle="--", linewidth=1.5, label=f"Angin={wind_speed:.1f}")
        ax.scatter([wind_speed],[aqi_val], color="#f97316", s=60, zorder=5)
        style_ax(ax, xlabel="Kecepatan Angin (m/s)", ylabel="AQI", title="Pengaruh Angin terhadap AQI")
        ax.legend(fontsize=8, labelcolor="#8b949e", framealpha=0)

        # ④ AQI kategori gauge bar
        ax = axes[1, 1]; ax.set_facecolor("#21293b")
        categories = ["Baik\n0–50","Sedang\n51–100","Tdk Sehat\n(Sensitif)\n101–150",
                      "Tdk Sehat\n151–200","Sangat Tdk\nSehat\n201–300","Berbahaya\n301+"]
        cat_colors = ["#22c55e","#eab308","#f97316","#ef4444","#a855f7","#6b7280"]
        cat_widths = [50, 50, 50, 50, 100, 100]
        lefts = [0, 50, 100, 150, 200, 300]
        for lft, wid, clr, cat in zip(lefts, cat_widths, cat_colors, categories):
            ax.barh(0, wid, left=lft, color=clr, alpha=0.7, height=0.6)
            ax.text(lft+wid/2, 0, cat, ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")
        ax.axvline(min(aqi_val,400), color="white", linewidth=3, ymin=0.15, ymax=0.85)
        ax.text(min(aqi_val,400), 0.38, f"▲\n{aqi_val:.0f}", color="white",
                ha="center", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 400); ax.set_ylim(-0.5, 0.5)
        ax.set_title("Posisi AQI pada Skala Kategori", color="#e6edf3", fontsize=10, fontweight="bold")
        ax.axis("off")

        fig2.tight_layout(pad=2)
        st.pyplot(fig2, use_container_width=True)

    # ── TAB 3 – Realtime weather ──────────────────────────────────────────────
    with tab3:
        if weather and "hourly" in weather:
            times    = pd.to_datetime(weather["hourly"]["time"])
            temps_h  = weather["hourly"]["temperature_2m"]
            hums_h   = weather["hourly"]["relative_humidity_2m"]
            winds_h  = weather["hourly"]["wind_speed_10m"]
            precs_h  = weather["hourly"]["precipitation"]

            # Show only next 48h
            n = min(48, len(times))

            fig3, axes3 = plt.subplots(2, 2, figsize=(9, 5.5))
            fig3.patch.set_facecolor("#1c2230")

            def wplot(ax, x, y, color, title, ylabel):
                ax.set_facecolor("#21293b")
                ax.plot(range(n), y[:n], color=color, linewidth=1.8)
                ax.fill_between(range(n), y[:n], alpha=0.15, color=color)
                ax.set_title(title, color="#e6edf3", fontsize=10, fontweight="bold")
                ax.set_ylabel(ylabel, color="#8b949e", fontsize=8)
                ax.spines[:].set_color("#30384a")
                ax.tick_params(colors="#8b949e", labelsize=7)
                ax.grid(**GRID)
                step = max(1, n//8)
                ax.set_xticks(range(0, n, step))
                ax.set_xticklabels([str(times[i].strftime('%d/%m\n%H:%M')) for i in range(0, n, step)], fontsize=6)

            wplot(axes3[0,0], times[:n], temps_h,  "#fb923c", "Suhu (°C)",           "°C")
            wplot(axes3[0,1], times[:n], hums_h,   "#60a5fa", "Kelembapan (%)",       "%")
            wplot(axes3[1,0], times[:n], winds_h,  "#34d399", "Kecepatan Angin (m/s)","m/s")
            axes3[1,1].set_facecolor("#21293b")
            axes3[1,1].bar(range(n), precs_h[:n], color="#818cf8", alpha=0.8, width=0.8)
            axes3[1,1].set_title("Curah Hujan (mm)", color="#e6edf3", fontsize=10, fontweight="bold")
            axes3[1,1].spines[:].set_color("#30384a")
            axes3[1,1].tick_params(colors="#8b949e", labelsize=7)
            axes3[1,1].grid(**GRID)

            fig3.tight_layout(pad=2)
            st.pyplot(fig3, use_container_width=True)

            w1, w2, w3, w4 = st.columns(4)
            w1.metric("🌡️ Suhu",      f"{weather['temperature']:.1f} °C")
            w2.metric("💧 Kelembapan", f"{weather['humidity']:.0f} %")
            w3.metric("💨 Angin",      f"{weather['wind_speed']:.1f} m/s")
            w4.metric("🌧️ Hujan",     f"{weather['precipitation']:.1f} mm")
        else:
            st.info("Data cuaca realtime tidak tersedia. Periksa koneksi internet.")

# ── BOTTOM SECTION – AQI Reference table ──────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📖 Tabel Referensi Standar AQI (US EPA)", expanded=False):
    ref_data = {
        "Nilai AQI":  ["0–50",   "51–100",   "101–150",              "151–200",     "201–300",              "301–500"],
        "Kategori":   ["Baik",   "Sedang",   "Tidak Sehat (Sensitif)","Tidak Sehat", "Sangat Tidak Sehat",   "Berbahaya"],
        "Warna":      ["🟢 Hijau","🟡 Kuning","🟠 Oranye",            "🔴 Merah",    "🟣 Ungu",              "⬛ Merun"],
        "Deskripsi":  [
            "Kualitas udara memuaskan, polusi udara minim",
            "Kualitas dapat diterima, kelompok sensitif mungkin terpengaruh",
            "Kelompok sensitif terpengaruh, masyarakat umum tidak",
            "Semua orang mulai terpengaruh, kelompok sensitif lebih parah",
            "Darurat kesehatan, seluruh populasi terpengaruh serius",
            "Darurat serius, seluruh masyarakat terpengaruh parah",
        ],
    }
    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;font-size:11px;color:#30384a;
            font-family:'JetBrains Mono',monospace">
    AirSense Dashboard · Powered by LightGBM · Data: OpenAQ + Open-Meteo API<br>
    © 2024 Prediksi Kualitas Udara – Yogyakarta Early Warning System
</div>
""", unsafe_allow_html=True)
