# Bettermin Rapid Simulator (Streamlit)
# -------------------------------------------------------------
# MVP de simulación rápida para evaluar relaves y recomendar rutas
# de proceso (Preconcentración/TOMRA, BIOX, Lixiviación-SX-EW) y
# entregar métricas metalúrgicas + económicas por relave.
#
# Ejecuta con:  streamlit run app.py
# Requerimientos (requirements.txt):
#   streamlit
#   pandas
#   numpy
# -------------------------------------------------------------

import io
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Configuración general UI
# -----------------------------
st.set_page_config(page_title="Bettermin Rapid Simulator", page_icon="🧪", layout="wide")
st.title("🧪 Bettermin Rapid Simulator — Beta IA")
st.caption(
    "Simulador didáctico y rápido para priorizar relaves y estimar desempeño en Preconcentración ▶ BIOX ▶ Lixiviación/SX-EW, con resultados técnico–económicos."
)

# -----------------------------
# Utilidades
# -----------------------------
DEFAULT_COL_MAP = {
    "Nombre": ["nombre", "site", "yacimiento", "deposito", "depósito", "proyecto", "Deposito"],
    "Comuna": ["comuna", "municipio", "Comuna"],
    "Au_gpt": ["au_gpt", "au (g/t)", "oro_gpt", "ley_au", "Au(g/t)"],
    "Ag_gpt": ["ag_gpt", "ag (g/t)", "plata_gpt", "Ag(g/t)"],
    "Cu_pct": ["cu_pct", "cu %", "cobre_%", "ley_cu", "cu_%", "Cu(g/t)"],
    "As_ppm": ["as_ppm", "arsenico_ppm", "as (ppm)", "As(g/t)"],
    "S_tot_pct": ["s_tot_pct", "azufre_%", "s%", "stot%", "S Total(%)"],
    "S_sulf_pct": ["s_sulf_pct", "sulfuro_%", "ssulf%"],
    "Fe_pct": ["fe_pct", "Fe2O3(%)"],
    "Humedad_pct": ["humedad_pct", "moisture%"],
    "P80_um": ["p80_um", "p80 (um)", "granulometria_p80"],
    "Tonelaje_t": ["tonelaje_t", "toneladas", "t", "masa (t)"]
}

def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea columnas del usuario a nombres estándar si existen coincidencias aproximadas."""
    cols = {c: c for c in df.columns}
    lower = {c.lower().strip(): c for c in df.columns}
    for std, candidates in DEFAULT_COL_MAP.items():
        for cand in [std] + candidates:
            key = cand.lower()
            if key in lower:
                cols[lower[key]] = std
                break
    return df.rename(columns=cols)

@st.cache_data(show_spinner=False)
def example_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n = 20
    df = pd.DataFrame({
        "Nombre": [f"Relave_{i+1}" for i in range(n)],
        "Comuna": np.random.choice(["La Serena","Coquimbo","Punitaqui","Andacollo"], size=n),
        "Au_gpt": np.round(np.random.lognormal(mean=0.0, sigma=0.5, size=n), 2),
        "Ag_gpt": np.round(np.random.lognormal(mean=0.3, sigma=0.6, size=n), 1),
        "Cu_pct": np.round(np.random.beta(2,10,size=n)*2.0, 3),
        "As_ppm": np.random.randint(100, 30000, size=n),
        "S_tot_pct": np.round(np.random.uniform(0.3, 6.0, size=n),2),
        "S_sulf_pct": lambda x: x,
        "Fe_pct": np.round(np.random.uniform(2, 12, size=n),2),
        "Humedad_pct": np.round(np.random.uniform(5, 15, size=n),1),
        "P80_um": np.random.choice([75,106,150,212], size=n),
        "Tonelaje_t": np.random.randint(80_000, 1_200_000, size=n)
    })
    df["S_sulf_pct"] = np.round(df["S_tot_pct"]*np.random.uniform(0.5,0.95,size=len(df)),2)
    return df

# -----------------------------
# Parámetros globales (editables)
# -----------------------------
st.sidebar.header("⚙️ Parámetros Globales")
with st.sidebar:
    st.subheader("Precios de Mercado (editable)")
    price_au = st.number_input("Precio Oro (USD/oz)", 1000.0, 4000.0, 2300.0, 10.0)
    price_ag = st.number_input("Precio Plata (USD/oz)", 10.0, 80.0, 28.0, 0.5)
    price_cu = st.number_input("Precio Cobre (USD/lb)", 2.0, 8.0, 4.0, 0.05)

    st.subheader("OPEX de referencia (editable)")
    opex_precon_t = st.number_input("OPEX Preconcentración (USD/t relave)", 0.0, 50.0, 6.0, 0.5)
    opex_biox_tconc = st.number_input("OPEX BIOX (USD/t preconcentrado)", 200.0, 3000.0, 1790.0, 10.0)
    opex_sxew_t = st.number_input("OPEX Lix/SX-EW (USD/t relave)", 5.0, 100.0, 20.0, 1.0)

    st.subheader("Umbrales & Penalidades")
    tomra_cut = st.number_input("Umbral TOMRA (Au g/t equivalente)*", 0.1, 5.0, 0.8, 0.1, help="Proxy combinado que incluye Au y mineralogía sulfídica.")
    as_pen_ppm = st.number_input("Penalidad por As a partir de (ppm)", 1000, 100000, 12000, 500)
    as_pen_factor = st.slider("Factor penalidad recuperaciones por As alto", 0.5, 1.0, 0.86, 0.01)

# -----------------------------
# Carga de datos
# -----------------------------
st.header("1) Cargar y explorar relaves")

uploaded = st.file_uploader("Sube tu CSV/Excel con relaves inactivos/abandonados", type=["csv","xlsx"])

if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            import pandas as pd
            df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()
else:
    st.info("No subiste archivo aún. Usando dataset de ejemplo (sintético) para probar el flujo).")
    df_raw = example_dataset()

# Normaliza nombres de columnas
_df = auto_map_columns(df_raw.copy())

# Derivaciones específicas para tu CSV de Coquimbo
# - Convierte unidades y completa columnas faltantes desde fuentes equivalentes.
def derive_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Tonelaje
    if "Tonelaje_t" not in df.columns and "masa (t)" in df_raw.columns:
        df["Tonelaje_t"] = pd.to_numeric(df_raw["masa (t)"], errors="coerce")
    # Au/Ag g/t directos
    if "Au_gpt" not in df.columns and "Au(g/t)" in df_raw.columns:
        df["Au_gpt"] = pd.to_numeric(df_raw["Au(g/t)"], errors="coerce")
    if "Ag_gpt" not in df.columns and "Ag(g/t)" in df_raw.columns:
        df["Ag_gpt"] = pd.to_numeric(df_raw["Ag(g/t)"], errors="coerce")
    # Cu: g/t → % (1% = 10,000 g/t)
    if "Cu_pct" not in df.columns:
        if "Cu(g/t)" in df_raw.columns:
            cu_gpt = pd.to_numeric(df_raw["Cu(g/t)"], errors="coerce")
            df["Cu_pct"] = cu_gpt / 10000.0
    # As: g/t ≈ ppm
    if "As_ppm" not in df.columns and "As(g/t)" in df_raw.columns:
        df["As_ppm"] = pd.to_numeric(df_raw["As(g/t)"], errors="coerce")
    # Azufre total
    if "S_tot_pct" not in df.columns and "S Total(%)" in df_raw.columns:
        df["S_tot_pct"] = pd.to_numeric(df_raw["S Total(%)"], errors="coerce")
    # Azufre sulfurado (estimación conservadora si no hay dato directo)
    if "S_sulf_pct" not in df.columns and "S_tot_pct" in df.columns:
        df["S_sulf_pct"] = np.round(df["S_tot_pct"] * 0.8, 2)
    # Hierro: Fe2O3% → Fe% (factor 0.6994)
    if "Fe_pct" not in df.columns and "Fe2O3(%)" in df_raw.columns:
        fe2o3 = pd.to_numeric(df_raw["Fe2O3(%)"], errors="coerce")
        df["Fe_pct"] = fe2o3 * 0.6994
    # P80: si no hay, asumir 150 µm
    if "P80_um" not in df.columns:
        df["P80_um"] = 150
    # Nombre
    if "Nombre" not in df.columns and "Deposito" in df_raw.columns:
        df["Nombre"] = df_raw["Deposito"].astype(str)
    # Comuna
    if "Comuna" not in df.columns and "Comuna" in df_raw.columns:
        df["Comuna"] = df_raw["Comuna"].astype(str)
    return df

_df = derive_common_fields(_df)

# Validación mínima de columnas claves
REQUIRED = ["Nombre","Comuna","Au_gpt","Cu_pct","As_ppm","S_tot_pct","S_sulf_pct","P80_um","Tonelaje_t"]
missing = [c for c in REQUIRED if c not in _df.columns]
if missing:
    st.warning(f"Faltan columnas recomendadas: {missing}. El simulador funcionará con lo disponible, pero los resultados serán menos precisos.")

# Limpieza simple
def to_num(s):
    try:
        return pd.to_numeric(s)
    except Exception:
        return np.nan

for c in ["Au_gpt","Ag_gpt","Cu_pct","As_ppm","S_tot_pct","S_sulf_pct","Fe_pct","Humedad_pct","P80_um","Tonelaje_t"]:
    if c in _df.columns:
        _df[c] = _df[c].apply(to_num)

# Filtros
st.subheader("Filtros rápidos")
colf1, colf2, colf3 = st.columns(3)
with colf1:
    comunas = sorted(_df["Comuna"].dropna().unique().tolist()) if "Comuna" in _df.columns else []
    sel_comunas = st.multiselect("Comunas", comunas, default=comunas[:2] if comunas else [])
with colf2:
    min_au = st.number_input("Au mín (g/t)", 0.0, 20.0, 0.2, 0.1)
    max_as = st.number_input("As máx (ppm)", 0.0, 100000.0, 30000.0, 100.0)
with colf3:
    min_cu = st.number_input("Cu mín (%)", 0.0, 5.0, 0.1, 0.01)
    p80_target = st.selectbox("P80 objetivo (µm)", [75, 106, 150, 212], index=0)

mask = pd.Series([True]*len(_df))
if sel_comunas and "Comuna" in _df.columns:
    mask &= _df["Comuna"].isin(sel_comunas)
if "Au_gpt" in _df.columns:
    mask &= (_df["Au_gpt"].fillna(0) >= min_au)
if "As_ppm" in _df.columns:
    mask &= (_df["As_ppm"].fillna(0) <= max_as)
if "Cu_pct" in _df.columns:
    mask &= (_df["Cu_pct"].fillna(0) >= min_cu)

view = _df[mask].copy()

st.dataframe(view, use_container_width=True)

# -----------------------------
# Núcleo de simulación (reglas + modelos sencillos)
# -----------------------------
@dataclass
class SimResult:
    route: str
    mass_pull: float  # fracción de masa al concentrado en precon
    rec_au: float     # recuperación Au global (0-1)
    rec_cu: float     # recuperación Cu global (0-1)
    conc_upgrade_au: float  # factor de enriquecimiento Au
    penalty_msg: List[str]
    biox: Dict[str, float]
    cu: Dict[str, float]
    economics: Dict[str, float]
    suggestions: List[str]


def precon_model(row: pd.Series) -> Tuple[float, float, float, List[str]]:
    """Modelo simple de preconcentración tipo TOMRA/gravimétrica.
    Retorna (mass_pull, rec_au, upgrade_au, warnings)
    """
    au = row.get("Au_gpt", 0.0) or 0.0
    s_sulf = row.get("S_sulf_pct", 0.0) or 0.0
    fe = row.get("Fe_pct", 5.0) or 5.0
    as_ppm = row.get("As_ppm", 0.0) or 0.0
    p80 = row.get("P80_um", 150) or 150

    # proxy de respuesta a sensores (densidad/contraste):
    response = 0.6*min(1.0, au/(tomra_cut)) + 0.3*min(1.0, s_sulf/2.0) + 0.1*min(1.0, fe/8.0)
    mass_pull = np.clip(0.05 + 0.35*response, 0.05, 0.5)  # 5–50%

    # enriquecimiento Au ~ inverso del mass pull + efecto de p80
    upgrade_au = np.clip((1.0/mass_pull)* (1.0 + (150/p80 - 1.0)*0.2), 1.2, 12.0)
    rec_au = np.clip(0.6*response + 0.2*(150.0/p80) + 0.2, 0.25, 0.98)

    warns = []
    if as_ppm > as_pen_ppm:
        rec_au *= as_pen_factor
        warns.append(f"As alto ({int(as_ppm)} ppm): reducción de recuperaciones por factor {as_pen_factor}.")
    return mass_pull, rec_au, upgrade_au, warns


def biox_model(row: pd.Series, mass_pull: float, upgrade_au: float) -> Dict[str, float]:
    """Modelo sencillo de BIOX en función de S_sulf, As, DO/pH/tiempo target.
    Devuelve dict con TONS proxy y liberación de Au refractario."""
    s_sulf = row.get("S_sulf_pct", 1.0) or 1.0
    as_ppm = row.get("As_ppm", 0.0) or 0.0
    # Parámetros objetivo (ajustables a futuro):
    t_res_d = 5.5
    do_mgL = 7.0
    ph = 1.7
    redox_mV = 430

    # Proxy de tasa de oxidación (sigmoidal con saturación):
    load = min(1.0, s_sulf/4.0)
    toxicity = 1.0 if as_ppm < 8000 else (0.9 if as_ppm < 15000 else 0.8)
    tons = np.clip( (0.65*load + 0.35)*(t_res_d/6.0)*(do_mgL/7.0)*toxicity , 0.2, 1.0)

    # Liberación de Au refractario como función de TONS:
    liberation = np.clip(0.5 + 0.5*tons, 0.5, 0.98)  # 50–98%

    return {
        "t_res_d": t_res_d,
        "do_mgL": do_mgL,
        "ph": ph,
        "redox_mV": redox_mV,
        "TONS": tons,
        "liberation": liberation,
    }


def cu_leach_model(row: pd.Series) -> Dict[str, float]:
    """Modelo simple para lixiviación + SX-EW. Estima recuperación de Cu y ácido."""
    cu = row.get("Cu_pct", 0.0) or 0.0
    s_tot = row.get("S_tot_pct", 0.0) or 0.0
    as_ppm = row.get("As_ppm", 0.0) or 0.0

    # Recuperación base según tipo de relave (proxy):
    rec_cu = np.clip(0.35 + 0.1*(cu/1.0) + 0.15*(s_tot/3.0), 0.15, 0.85)
    if as_ppm > as_pen_ppm:
        rec_cu *= as_pen_factor

    # Consumo de ácido (proxy)
    acid_kgpt = np.clip(2.5 + 8.0*(s_tot/5.0), 2.0, 12.0)  # kg H2SO4/t

    return {"rec_cu": rec_cu, "acid_kgpt": acid_kgpt}


def economics(row: pd.Series, sim: Dict[str, float]) -> Dict[str, float]:
    """Calcula ingresos y margen por tonelada de relave tratado."""
    au_gpt = row.get("Au_gpt", 0.0) or 0.0
    ag_gpt = row.get("Ag_gpt", 0.0) or 0.0
    cu_pct = row.get("Cu_pct", 0.0) or 0.0

    mass_pull = sim["mass_pull"]
    rec_au = sim["rec_au"]
    rec_cu = sim["rec_cu"]
    upgrade_au = sim["upgrade_au"]

    # Oro y plata (asume venta como doré a partir de BIOX + refino)
    oz_au_per_t = au_gpt / 31.1035 / 1000.0 * 1e6  # g/t → oz/t (1 g/t = 1 g por t)
    oz_ag_per_t = (ag_gpt or 0.0) / 31.1035 / 1000.0 * 1e6

    rev_au = oz_au_per_t * rec_au * price_au
    rev_ag = oz_ag_per_t * 0.5 * price_ag  # sup. 50% rec Ag por ruta Au

    # Cobre por lixiviación (en paralelo o alternativa)
    lb_cu_per_t = cu_pct/100.0 * 2204.62  # lb/t
    rev_cu = lb_cu_per_t * rec_cu * price_cu

    # Costos (simplificados)
    cost_precon = opex_precon_t
    cost_biox = opex_biox_tconc * mass_pull  # por tonelada de preconcentrado
    cost_sxew = opex_sxew_t

    # Escoger mejor ruta económica: Au (precon+BIOX) vs Cu (SX-EW) vs Mixta
    margin_au = rev_au + rev_ag - (cost_precon + cost_biox)
    margin_cu = rev_cu - cost_sxew
    margin_mix = rev_au + rev_ag + rev_cu - (cost_precon + cost_biox + cost_sxew)

    best_margin = max(margin_au, margin_cu, margin_mix)
    if best_margin == margin_au:
        route = "Au (Precon + BIOX)"
    elif best_margin == margin_cu:
        route = "Cu (Lixiviación + SX-EW)"
    else:
        route = "Mixta"

    return {
        "rev_au_usdpt": rev_au,
        "rev_ag_usdpt": rev_ag,
        "rev_cu_usdpt": rev_cu,
        "cost_precon_usdpt": cost_precon,
        "cost_biox_usdpt": cost_biox,
        "cost_sxew_usdpt": cost_sxew,
        "margin_au_usdpt": margin_au,
        "margin_cu_usdpt": margin_cu,
        "margin_mix_usdpt": margin_mix,
        "best_route": route,
    }


def recommendations(row: pd.Series, sim: Dict[str, float], biox: Dict[str, float], cu: Dict[str, float], warns: List[str]) -> List[str]:
    recs = []
    au = row.get("Au_gpt", 0.0) or 0.0
    cu_pct = row.get("Cu_pct", 0.0) or 0.0
    as_ppm = row.get("As_ppm", 0.0) or 0.0
    s_sulf = row.get("S_sulf_pct", 0.0) or 0.0
    p80 = row.get("P80_um", 150) or 150

    if au >= tomra_cut:
        recs.append("Aplicar TOMRA/gravimetría: objetivo mass pull 15–30% con upgrade Au ≥3x.")
    else:
        recs.append("Au bajo umbral TOMRA: considerar gravimetría suave y aumentar capacidad para reducir costo/t.")

    if as_ppm > as_pen_ppm:
        recs.append("As elevado: incorporar dilución, lavado ácido previo o blending para bajar toxicidad al BIOX.")

    if s_sulf < 1.0:
        recs.append("Pobre en sulfuros: BIOX puede no justificar CAPEX/OPEX; priorizar ruta Cu si Cu% ≥ 0.3.")

    if p80 > 106:
        recs.append("Moler a P80 75–106 µm para mejorar exposición de Au refractario antes de BIOX.")

    if cu_pct >= 0.3:
        recs.append("Ruta Cu viable: planificar irrigación y control de ácido (2–10 kg/t) y solución rica para SX.")

    if biox["TONS"] < 0.6:
        recs.append("Aumentar tiempo de residencia BIOX a ≥6 días y DO ≥7 mg/L; ajustar pH 1.6–1.8.")

    recs.extend(warns)
    return recs


def simulate_row(row: pd.Series) -> SimResult:
    mass_pull, rec_au, upgrade_au, warns = precon_model(row)
    biox = biox_model(row, mass_pull, upgrade_au)
    cu_d = cu_leach_model(row)

    # Ruta Cu recuperación
    rec_cu = cu_d["rec_cu"]

    econ = economics(row, {
        "mass_pull": mass_pull,
        "rec_au": rec_au * biox["liberation"],  # aplica liberación BIOX
        "rec_cu": rec_cu,
        "upgrade_au": upgrade_au,
    })

    sugg = recommendations(row, {
        "mass_pull": mass_pull,
        "rec_au": rec_au,
        "upgrade_au": upgrade_au
    }, biox, cu_d, warns)

    return SimResult(
        route=econ["best_route"],
        mass_pull=mass_pull,
        rec_au=rec_au * biox["liberation"],
        rec_cu=rec_cu,
        conc_upgrade_au=upgrade_au,
        penalty_msg=warns,
        biox=biox,
        cu=cu_d,
        economics=econ,
        suggestions=sugg
    )

# -----------------------------
# Simulación en lote
# -----------------------------
st.header("2) Simular y priorizar")

if st.button("▶ Ejecutar simulación para relaves filtrados"):
    results = []
    for _, r in view.iterrows():
        sim = simulate_row(r)
        results.append({
            "Nombre": r.get("Nombre","?"),
            "Comuna": r.get("Comuna","?"),
            "Au_gpt": r.get("Au_gpt", np.nan),
            "Cu_pct": r.get("Cu_pct", np.nan),
            "As_ppm": r.get("As_ppm", np.nan),
            "MassPull_%": round(sim.mass_pull*100,1),
            "Rec_Au_%": round(sim.rec_au*100,1),
            "Rec_Cu_%": round(sim.rec_cu*100,1),
            "Upgrade_Au_x": round(sim.conc_upgrade_au,2),
            "TONS_BIOX": round(sim.biox["TONS"],2),
            "Ruta_Óptima": sim.route,
            "Margen_Au_USD/t": round(sim.economics["margin_au_usdpt"],2),
            "Margen_Cu_USD/t": round(sim.economics["margin_cu_usdpt"],2),
            "Margen_Mix_USD/t": round(sim.economics["margin_mix_usdpt"],2),
            "Ruta_Sugerida": sim.economics["best_route"],
            "Sugerencias": " | ".join(sim.suggestions)
        })
    res_df = pd.DataFrame(results)
    st.success("Simulación completada.")

    # Ranking por mejor margen por t
    res_df["Best_Margen_USD/t"] = res_df[["Margen_Au_USD/t","Margen_Cu_USD/t","Margen_Mix_USD/t"]].max(axis=1)
    res_df = res_df.sort_values("Best_Margen_USD/t", ascending=False)

    st.subheader("📊 Ranking de relaves (mejor margen por t)")
    st.dataframe(res_df, use_container_width=True)

    # Exportar
    buf = io.StringIO()
    res_df.to_csv(buf, index=False)
    st.download_button("💾 Descargar resultados CSV", buf.getvalue(), file_name="bettermin_simulacion_resultados.csv", mime="text/csv")

    # Detalle de un relave
    st.subheader("🔎 Detalle de un relave")
    sel = st.selectbox("Selecciona", res_df["Nombre"].tolist())
    row = view[view["Nombre"]==sel].iloc[0]
    sim = simulate_row(row)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mass Pull (%)", f"{sim.mass_pull*100:.1f}")
        st.metric("Upgrade Au (x)", f"{sim.conc_upgrade_au:.2f}")
    with c2:
        st.metric("Recuperación Au (%)", f"{sim.rec_au*100:.1f}")
        st.metric("TONS BIOX", f"{sim.biox['TONS']:.2f}")
    with c3:
        st.metric("Recuperación Cu (%)", f"{sim.rec_cu*100:.1f}")
        st.metric("Ruta Óptima", sim.economics["best_route"]) 

    st.markdown("**Sugerencias operacionales**")
    for s in sim.suggestions:
        st.write("• ", s)

else:
    st.info("Ajusta filtros y parámetros. Luego presiona **▶ Ejecutar simulación**.")

# -----------------------------
# 3) Pasaporte Digital (preview)
# -----------------------------
st.header("3) Pasaporte Digital — Preview")
st.caption("Identidad del lote para ‘feed-forward’ hacia BIOX-PILOT y control económico (Rentabilidad Neta por Hora).")

cols_for_passport = [c for c in ["Nombre","Comuna","Au_gpt","Ag_gpt","Cu_pct","As_ppm","S_tot_pct","S_sulf_pct","P80_um","Tonelaje_t"] if c in view.columns]
passport_df = view[cols_for_passport].copy()
if len(passport_df):
    st.dataframe(passport_df, use_container_width=True)
else:
    st.write("No hay columnas suficientes para desplegar el pasaporte digital.")

st.divider()
st.caption("Beta educativa. Modelos simplificados a refinar con datos reales de pruebas TOMRA, BIOX y lixiviación. Permite priorización rápida y definición de test de laboratorio.")
