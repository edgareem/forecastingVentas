from __future__ import annotations

from collections import deque
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


sns.set_theme(style="whitegrid", context="talk")

st.set_page_config(
    page_title="Forecasting Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(16, 185, 129, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(14, 116, 144, 0.10), transparent 24%),
            linear-gradient(180deg, #f7fbfc 0%, #eef6f7 100%);
    }
    .hero-card {
        padding: 1.2rem 1.4rem;
        border-radius: 1.2rem;
        background: linear-gradient(135deg, #062c30 0%, #0f766e 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(6, 44, 48, 0.20);
        margin-bottom: 1rem;
    }
    .hero-card h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
        line-height: 1.1;
    }
    .hero-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.95;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 118, 110, 0.10);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }
    .metric-card .label {
        color: #48606a;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .metric-card .value {
        color: #062c30;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-card .subvalue {
        color: #0f766e;
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_CANDIDATES = [
    Path("models/modelo_final.joblib"),
    Path("models/modelofinal.joblib"),
    Path("notebooks/models/modelofinal.joblib"),
]

DATA_CANDIDATES = [
    Path("data/processed/inferencia_df_transformado.csv"),
    Path("notebooks/data/processed/inferencia_df_transformado.csv"),
]


def resolve_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró ninguno de los archivos esperados:\n"
        + "\n".join(str(path) for path in candidates)
    )


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[object, Path]:
    model_path = resolve_existing_path(MODEL_CANDIDATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
    return model, model_path


@st.cache_data(show_spinner=False)
def load_inference_data() -> tuple[pd.DataFrame, Path]:
    data_path = resolve_existing_path(DATA_CANDIDATES)
    df = pd.read_csv(data_path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df, data_path


def infer_lag_columns(columns: list[str]) -> list[str]:
    lag_cols = [col for col in columns if col.startswith("lag_")]
    return sorted(lag_cols, key=lambda name: int(name.split("_")[-1]))


def infer_ma_column(columns: list[str]) -> str | None:
    for candidate in ["media_movil_7_dias", "unidades_vendidas_ma7"]:
        if candidate in columns:
            return candidate
    return None


def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned[feature_names].copy()


def apply_scenario(
    df: pd.DataFrame,
    own_price_pct: float,
    competitor_price_pct: float,
    target_products: list[str],
    stars_only: bool,
) -> pd.DataFrame:
    scenario_df = df.copy()

    if target_products:
        mask = scenario_df["producto_id"].isin(target_products)
    else:
        mask = pd.Series(True, index=scenario_df.index)

    if stars_only and "es_estrella" in scenario_df.columns:
        mask &= scenario_df["es_estrella"].astype(bool)

    own_factor = 1 + own_price_pct / 100.0
    competitor_factor = 1 + competitor_price_pct / 100.0

    scenario_df.loc[mask, "precio_venta"] = scenario_df.loc[mask, "precio_venta"] * own_factor
    if "precio_base" in scenario_df.columns:
        scenario_df.loc[mask, "descuento_porcentaje"] = (
            (scenario_df.loc[mask, "precio_venta"] - scenario_df.loc[mask, "precio_base"])
            / scenario_df.loc[mask, "precio_base"]
        ) * 100

    if "precio_competencia" in scenario_df.columns:
        scenario_df.loc[mask, "precio_competencia"] = (
            scenario_df.loc[mask, "precio_competencia"] * competitor_factor
        )

    if {"precio_venta", "precio_competencia"}.issubset(scenario_df.columns):
        scenario_df.loc[mask, "ratio_precio"] = (
            scenario_df.loc[mask, "precio_venta"] / scenario_df.loc[mask, "precio_competencia"]
        )

    return scenario_df


def recursive_predict(
    df: pd.DataFrame,
    model,
    feature_names: list[str],
    lag_cols: list[str],
    ma_col: str | None,
) -> pd.DataFrame:
    predictions = []

    for producto_id, product_df in df.groupby("producto_id", sort=True):
        product_df = product_df.sort_values("fecha").copy()

        history_seed = []
        if lag_cols:
            first_row = product_df.iloc[0]
            history_seed = [first_row.get(col, np.nan) for col in reversed(lag_cols)]
        history = deque(history_seed, maxlen=max(len(lag_cols), 7, 1))

        for idx in product_df.index:
            row = product_df.loc[idx].copy()

            if lag_cols:
                current_history = list(history)
                for lag_number, col in enumerate(lag_cols, start=1):
                    if len(current_history) >= lag_number:
                        row[col] = current_history[-lag_number]
                    else:
                        row[col] = np.nan

            if ma_col:
                recent_values = [value for value in list(history)[-7:] if pd.notna(value)]
                row[ma_col] = float(np.mean(recent_values)) if recent_values else np.nan

            feature_row = align_features(pd.DataFrame([row]), feature_names)
            prediction = float(model.predict(feature_row)[0])

            row["prediccion_unidades_vendidas"] = prediction
            predictions.append(row)

            history.append(prediction)

    prediction_df = pd.DataFrame(predictions).sort_values(["fecha", "producto_id"]).reset_index(drop=True)
    prediction_df["prediccion_unidades_vendidas"] = prediction_df["prediccion_unidades_vendidas"].clip(lower=0)
    return prediction_df


def metric_card(label: str, value: str, subvalue: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="subvalue">{subvalue}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


model, model_path = load_model()
base_df, data_path = load_inference_data()

feature_names = list(model.feature_names_in_)
lag_columns = infer_lag_columns(feature_names)
moving_average_column = infer_ma_column(feature_names)

product_options = sorted(base_df["producto_id"].unique().tolist())

with st.sidebar:
    st.title("Simulación")
    st.caption("Ajusta el escenario y recalcula las predicciones recursivas de noviembre 2025.")

    own_price_pct = st.slider(
        "Variación de precio propio (%)",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        help="Ajusta `precio_venta` y recalcula `descuento_porcentaje` y `ratio_precio`.",
    )
    competitor_price_pct = st.slider(
        "Variación de precio competencia (%)",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        help="Ajusta `precio_competencia` y recalcula `ratio_precio`.",
    )
    stars_only = st.checkbox(
        "Aplicar ajustes solo a productos estrella",
        value=False,
    )
    selected_products = st.multiselect(
        "Productos incluidos en el escenario",
        options=product_options,
        default=[],
        help="Si no eliges ninguno, el escenario se aplica a todos los productos.",
    )
    selected_product_chart = st.selectbox(
        "Producto para detalle diario",
        options=sorted(base_df["nombre"].unique().tolist()),
    )

    st.markdown("---")
    st.caption(f"Modelo cargado desde: `{model_path}`")
    st.caption(f"Datos cargados desde: `{data_path}`")
    st.caption(f"Variables del modelo: `{len(feature_names)}`")


scenario_df = apply_scenario(
    df=base_df,
    own_price_pct=own_price_pct,
    competitor_price_pct=competitor_price_pct,
    target_products=selected_products,
    stars_only=stars_only,
)

prediction_df = recursive_predict(
    df=scenario_df,
    model=model,
    feature_names=feature_names,
    lag_cols=lag_columns,
    ma_col=moving_average_column,
)


summary_by_product = (
    prediction_df.groupby(["producto_id", "nombre"], as_index=False)
    .agg(
        prediccion_total=("prediccion_unidades_vendidas", "sum"),
        precio_medio=("precio_venta", "mean"),
        estrella=("es_estrella", "max"),
    )
    .sort_values("prediccion_total", ascending=False)
)

summary_by_day = (
    prediction_df.groupby("fecha", as_index=False)["prediccion_unidades_vendidas"]
    .sum()
    .rename(columns={"prediccion_unidades_vendidas": "prediccion_total_dia"})
)

selected_product_df = prediction_df[prediction_df["nombre"] == selected_product_chart].copy()
selected_product_total = (
    selected_product_df["prediccion_unidades_vendidas"].sum() if not selected_product_df.empty else 0
)
top_product_name = summary_by_product.iloc[0]["nombre"] if not summary_by_product.empty else "-"
top_product_value = summary_by_product.iloc[0]["prediccion_total"] if not summary_by_product.empty else 0
bf_mask = prediction_df["es_black_friday"].astype(bool) if "es_black_friday" in prediction_df.columns else pd.Series(False, index=prediction_df.index)
black_friday_total = prediction_df.loc[bf_mask, "prediccion_unidades_vendidas"].sum()


st.markdown(
    """
    <div class="hero-card">
        <h1>Predicción de ventas · Noviembre 2025</h1>
        <p>
            Simulación recursiva día a día con actualización automática de lags y media móvil
            para los 24 productos del catálogo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card(
        "Unidades previstas",
        f"{prediction_df['prediccion_unidades_vendidas'].sum():,.0f}",
        "Total proyectado para noviembre",
    )
with col2:
    metric_card(
        "Producto líder",
        top_product_name,
        f"{top_product_value:,.0f} unidades previstas",
    )
with col3:
    metric_card(
        "Black Friday",
        f"{black_friday_total:,.0f}",
        "Unidades previstas ese día",
    )
with col4:
    metric_card(
        "Producto seleccionado",
        selected_product_chart,
        f"{selected_product_total:,.0f} unidades previstas",
    )

st.markdown("### Dashboard")

left_col, right_col = st.columns([1.25, 1])

with left_col:
    fig_day, ax_day = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=summary_by_day,
        x="fecha",
        y="prediccion_total_dia",
        marker="o",
        color="#0f766e",
        linewidth=2.5,
        ax=ax_day,
    )
    ax_day.set_title("Predicción total diaria de noviembre 2025")
    ax_day.set_xlabel("Fecha")
    ax_day.set_ylabel("Unidades previstas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_day, use_container_width=True)

with right_col:
    top_bar_df = summary_by_product.head(10).sort_values("prediccion_total", ascending=True)
    fig_top, ax_top = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=top_bar_df,
        x="prediccion_total",
        y="nombre",
        hue="nombre",
        dodge=False,
        palette="crest",
        legend=False,
        ax=ax_top,
    )
    ax_top.set_title("Top 10 productos por predicción acumulada")
    ax_top.set_xlabel("Unidades previstas")
    ax_top.set_ylabel("Producto")
    plt.tight_layout()
    st.pyplot(fig_top, use_container_width=True)


bottom_left, bottom_right = st.columns([1.15, 1.1])

with bottom_left:
    fig_product, ax_product = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=selected_product_df,
        x="fecha",
        y="prediccion_unidades_vendidas",
        marker="o",
        color="#ea580c",
        linewidth=2.5,
        ax=ax_product,
    )
    ax_product.set_title(f"Evolución diaria · {selected_product_chart}")
    ax_product.set_xlabel("Fecha")
    ax_product.set_ylabel("Unidades previstas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_product, use_container_width=True)

with bottom_right:
    heatmap_df = prediction_df.pivot_table(
        index="nombre",
        columns=prediction_df["fecha"].dt.day,
        values="prediccion_unidades_vendidas",
        aggfunc="sum",
    )
    fig_heat, ax_heat = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        heatmap_df,
        cmap="YlGnBu",
        linewidths=0.3,
        cbar_kws={"label": "Unidades previstas"},
        ax=ax_heat,
    )
    ax_heat.set_title("Mapa de calor diario por producto")
    ax_heat.set_xlabel("Día de noviembre")
    ax_heat.set_ylabel("Producto")
    plt.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)


st.markdown("### Tabla de resultados")
display_df = prediction_df[
    [
        "fecha",
        "producto_id",
        "nombre",
        "precio_venta",
        "descuento_porcentaje",
        "precio_competencia",
        "ratio_precio",
        "prediccion_unidades_vendidas",
    ]
].copy()
display_df["fecha"] = display_df["fecha"].dt.strftime("%Y-%m-%d")
display_df = display_df.sort_values(["fecha", "producto_id"])

st.dataframe(display_df, use_container_width=True, hide_index=True)

csv_bytes = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Descargar predicciones de noviembre 2025",
    data=csv_bytes,
    file_name="predicciones_noviembre_2025.csv",
    mime="text/csv",
)
