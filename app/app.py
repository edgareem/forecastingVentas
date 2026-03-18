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


PRIMARY = "#667eea"
SECONDARY = "#764ba2"
ACCENT = "#ef4444"
BACKGROUND = "#f5f7ff"
BLACK_FRIDAY_DAY = 28
MODEL_CANDIDATES = [
    Path("models/modelo_final.joblib"),
    Path("models/modelofinal.joblib"),
    Path("notebooks/models/modelofinal.joblib"),
]
DATA_CANDIDATES = [
    Path("data/processed/inferencia_df_transformado.csv"),
    Path("notebooks/data/processed/inferencia_df_transformado.csv"),
]
COMPETITION_SCENARIOS = {
    "Actual (0%)": 0.0,
    "Competencia -5%": -5.0,
    "Competencia +5%": 5.0,
}


sns.set_theme(style="whitegrid", context="talk")


st.set_page_config(
    page_title="Simulador de ventas - Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(102, 126, 234, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(118, 75, 162, 0.16), transparent 25%),
            linear-gradient(180deg, {BACKGROUND} 0%, #ffffff 100%);
    }}
    section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.14) 0%, rgba(118, 75, 162, 0.10) 100%);
    }}
    .hero-card {{
        padding: 1.4rem 1.5rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        box-shadow: 0 20px 45px rgba(102, 126, 234, 0.28);
        margin-bottom: 1rem;
    }}
    .hero-card h1 {{
        margin: 0;
        font-size: 2rem;
        line-height: 1.1;
    }}
    .hero-card p {{
        margin: 0.55rem 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
    }}
    .scenario-card {{
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(102, 126, 234, 0.18);
        box-shadow: 0 10px 24px rgba(79, 70, 229, 0.08);
    }}
    .scenario-card .title {{
        color: {SECONDARY};
        font-weight: 700;
        margin-bottom: 0.35rem;
    }}
    .scenario-card .big {{
        font-size: 1.4rem;
        color: #1f2937;
        font-weight: 700;
        line-height: 1.2;
    }}
    .scenario-card .small {{
        color: #475569;
        margin-top: 0.25rem;
    }}
    div[data-testid="stButton"] > button {{
        width: 100%;
        min-height: 3rem;
        border-radius: 0.9rem;
        border: none;
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        font-weight: 700;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def resolve_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No se encontro ninguno de los archivos esperados:\n" + "\n".join(str(path) for path in candidates)
    )


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[object, Path]:
    model_path = resolve_existing_path(MODEL_CANDIDATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
    return model, model_path


@st.cache_data(show_spinner=False)
def load_base_data() -> tuple[pd.DataFrame, Path]:
    data_path = resolve_existing_path(DATA_CANDIDATES)
    df = pd.read_csv(data_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df, data_path


def infer_lag_columns(feature_names: list[str]) -> list[str]:
    lag_cols = [col for col in feature_names if col.startswith("lag_")]
    return sorted(lag_cols, key=lambda name: int(name.split("_")[-1]))


def infer_moving_average_column(feature_names: list[str]) -> str | None:
    for candidate in ["media_movil_7_dias", "unidades_vendidas_ma7"]:
        if candidate in feature_names:
            return candidate
    return None


def validate_base_columns(df: pd.DataFrame, lag_cols: list[str], ma_col: str | None) -> list[str]:
    required = [
        "fecha",
        "producto_id",
        "nombre",
        "precio_base",
        "precio_venta",
        "precio_competencia",
        "ratio_precio",
        "descuento_porcentaje",
    ] + lag_cols
    if ma_col:
        required.append(ma_col)
    return [col for col in required if col not in df.columns]


def align_to_model(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned[feature_names].copy()


def format_currency(value: float) -> str:
    return f"EUR {value:,.2f}"


def format_units(value: float) -> str:
    return f"{value:,.0f}"


def apply_simulation_controls(product_df: pd.DataFrame, discount_pct: float, competitor_pct: float) -> pd.DataFrame:
    scenario_df = product_df.sort_values("fecha").copy()
    own_factor = 1 - discount_pct / 100.0
    competitor_factor = 1 + competitor_pct / 100.0

    scenario_df["precio_venta"] = scenario_df["precio_base"] * own_factor
    scenario_df["descuento_porcentaje"] = (
        (scenario_df["precio_venta"] - scenario_df["precio_base"]) / scenario_df["precio_base"]
    ) * 100
    scenario_df["descuento_aplicado"] = (
        (scenario_df["precio_base"] - scenario_df["precio_venta"]) / scenario_df["precio_base"]
    ) * 100

    competitor_cols = [col for col in ["Amazon", "Decathlon", "Deporvillage"] if col in scenario_df.columns]
    if competitor_cols:
        scenario_df[competitor_cols] = scenario_df[competitor_cols] * competitor_factor
        scenario_df["precio_competencia"] = scenario_df[competitor_cols].mean(axis=1)
    else:
        scenario_df["precio_competencia"] = scenario_df["precio_competencia"] * competitor_factor

    scenario_df["ratio_precio"] = scenario_df["precio_venta"] / scenario_df["precio_competencia"]
    return scenario_df


def recursive_predict_product(
    product_df: pd.DataFrame,
    model,
    feature_names: list[str],
    lag_cols: list[str],
    ma_col: str | None,
) -> pd.DataFrame:
    scenario_df = product_df.sort_values("fecha").reset_index(drop=True).copy()
    if scenario_df.empty:
        return scenario_df

    max_history = max(len(lag_cols), 7, 1)
    history = deque(maxlen=max_history)

    first_row = scenario_df.iloc[0]
    seed_values = []
    for col in reversed(lag_cols):
        value = first_row.get(col, np.nan)
        if pd.notna(value):
            seed_values.append(float(value))
    history.extend(seed_values)

    predicted_values: list[float] = []
    result_rows = []

    for idx in range(len(scenario_df)):
        row = scenario_df.iloc[idx].copy()

        if idx > 0:
            current_history = list(history)
            for lag_number, lag_col in enumerate(lag_cols, start=1):
                row[lag_col] = current_history[-lag_number] if len(current_history) >= lag_number else np.nan
            if ma_col:
                recent_predictions = predicted_values[-7:]
                row[ma_col] = float(np.mean(recent_predictions)) if recent_predictions else row.get(ma_col, np.nan)

        feature_row = align_to_model(pd.DataFrame([row]), feature_names)
        prediction = float(model.predict(feature_row)[0])
        prediction = max(prediction, 0.0)

        predicted_values.append(prediction)
        history.append(prediction)

        row["prediccion_unidades_vendidas"] = prediction
        row["ingresos_proyectados"] = prediction * float(row["precio_venta"])
        row["dia_mes"] = int(pd.to_datetime(row["fecha"]).day)
        row["dia_semana_mostrar"] = row.get("dia_semana") or pd.to_datetime(row["fecha"]).day_name()
        row["evento"] = "🛍️ Black Friday" if bool(row.get("es_black_friday", False)) else ""
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    return result_df.sort_values("fecha").reset_index(drop=True)


def simulate_scenario(
    base_df: pd.DataFrame,
    product_name: str,
    discount_pct: float,
    competitor_pct: float,
    model,
    feature_names: list[str],
    lag_cols: list[str],
    ma_col: str | None,
) -> pd.DataFrame:
    product_df = base_df[base_df["nombre"] == product_name].copy()
    product_df = product_df[pd.to_datetime(product_df["fecha"]).dt.month == 11].copy()
    scenario_df = apply_simulation_controls(product_df, discount_pct, competitor_pct)
    return recursive_predict_product(scenario_df, model, feature_names, lag_cols, ma_col)


def summarize_predictions(prediction_df: pd.DataFrame) -> dict[str, float]:
    return {
        "unidades": float(prediction_df["prediccion_unidades_vendidas"].sum()),
        "ingresos": float(prediction_df["ingresos_proyectados"].sum()),
        "precio_medio": float(prediction_df["precio_venta"].mean()),
        "descuento_medio": float(prediction_df["descuento_aplicado"].mean()),
    }


def render_scenario_card(title: str, units: float, revenue: float) -> None:
    st.markdown(
        f"""
        <div class="scenario-card">
            <div class="title">{title}</div>
            <div class="big">{format_units(units)} uds</div>
            <div class="small">{format_currency(revenue)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_daily_chart(prediction_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    chart_df = prediction_df.copy()
    sns.lineplot(
        data=chart_df,
        x="dia_mes",
        y="prediccion_unidades_vendidas",
        marker="o",
        linewidth=2.8,
        color=PRIMARY,
        ax=ax,
    )

    bf_df = chart_df[chart_df["es_black_friday"].astype(bool)].copy()
    if not bf_df.empty:
        bf_row = bf_df.iloc[0]
        ax.axvline(x=int(bf_row["dia_mes"]), color=ACCENT, linestyle="--", linewidth=2)
        ax.scatter(
            [int(bf_row["dia_mes"])],
            [float(bf_row["prediccion_unidades_vendidas"])],
            color=ACCENT,
            s=110,
            zorder=5,
        )
        ax.annotate(
            "Black Friday",
            xy=(int(bf_row["dia_mes"]), float(bf_row["prediccion_unidades_vendidas"])),
            xytext=(int(bf_row["dia_mes"]) - 6, float(bf_row["prediccion_unidades_vendidas"]) * 1.06),
            arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1.5},
            color=ACCENT,
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("Prediccion diaria de unidades para noviembre 2025")
    ax.set_xlabel("Dia de noviembre")
    ax.set_ylabel("Unidades vendidas predichas")
    ax.set_xticks(range(1, 31, 2))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def build_detail_table(prediction_df: pd.DataFrame) -> pd.DataFrame:
    table_df = prediction_df[
        [
            "fecha",
            "dia_semana_mostrar",
            "precio_venta",
            "precio_competencia",
            "descuento_aplicado",
            "prediccion_unidades_vendidas",
            "ingresos_proyectados",
            "evento",
        ]
    ].copy()
    table_df["fecha"] = pd.to_datetime(table_df["fecha"]).dt.strftime("%Y-%m-%d")
    table_df["precio_venta"] = table_df["precio_venta"].map(lambda x: f"{x:,.2f}")
    table_df["precio_competencia"] = table_df["precio_competencia"].map(lambda x: f"{x:,.2f}")
    table_df["descuento_aplicado"] = table_df["descuento_aplicado"].map(lambda x: f"{x:,.1f}%")
    table_df["prediccion_unidades_vendidas"] = table_df["prediccion_unidades_vendidas"].map(lambda x: f"{x:,.0f}")
    table_df["ingresos_proyectados"] = table_df["ingresos_proyectados"].map(lambda x: f"EUR {x:,.2f}")
    table_df = table_df.rename(
        columns={
            "dia_semana_mostrar": "dia_semana",
            "precio_venta": "precio_venta",
            "precio_competencia": "precio_competencia",
            "descuento_aplicado": "descuento_aplicado",
            "prediccion_unidades_vendidas": "unidades_predichas",
            "ingresos_proyectados": "ingresos_proyectados",
        }
    )
    return table_df


def main() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>📈 Simulacion de ventas - Noviembre 2025</h1>
            <p>Prediccion recursiva dia a dia con actualizacion automatica de lags, media movil y escenarios de precio.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, model_path = load_model()
        base_df, data_path = load_base_data()
    except Exception as exc:
        st.error(f"No se pudo cargar la app de forecasting: {exc}")
        st.stop()

    feature_names = list(model.feature_names_in_)
    lag_cols = infer_lag_columns(feature_names)
    ma_col = infer_moving_average_column(feature_names)
    missing_columns = validate_base_columns(base_df, lag_cols, ma_col)
    if missing_columns:
        st.error("Faltan columnas necesarias en el dataframe de inferencia: " + ", ".join(missing_columns))
        st.stop()

    product_names = sorted(base_df["nombre"].dropna().unique().tolist())
    if not product_names:
        st.error("No hay productos disponibles en el archivo de inferencia.")
        st.stop()

    with st.sidebar:
        st.title("Controles de Simulacion")
        st.caption("Selecciona un producto, ajusta el descuento y compara el efecto de la competencia.")

        selected_product = st.selectbox("Producto", options=product_names)
        discount_pct = st.slider(
            "Ajuste de descuento (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
        )
        competition_label = st.radio(
            "Escenario de competencia",
            options=list(COMPETITION_SCENARIOS.keys()),
        )
        simulate_clicked = st.button("Simular Ventas")

        st.markdown("---")
        st.caption(f"Modelo: `{model_path}`")
        st.caption(f"Datos: `{data_path}`")
        st.caption(f"Variables del modelo: `{len(feature_names)}`")

    current_config = {
        "producto": selected_product,
        "descuento": discount_pct,
        "competencia": competition_label,
    }
    previous_config = st.session_state.get("sim_config")
    should_run = simulate_clicked or "sim_results" not in st.session_state

    if should_run:
        with st.spinner("Calculando predicciones recursivas para noviembre 2025..."):
            selected_competition_pct = COMPETITION_SCENARIOS[competition_label]
            selected_result = simulate_scenario(
                base_df,
                selected_product,
                discount_pct,
                selected_competition_pct,
                model,
                feature_names,
                lag_cols,
                ma_col,
            )
            comparison_results = {}
            for label, pct in COMPETITION_SCENARIOS.items():
                scenario_result = simulate_scenario(
                    base_df,
                    selected_product,
                    discount_pct,
                    pct,
                    model,
                    feature_names,
                    lag_cols,
                    ma_col,
                )
                comparison_results[label] = summarize_predictions(scenario_result)

        st.session_state["sim_results"] = {
            "prediction_df": selected_result,
            "comparison": comparison_results,
        }
        st.session_state["sim_config"] = current_config
    elif previous_config != current_config:
        st.info("Has cambiado los controles. Pulsa `Simular Ventas` para actualizar el dashboard.")

    results = st.session_state.get("sim_results")
    if not results:
        st.warning("Todavia no hay resultados para mostrar.")
        st.stop()

    prediction_df = results["prediction_df"].copy()
    comparison = results["comparison"]
    if prediction_df.empty:
        st.warning("No hay datos de noviembre 2025 para el producto seleccionado.")
        st.stop()

    summary = summarize_predictions(prediction_df)

    st.markdown(f"## Dashboard de simulacion - {selected_product}")
    st.caption(
        "El dia 1 usa los lags ya preparados en el archivo. A partir del dia 2, la app actualiza lag_1 a lag_7 y la media movil con las predicciones previas."
    )

    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Unidades totales proyectadas", format_units(summary["unidades"]))
    kpi_2.metric("Ingresos proyectados", format_currency(summary["ingresos"]))
    kpi_3.metric("Precio promedio de venta", format_currency(summary["precio_medio"]))
    kpi_4.metric("Descuento promedio", f"{summary['descuento_medio']:.1f}%")

    st.divider()
    st.markdown("### Prediccion diaria")
    daily_chart = build_daily_chart(prediction_df)
    st.pyplot(daily_chart, use_container_width=True)

    st.divider()
    st.markdown("### Tabla detallada de noviembre")
    detail_table = build_detail_table(prediction_df)
    st.dataframe(detail_table, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Comparativa de escenarios de competencia")
    cmp_1, cmp_2, cmp_3 = st.columns(3)
    with cmp_1:
        render_scenario_card(
            "Actual (0%)",
            comparison["Actual (0%)"]["unidades"],
            comparison["Actual (0%)"]["ingresos"],
        )
    with cmp_2:
        render_scenario_card(
            "Competencia -5%",
            comparison["Competencia -5%"]["unidades"],
            comparison["Competencia -5%"]["ingresos"],
        )
    with cmp_3:
        render_scenario_card(
            "Competencia +5%",
            comparison["Competencia +5%"]["unidades"],
            comparison["Competencia +5%"]["ingresos"],
        )

    st.divider()
    st.success("Simulacion completada correctamente. Puedes ajustar los controles y volver a lanzar la prediccion.")


if __name__ == "__main__":
    main()
