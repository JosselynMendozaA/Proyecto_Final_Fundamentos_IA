import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib  # Carga ligera en RAM
import gdown   # Para descargar desde Google Drive
from openai import OpenAI

# 1. Configuración de la página
st.set_page_config(page_title="Walmart Sales Predictor and AI Advisor", layout="wide")

st.title("Walmart Sales Forecasting and Executive AI Advisor")
st.markdown("Plataforma interactiva para proyección de ventas e impacto de promociones.")

# 2. Cargar recursos optimizados desde Google Drive
FILE_ID = '1tshTEVLgK-qxPkp7jBeOu652Um_5Nrll'
MODEL_PATH = 'walmart_best_model_compressed.pkl'

@st.cache_resource
def load_resources():
    # Descarga el modelo solo si no existe localmente en el servidor
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    return joblib.load(MODEL_PATH)

model = load_resources()

# Función auxiliar para realizar predicciones directamente con el modelo cargado
def predict_model_direct(model, data):
    data_copy = data.copy()
    data_copy['prediction_label'] = model.predict(data_copy)
    return data_copy

# 3. Barra Lateral: Parámetros del Escenario
st.sidebar.header("Configuracion del Escenario")
store = st.sidebar.number_input("Numero de Tienda (1-45)", 1, 45, 1)
dept = st.sidebar.number_input("Departamento (1-99)", 1, 99, 1)

store_type = st.sidebar.selectbox("Tipo de Tienda", ["Tipo A (Grande)", "Tipo B (Mediana)", "Tipo C (Pequeña)"])
type_encoded = {"Tipo A (Grande)": 0, "Tipo B (Mediana)": 1, "Tipo C (Pequeña)": 2}[store_type]
store_size = {"Tipo A (Grande)": 180000, "Tipo B (Mediana)": 100000, "Tipo C (Pequeña)": 40000}[store_type]

selected_year = st.sidebar.selectbox("Año a Simular", [2024, 2025, 2026])

# SECCIÓN: Clima y Macroeconomía
st.sidebar.markdown("---")
with st.sidebar.expander("🌡️ Clima y Entorno Economico", expanded=True):
    temp_val = st.slider("Temperatura Promedio (°F)", 20.0, 100.0, 65.0)
    fuel_val = st.slider("Precio Gasolina ($/Galon)", 2.5, 5.0, 3.5)
    cpi_val = st.slider("Indice CPI (Inflacion)", 120.0, 230.0, 220.0)
    unemp_val = st.slider("Tasa de Desempleo (%)", 3.0, 15.0, 6.0)

# Presupuesto de Promociones
st.sidebar.markdown("---")
with st.sidebar.expander("Presupuesto de Promociones (MarkDowns $)", expanded=True):
    m1 = st.slider("MarkDown 1 (Publicidad General)", 0, 10000, 1500)
    m2 = st.slider("MarkDown 2 (Post-Festivo)", 0, 10000, 0)
    m3 = st.slider("MarkDown 3 (Festividades/Black Friday)", 0, 10000, 500)
    m4 = st.slider("MarkDown 4 (Volantes/Catalogos)", 0, 10000, 0)
    m5 = st.slider("MarkDown 5 (Fidelizacion)", 0, 10000, 0)

# 4. Generación de Datos para las 52 Semanas
weeks = list(range(1, 53))
future_data = []

for w in weeks:
    is_holiday = 1 if w in [6, 36, 47, 51] else 0
    month = int(np.ceil(w / 4.33))
    month = min(month, 12)

    future_data.append({
        'Store': store, 'Dept': dept, 'Size': store_size, 'Type_Encoded': type_encoded,
        'IsHoliday': is_holiday, 'Temperature': temp_val, 'Fuel_Price': fuel_val,
        'CPI': cpi_val, 'Unemployment': unemp_val, 'Month': month, 'Week': w, 'Year': selected_year,
        'MarkDown1': m1, 'MarkDown2': m2, 'MarkDown3': m3, 'MarkDown4': m4, 'MarkDown5': m5
    })

df_future = pd.DataFrame(future_data)
predictions = predict_model_direct(model, df_future)
df_future['Ventas_Proyectadas'] = predictions['prediction_label']

# 5. Tarjetas de Métricas (KPIs)
col_m1, col_m2, col_m3 = st.columns(3)
total_annual = df_future['Ventas_Proyectadas'].sum()
peak_week = df_future.loc[df_future['Ventas_Proyectadas'].idxmax()]['Week']
peak_sales = df_future['Ventas_Proyectadas'].max()

col_m1.metric("Proyeccion Total Anual", f"${total_annual:,.2f} USD")
col_m2.metric("Semana de Mayor Venta", f"Semana {int(peak_week)}")
col_m3.metric("Venta Maxima Estimada", f"${peak_sales:,.2f} USD")

st.markdown("---")

# 6. Panel de Visualizaciones Analíticas
st.header("Analisis Visual de Proyecciones")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader(f"Serie Temporal Semanal ({selected_year})")
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    sns.lineplot(data=df_future, x='Week', y='Ventas_Proyectadas', ax=ax1, color='#1f77b4', marker='o')
    ax1.set_xlabel("Semana del Año")
    ax1.set_ylabel("Ventas ($USD)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig1)

with col_g2:
    st.subheader("Impacto: Semanas Normales vs. Festivas")
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    df_future['Festivo_Texto'] = df_future['IsHoliday'].map({0: 'Semana Normal', 1: 'Semana Festiva'})
    sns.barplot(data=df_future, x='Festivo_Texto', y='Ventas_Proyectadas', ax=ax2,
                hue='Festivo_Texto', palette='Set2', legend=False)
    ax2.set_xlabel("")
    ax2.set_ylabel("Ventas Promedio ($USD)")
    st.pyplot(fig2)

# 7. Análisis de Inversión y Sensibilidad Promocional
st.header("Analisis de Inversion y Sensibilidad Promocional")
col_md1, col_md2 = st.columns(2)

with col_md1:
    st.subheader("Distribucion del Presupuesto (MarkDowns)")
    markdown_data = pd.DataFrame({
        'Campañas': ['MD1 (General)', 'MD2 (Post-Festivo)', 'MD3 (Festivo)', 'MD4 (Catalogos)', 'MD5 (Fidelidad)'],
        'Inversion': [m1, m2, m3, m4, m5]
    })
    
    fig_md, ax_md = plt.subplots(figsize=(6, 3.5))
    sns.barplot(data=markdown_data, x='Campañas', y='Inversion', ax=ax_md, palette='Blues_r')
    ax_md.set_xlabel("")
    ax_md.set_ylabel("Presupuesto ($USD)")
    plt.xticks(rotation=25)
    st.pyplot(fig_md)

with col_md2:
    st.subheader("Curva de Sensibilidad MarkDown 1")
    budget_range = np.linspace(0, 10000, 10)
    simulated_sales = []

    for b in budget_range:
        temp_row = df_future.iloc[0:1].copy()
        temp_row['MarkDown1'] = b
        pred = predict_model_direct(model, temp_row)['prediction_label'].iloc[0]
        simulated_sales.append(pred)

    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    ax3.plot(budget_range, simulated_sales, marker='s', color='#2ca02c', linewidth=2)
    ax3.set_xlabel("Presupuesto MarkDown 1 ($USD)")
    ax3.set_ylabel("Venta Semanal Estimada ($USD)")
    ax3.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig3)

st.markdown("---")

# 8. Módulo de Decisiones Ejecutivas
st.subheader("Modulo de Decisiones Ejecutivas")

def generar_informe_ejecutivo(store, dept, year, total_annual, peak_week, peak_sales,
                               m1, m3, df_future, temp_val, unemp_val, cpi_val):

    promedio_normal = df_future.loc[df_future['IsHoliday'] == 0, 'Ventas_Proyectadas'].mean()
    promedio_festivo = df_future.loc[df_future['IsHoliday'] == 1, 'Ventas_Proyectadas'].mean()
    incremento_festivo = ((promedio_festivo - promedio_normal) / promedio_normal * 100) if promedio_normal else 0

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

    if not api_key:
        return (
            "No se encontró OPENAI_API_KEY configurada. Mostrando resumen automático:\n\n"
            f"- **Proyección total anual:** ${total_annual:,.2f} USD\n"
            f"- **Semana de mayor venta:** Semana {int(peak_week)} (${peak_sales:,.2f} USD)\n"
            f"- **Impacto por festividades:** {incremento_festivo:+.1f}% vs semanas normales\n"
            f"- **Entorno Simulado:** Temperatura {temp_val}°F | Inflación CPI {cpi_val} | Desempleo {unemp_val}%\n"
        )

    client = OpenAI(api_key=api_key)

    prompt = (
        f"Summarize the annual sales forecast for Store {store}, Department {dept} in year {year}. "
        f"Total projected sales: ${total_annual:,.2f} USD. Peak sales occur at week {int(peak_week)} "
        f"(${peak_sales:,.2f} USD). Holiday weeks perform {incremento_festivo:+.1f}% vs normal weeks. "
        f"Environment conditions: Temperature {temp_val}°F, CPI {cpi_val}, Unemployment {unemp_val}%. "
        f"Provide 2 concise strategic recommendations in Spanish."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Senior Financial Analyst at Walmart."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=250
    )

    return response.choices[0].message.content

if st.button("Generar Informe Financiero", use_container_width=True):
    with st.spinner("Procesando métricas y redactando el informe gerencial..."):
        output_report = generar_informe_ejecutivo(
            store, dept, selected_year, total_annual, peak_week, peak_sales,
            m1, m3, df_future, temp_val, unemp_val, cpi_val
        )
        st.success("Dictamen Estratégico Generado:")
        st.markdown(output_report)
