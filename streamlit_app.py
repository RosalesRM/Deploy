import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts

# Configuración de la página
st.set_page_config(page_title="Clasificación Monte Carlo + ECharts", layout="wide")
st.title("🎲 Clasificación Monte Carlo con Visualización ECharts")

st.markdown("""
Este ejercicio genera dos clases de datos en 2D de manera aleatoria.  
Puedes ingresar múltiples puntos para clasificarlos según su distancia euclidiana promedio  
y visualizarlos de forma interactiva con **ECharts**.
""")

# ----------------------- PARÁMETROS ----------------------- #
st.sidebar.header("⚙️ Parámetros de la Simulación")

n_samples = st.sidebar.slider("Número de muestras por clase", 100, 5000, 1000)
std_dev = st.sidebar.slider("Dispersión (std dev)", 0.5, 3.0, 1.2)
seed = st.sidebar.number_input("Seed aleatoria", value=42, step=1)
np.random.seed(seed)

st.sidebar.markdown("### 📍 Centro Clase 0")
class_0_x = st.sidebar.number_input("X Clase 0", value=2.0)
class_0_y = st.sidebar.number_input("Y Clase 0", value=2.0)

st.sidebar.markdown("### 📍 Centro Clase 1")
class_1_x = st.sidebar.number_input("X Clase 1", value=6.0)
class_1_y = st.sidebar.number_input("Y Clase 1", value=6.0)

# ----------------------- PUNTOS NUEVOS ----------------------- #
st.sidebar.header("🧪 Puntos Nuevos")
user_input = st.sidebar.text_area("Introduce coordenadas (X,Y) por línea", value="4,4\n3,2\n6,5\n1,8")
new_points = []
for line in user_input.strip().split("\n"):
    try:
        x, y = map(float, line.split(","))
        new_points.append([x, y])
    except:
        continue

# ----------------------- DATOS SIMULADOS ----------------------- #
center_0 = np.array([class_0_x, class_0_y])
center_1 = np.array([class_1_x, class_1_y])

class_0 = np.random.normal(loc=center_0, scale=std_dev, size=(n_samples, 2))
class_1 = np.random.normal(loc=center_1, scale=std_dev, size=(n_samples, 2))

# ----------------------- CLASIFICACIÓN ----------------------- #
results = []
for pt in new_points:
    pt_np = np.array(pt)
    dist_0 = np.linalg.norm(class_0 - pt_np, axis=1).mean()
    dist_1 = np.linalg.norm(class_1 - pt_np, axis=1).mean()
    pred = 0 if dist_0 < dist_1 else 1
    results.append({
        "X": pt[0],
        "Y": pt[1],
        "Distancia Clase 0": round(dist_0, 2),
        "Distancia Clase 1": round(dist_1, 2),
        "Clase Predicha": pred
    })

# ----------------------- VISUALIZACIÓN CON ECHARTS ----------------------- #
st.subheader("📈 Visualización Interactiva (ECharts)")

scatter_data_0 = [{"value": list(p)} for p in class_0]
scatter_data_1 = [{"value": list(p)} for p in class_1]
new_point_data = [{"value": [r["X"], r["Y"]], "itemStyle": {"color": "green" if r["Clase Predicha"] == 0 else "orange"}} for r in results]

option = {
    "tooltip": {"trigger": "item"},
    "legend": {"data": ["Clase 0", "Clase 1", "Puntos Nuevos"]},
    "xAxis": {"type": "value", "name": "X"},
    "yAxis": {"type": "value", "name": "Y"},
    "series": [
        {
            "name": "Clase 0",
            "type": "scatter",
            "data": scatter_data_0,
            "symbolSize": 6,
            "itemStyle": {"color": "blue", "opacity": 0.4}
        },
        {
            "name": "Clase 1",
            "type": "scatter",
            "data": scatter_data_1,
            "symbolSize": 6,
            "itemStyle": {"color": "red", "opacity": 0.4}
        },
        {
            "name": "Puntos Nuevos",
            "type": "scatter",
            "data": new_point_data,
            "symbolSize": 12,
            "symbol": "pin",
        },
    ]
}

st_echarts(options=option, height="500px")

# ----------------------- RESULTADOS ----------------------- #
st.subheader("📋 Resultados de Clasificación")
df_results = pd.DataFrame(results)
st.dataframe(df_results)

csv = df_results.to_csv(index=False).encode("utf-8")
st.download_button("📥 Descargar Resultados CSV", data=csv, file_name="resultados_clasificacion.csv", mime="text/csv")

# ----------------------- VER DATOS SIMULADOS ----------------------- #
st.subheader("📊 Datos Simulados por Clase")
df_c0 = pd.DataFrame(class_0, columns=["x", "y"])
df_c0["Clase"] = 0
df_c1 = pd.DataFrame(class_1, columns=["x", "y"])
df_c1["Clase"] = 1
df_full = pd.concat([df_c0, df_c1], ignore_index=True)

if st.toggle("Mostrar datos simulados"):
    st.dataframe(df_full)
