import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración inicial de la app
st.set_page_config(page_title="Simulación Monte Carlo", layout="wide")
st.title("🎲 Clasificación Monte Carlo con Múltiples Puntos y Visualización Ajustable")

st.markdown("""
Esta app genera dos clases de datos en 2D simulados.  
Luego puedes ingresar **múltiples puntos nuevos** para clasificarlos según su **distancia euclidiana promedio**.  
Puedes ajustar los parámetros de dispersión, centros y la visualización del gráfico.
""")

# ---------- ⚙️ Parámetros de simulación ---------- #
st.sidebar.header("⚙️ Parámetros de la Simulación")

n_samples = st.sidebar.slider("Número de muestras por clase", 100, 5000, 1000)
std_dev = st.sidebar.slider("Dispersión (std dev)", 0.5, 3.0, 1.2)
seed = st.sidebar.number_input("Seed aleatoria", value=42, step=1)
np.random.seed(seed)

# ---------- 📍 Centros de las clases ---------- #
st.sidebar.markdown("### 📍 Centro Clase 0")
class_0_x = st.sidebar.number_input("X Clase 0", value=2.0)
class_0_y = st.sidebar.number_input("Y Clase 0", value=2.0)

st.sidebar.markdown("### 📍 Centro Clase 1")
class_1_x = st.sidebar.number_input("X Clase 1", value=6.0)
class_1_y = st.sidebar.number_input("Y Clase 1", value=6.0)

# ---------- 🎨 Ajustes visuales ---------- #
st.sidebar.header("🎨 Ajustes del Gráfico")
fig_width = st.sidebar.slider("Ancho del gráfico", 6, 16, 10)
fig_height = st.sidebar.slider("Alto del gráfico", 4, 12, 6)
font_scale = st.sidebar.slider("Tamaño de letra", 10, 30, 14)

# ---------- 🧪 Ingreso de puntos nuevos ---------- #
st.sidebar.header("🧪 Puntos Nuevos")
user_input = st.sidebar.text_area("Introduce coordenadas (X,Y) por línea", value="4,4\n3,2\n6,5\n1,8")
new_points = []
for line in user_input.strip().split("\n"):
    try:
        x, y = map(float, line.split(","))
        new_points.append([x, y])
    except:
        continue

# ---------- 🧬 Generación de datos ---------- #
center_0 = np.array([class_0_x, class_0_y])
center_1 = np.array([class_1_x, class_1_y])

class_0 = np.random.normal(loc=center_0, scale=std_dev, size=(n_samples, 2))
class_1 = np.random.normal(loc=center_1, scale=std_dev, size=(n_samples, 2))

# ---------- 📏 Clasificación ---------- #
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

# ---------- 📈 Visualización ---------- #
st.subheader("📈 Visualización del Espacio 2D y Clasificación")

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.scatter(class_0[:, 0], class_0[:, 1], alpha=0.3, label="Clase 0", color="blue")
ax.scatter(class_1[:, 0], class_1[:, 1], alpha=0.3, label="Clase 1", color="red")

for r in results:
    clr = "green" if r["Clase Predicha"] == 0 else "orange"
    ax.scatter(r["X"], r["Y"], color=clr, s=100, marker="X", label=f"Punto → Clase {r['Clase Predicha']}")

ax.set_title("Clasificación de Puntos Nuevos por Distancia Euclidiana", fontsize=font_scale + 4)
ax.set_xlabel("X", fontsize=font_scale)
ax.set_ylabel("Y", fontsize=font_scale)
ax.tick_params(labelsize=font_scale - 2)
ax.legend(fontsize=font_scale - 2)
ax.grid(True)

st.pyplot(fig)

# ---------- 📋 Resultados de Clasificación ---------- #
st.subheader("📋 Resultados de Clasificación")
df_results = pd.DataFrame(results)
st.dataframe(df_results)

csv = df_results.to_csv(index=False).encode("utf-8")
st.download_button("📥 Descargar Resultados CSV", data=csv, file_name="resultados_clasificacion.csv", mime="text/csv")

# ---------- 🧬 Ver Datos Simulados ---------- #
st.subheader("📊 Datos Simulados por Clase")
df_c0 = pd.DataFrame(class_0, columns=["x", "y"])
df_c0["Clase"] = 0
df_c1 = pd.DataFrame(class_1, columns=["x", "y"])
df_c1["Clase"] = 1
df_full = pd.concat([df_c0, df_c1], ignore_index=True)

if st.toggle("Mostrar datos simulados"):
    st.dataframe(df_full)
