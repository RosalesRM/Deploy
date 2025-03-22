import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

st.set_page_config(page_title="Explorador de Hoteles CDMX", layout="wide")
st.title("🏨 Explorador Interactivo de Hoteles en Ciudad de México")

st.markdown("""
Este playground une visualizaciones, datos y mapas para ayudarte a explorar y testear un dataset simulado de hoteles en CDMX.  
Usamos una base generada aleatoriamente para mostrar cómo podrías armar un panel de exploración real.  
""")

# ---------- 🌐 Dataset Simulado de Hoteles en CDMX ---------- #

np.random.seed(42)

N = st.slider("Número de hoteles simulados", 100, 10000, 1000)

hotel_data = pd.DataFrame({
    "Nombre": [f"Hotel {i}" for i in range(N)],
    "Preview": [f"https://picsum.photos/400/200?lock={i}" for i in range(N)],
    "Latitud": np.random.randn(N) / 100 + 19.4326,
    "Longitud": np.random.randn(N) / 100 + -99.1332,
    "Categoría": np.random.choice(["⭐ Económico", "⭐⭐⭐ Estándar", "⭐⭐⭐⭐ Lujo", "🏨 Boutique"], size=N),
    "Vistas": np.random.randint(0, 2000, size=N),
    "Disponible": np.random.choice([True, False], size=N),
    "Evaluación": np.random.randint(50, 100, size=N),
    "Usuario": np.random.choice(["Alice", "Bob", "Charly"], size=N),
})

# ---------- 🎛️ Filtros ---------- #

with st.container(border=True):
    usuarios = st.multiselect("👤 Filtrar por usuarios", ["Alice", "Bob", "Charly"], default=["Alice", "Bob", "Charly"])
    categorias = st.multiselect("🏷️ Categoría de hotel", hotel_data["Categoría"].unique(), default=hotel_data["Categoría"].unique())
    rolling_avg = st.toggle("Promedio móvil por usuario")

filtered_data = hotel_data[(hotel_data["Usuario"].isin(usuarios)) & (hotel_data["Categoría"].isin(categorias))]

# ---------- 📈 Gráfica de Vistas por Usuario ---------- #

tab1, tab2 = st.tabs(["📊 Gráfico", "📋 Tabla"])

views_data = filtered_data.groupby(["Usuario"]).agg({"Vistas": "sum"}).reset_index()

chart_df = pd.DataFrame()
for u in usuarios:
    subset = filtered_data[filtered_data["Usuario"] == u]["Vistas"].reset_index(drop=True)
    if rolling_avg:
        subset = subset.rolling(7).mean().dropna()
    chart_df[u] = subset

tab1.line_chart(chart_df, height=250)
tab2.dataframe(filtered_data[["Nombre", "Usuario", "Categoría", "Vistas", "Disponible", "Evaluación"]], use_container_width=True)

# ---------- 🧾 Tabla editable con imágenes ---------- #

st.markdown("### 🖼️ Tabla editable con info general y fotos")
config = {
    "Preview": st.column_config.ImageColumn("Foto del hotel"),
    "Evaluación": st.column_config.ProgressColumn("Evaluación /100", min_value=0, max_value=100),
}
if st.toggle("Editar información"):
    edited_data = st.data_editor(filtered_data, column_config=config, use_container_width=True)
else:
    st.dataframe(filtered_data, column_config=config, use_container_width=True)

# ---------- 🗺️ Mapa geoespacial ---------- #

st.markdown("### 🌍 Mapa de Hoteles en CDMX")

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=19.4326,
        longitude=-99.1332,
        zoom=12,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HexagonLayer',
            data=filtered_data,
            get_position='[Longitud, Latitud]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=filtered_data,
            get_position='[Longitud, Latitud]',
            get_color='[0, 120, 255, 160]',
            get_radius=100,
        ),
    ],
))
