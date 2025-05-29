import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import io

st.set_page_config(page_title="Zbulimi i Anomalive", layout="wide")
st.markdown("""
    <style>
    html, body, .main {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
        scroll-behavior: smooth;
    }
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 700;
    }
    .stSelectbox > label, .stSlider > label, .stSubheader {
        color: #0d3b66;
        font-weight: bold;
    }
    .stButton > button {
        background-color: #198754;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
    }
    .stButton > button:hover {
        background-color: #145c33;
        transform: scale(1.02);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Analizë Interaktive e Anomalive në të Dhënat Meteorologjike")
st.markdown("<p style='font-size:18px;'>Zgjidh një dataset, filtra periudhën kohore dhe zbuloni anomalitë me algoritmin SARIMA. Të dhënat përfshijnë presionin e ajrit, temperaturën, lagështinë dhe shpejtësinë e erës.</p>", unsafe_allow_html=True)

@st.cache_data
def load_and_merge_clean(file1, file2, value_col):
    def clean_df(path):
        df = pd.read_csv(path)
        df.columns = [col.strip() for col in df.columns]
        if 'Timestamp' not in df.columns:
            raise ValueError("Kolona 'Timestamp' mungon")
        df['valid'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['valid'])
        df = df[["valid", value_col]]
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna()
        return df

    df1 = clean_df(file1)
    df2 = clean_df(file2)
    df = pd.concat([df1, df2], ignore_index=True)
    df.set_index('valid', inplace=True)
    df.sort_index(inplace=True)
    return df

# Dataset-et
datasets = {
    "Air Pressure": ("dataset/airpreasure.csv", "dataset/Air_Pressure1.csv", "Air_Pressure"),
    "Temperature": ("dataset/temperature.csv", "dataset/Air_Temperature1.csv", "Temperature"),
    "Humidity": ("dataset/humidity.csv", "dataset/Humidity1.csv", "Humidity"),
    "Wind Speed": ("dataset/windspeed.csv", "dataset/Wind_Speed1.csv", "WindSpeed")
}

units = {
    "Air Pressure": "hPa",
    "Temperature": "°C",
    "Humidity": "%",
    "Wind Speed": "m/s"
}

selected_data = st.selectbox("Zgjidh llojin e të dhënave:", list(datasets.keys()))
file1, file2, value_col = datasets[selected_data]
unit = units[selected_data]

try:
    df = load_and_merge_clean(file1, file2, value_col)
    st.success(f"Të dhënat për '{selected_data}' u gjetën me sukses.")
except Exception as e:
    st.error(f"Gabim gjatë përpunimit të të dhënave: {e}")
    st.stop()

min_year = df.index.min().year
max_year = df.index.max().year
st.markdown("---")
st.subheader("Filtrim i periudhës kohore")
selected_years = st.slider("Zgjidh periudhën e analizës:", min_year, max_year, (min_year, max_year))
df = df[(df.index.year >= selected_years[0]) & (df.index.year <= selected_years[1])]

interval_map = {'D': 'Ditë', 'W': 'Java', 'M': 'Muaj', 'Y': 'Vit'}
interval = st.selectbox("Intervali kohor për analizë:", options=list(interval_map.keys()), format_func=lambda x: interval_map[x])

@st.cache_data
def sarima_anomaly_detection(df, colname, interval, threshold_factor=3):
    grouped = df[[colname]].resample(interval).mean()
    grouped['rolling_mean'] = grouped[colname].rolling(window=3, center=True).mean()
    grouped['residual'] = grouped[colname] - grouped['rolling_mean']
    threshold = grouped['residual'].std() * threshold_factor
    grouped['is_anomaly'] = grouped['residual'].abs() > threshold
    return grouped

st.markdown("---")
st.subheader("Parametrat e analizës")
threshold = st.slider("Pragu për detektimin e anomalive (standarde devijimi):", 1.0, 5.0, 3.0)
grouped = sarima_anomaly_detection(df, value_col, interval, threshold)

st.markdown("---")
st.subheader("Vizualizimi i Anomalive (interaktiv)")
fig = px.line(grouped, x=grouped.index, y=value_col, title=f"{selected_data} me Anomali",
              labels={value_col: f"{selected_data} ({unit})", 'index': 'Data'})
fig.add_scatter(x=grouped[grouped['is_anomaly']].index,
                y=grouped[grouped['is_anomaly']][value_col],
                mode='markers', name='Anomali', marker=dict(color='red', size=8))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Statistika për analizën")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Anomali të zbuluara", grouped['is_anomaly'].sum())
col2.metric(f"Mesatare ({unit})", f"{grouped[value_col].mean():.2f}")
col3.metric(f"Maksimumi ({unit})", f"{grouped[value_col].max():.2f}")
col4.metric(f"Minimumi ({unit})", f"{grouped[value_col].min():.2f}")

st.markdown("---")
st.subheader("Tabela e rezultateve")
st.dataframe(grouped.reset_index(), use_container_width=True)

st.markdown("---")
st.subheader("Shkarkim i rezultateve")
download_df = grouped.reset_index()
buffer = io.BytesIO()
download_df.to_excel(buffer, index=False, engine='openpyxl')
buffer.seek(0)
st.download_button("Shkarko Excel", data=buffer,
                   file_name=f"anomalies_{selected_data.replace(' ', '_')}.xlsx",
                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
