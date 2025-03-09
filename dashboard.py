import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set(style='dark')

# Load dataset
data_urls = {
    "day": "https://raw.githubusercontent.com/DeffinP/dataset_analisis_data/refs/heads/main/day.csv",
    "hour": "https://raw.githubusercontent.com/DeffinP/dataset_analisis_data/refs/heads/main/hour.csv"
}

# Membaca file CSV
day_df = pd.read_csv(data_urls["day"])
hour_df = pd.read_csv(data_urls["hour"])
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(day_df['dteday'])

# Mapping season and weather
season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
weather_map = {1: "Clear/Partly Cloudy", 2: "Misty/Overcast", 3: "Light Rain/Light Snow", 4: "Heavy Rain/Thunderstorm"}

day_df["season"] = day_df["season"].map(season_map)
day_df["weathersit"] = day_df["weathersit"].map(weather_map)

# Function to categorize time
def categorize_time(hour):
    if 6 <= hour < 12:
        return "Pagi"
    elif 12 <= hour < 16:
        return "Siang"
    elif 16 <= hour < 20:
        return "Sore"
    else:
        return "Malam"

# Group hourly data
hourly_trend = hour_df.groupby("hr")[["casual", "registered"]].sum()
hourly_trend = hourly_trend.copy()
hourly_trend["time_category"] = hourly_trend.index.map(categorize_time)
time_based_trend = hourly_trend.groupby("time_category")[["casual", "registered"]].sum().sort_values(by="casual", ascending=False)

def plot_seasonal():
    seasonal_df = day_df.groupby("season")["cnt"].sum().sort_values(ascending=False)
    colors = ["#72BCD4" if season == seasonal_df.idxmax() else "#D3D3D3" for season in seasonal_df.index]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=seasonal_df.index, y=seasonal_df.values, palette=colors)
    plt.title("Jumlah Penyewaan Sepeda Berdasarkan Musim")
    plt.xlabel("Musim")
    plt.ylabel("Jumlah Penyewaan Sepeda")
    st.pyplot(plt)

def plot_weather():
    weather_df = day_df.groupby("weathersit")["cnt"].sum().sort_values(ascending=False)
    colors = ["#72BCD4" if weather == weather_df.idxmax() else "#D3D3D3" for weather in weather_df.index]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=weather_df.index, y=weather_df.values, palette=colors)
    plt.title("Jumlah Penyewaan Sepeda Berdasarkan Kondisi Cuaca")
    plt.xlabel("Kondisi Cuaca")
    plt.ylabel("Jumlah Penyewaan Sepeda")
    plt.xticks(rotation=15)
    st.pyplot(plt)

def plot_workingday():
    week_df = day_df.groupby("workingday")["cnt"].sum()
    week_df.index = ["Weekend/Holiday", "Working Day"]
    colors = ["#72BCD4", "#D3D3D3"]
    plt.figure(figsize=(6, 5))
    sns.barplot(x=week_df.index, y=week_df.values, palette=colors)
    plt.title("Jumlah Penyewaan Sepeda: Hari Kerja vs Weekend/Holiday")
    plt.xlabel("Hari")
    plt.ylabel("Jumlah Penyewaan Sepeda")
    st.pyplot(plt)

def plot_time_category():
    colors = ["#4C72B0", "#B0B0B0"]
    fig, ax = plt.subplots(figsize=(10, 6))
    time_based_trend.plot(kind="bar", stacked=True, color=colors, ax=ax)
    plt.xlabel("Kategori Waktu", fontsize=12)
    plt.ylabel("Jumlah Penyewaan Sepeda", fontsize=12)
    plt.title("Perbandingan Peminjaman Sepeda Casual vs Registered Berdasarkan Waktu", fontsize=14)
    plt.legend(["Casual", "Registered"], title="Tipe Pengguna")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for i, index in enumerate(time_based_trend.index):
        casual_value = time_based_trend.loc[index, "casual"]
        registered_value = time_based_trend.loc[index, "registered"]
        ax.text(i, casual_value / 2, f"{casual_value:,}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(i, casual_value + (registered_value / 2), f"{registered_value:,}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    st.pyplot(fig)

# Streamlit UI
st.title("Bike Sharing Dashboard")
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Visualisasi", ["Musim", "Cuaca", "Hari Kerja vs Libur", "Kategori Waktu"])

if page == "Musim":
    st.header("Jumlah Penyewaan Sepeda Berdasarkan Musim")
    plot_seasonal()
elif page == "Cuaca":
    st.header("Jumlah Penyewaan Sepeda Berdasarkan Cuaca")
    plot_weather()
elif page == "Hari Kerja vs Libur":
    st.header("Jumlah Penyewaan Sepeda pada Hari Kerja vs Libur")
    plot_workingday()
elif page == "Kategori Waktu":
    st.header("Perbandingan Peminjaman Sepeda Casual vs Registered Berdasarkan Waktu")
    plot_time_category()

