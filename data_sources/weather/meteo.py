"""
Données météorologiques via Open-Meteo (gratuit, sans clé API).
Fournit: vitesse du vent, direction du vent, humidité relative au moment du feu.

Archive ERA5 (historique) : https://archive-api.open-meteo.com
Forecast (5 derniers jours) : https://api.open-meteo.com
"""

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_type

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _is_recent(date_str):
    """True si la date est dans les 5 derniers jours (utiliser forecast API)"""
    try:
        d = date_type.fromisoformat(date_str)
        return (date_type.today() - d).days <= 5
    except Exception:
        return False


def _fetch_weather_range(lat, lon, start_date, end_date):
    """
    Récupère vent + humidité horaires pour une localisation sur toute la plage de dates.
    Retourne: {date_str: {hour_int: {wind_speed, wind_dir, humidity}}}
    """
    url = FORECAST_URL if _is_recent(start_date) else ARCHIVE_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "wind_speed_10m,wind_direction_10m,relative_humidity_2m,temperature_2m",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return {}
        hourly = r.json().get("hourly", {})
        times = hourly.get("time", [])
        speeds = hourly.get("wind_speed_10m", [])
        dirs = hourly.get("wind_direction_10m", [])
        hums = hourly.get("relative_humidity_2m", [])
        temps = hourly.get("temperature_2m", [])
        result = {}
        for i, t in enumerate(times):
            d_str = t[:10]
            h = int(t[11:13])
            result.setdefault(d_str, {})[h] = {
                "wind_speed": speeds[i] if i < len(speeds) else None,
                "wind_dir": dirs[i] if i < len(dirs) else None,
                "humidity": hums[i] if i < len(hums) else None,
                "temperature": temps[i] if i < len(temps) else None,
            }
        return result
    except Exception:
        return {}


def _get_acq_hour(acq_time):
    """Extrait l'heure UTC depuis acq_time FIRMS (format HHMM int)"""
    try:
        return int(float(acq_time)) // 100
    except Exception:
        return 12


def wind_direction_label(degrees):
    """Degrés → direction cardinale (ex: 'NO (315°)')"""
    if degrees is None:
        return "N/A"
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    idx = round(float(degrees) / 45) % 8
    return f"{dirs[idx]} ({int(degrees)}°)"


def fetch_weather_for_fires(fires_df, progress_callback=None):
    """
    Ajoute les colonnes wind_speed, wind_dir, humidity au DataFrame.

    Stratégie : grouper par grille 1° (~100 km) → 1 requête API par cellule
    couvrant toute la plage de dates → nombre d'appels = nb de cellules uniques.

    Returns: nouveau DataFrame avec colonnes météo ajoutées.
    """
    df = fires_df.copy()
    df["_lat_r"] = df["latitude"].round(0).astype(float)
    df["_lon_r"] = df["longitude"].round(0).astype(float)
    df["_date_str"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")

    date_min = df["_date_str"].min()
    date_max = df["_date_str"].max()
    unique_locs = df[["_lat_r", "_lon_r"]].drop_duplicates().values.tolist()

    # Cache : (lat_r, lon_r, date_str, hour) → {wind_speed, wind_dir, humidity}
    cache = {}

    def fetch_one(lat_r, lon_r):
        data = _fetch_weather_range(lat_r, lon_r, date_min, date_max)
        return lat_r, lon_r, data

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, lat, lon): (lat, lon) for lat, lon in unique_locs}
        done = 0
        for fut in as_completed(futures):
            lat_r, lon_r, data = fut.result()
            for d_str, hours in data.items():
                for h, vals in hours.items():
                    cache[(lat_r, lon_r, d_str, h)] = vals
            done += 1
            if progress_callback:
                progress_callback(done / len(unique_locs))

    def assign(row):
        lat_r = row["_lat_r"]
        lon_r = row["_lon_r"]
        d_str = row["_date_str"]
        hour = _get_acq_hour(row.get("acq_time", 1200))
        for h in [hour, (hour + 1) % 24, (hour - 1) % 24, 12]:
            v = cache.get((lat_r, lon_r, d_str, h))
            if v:
                return v
        return {"wind_speed": None, "wind_dir": None, "humidity": None, "temperature": None}

    weather = df.apply(assign, axis=1, result_type="expand")
    df["wind_speed"] = weather["wind_speed"]
    df["wind_dir"] = weather["wind_dir"]
    df["humidity"] = weather["humidity"]
    df["temperature"] = weather["temperature"]
    df.drop(columns=["_lat_r", "_lon_r", "_date_str"], inplace=True)
    return df
