"""
Construction du dataset d'entraînement pour la prédiction de risque de feu.

Principe :
- Exemples positifs (label=1) : points où un feu a été détecté (FIRMS)
- Exemples négatifs (label=0) : points aléatoires dans le même pays/période sans feu
- Pour chaque point : on récupère la météo historique via Open-Meteo Archive API

Optimisation : les requêtes météo sont regroupées par cellule de grille 0.5°/0.25
et par plage de dates (max 1 an par requête) pour minimiser les appels API.

Usage :
    python -m ml.build_dataset
    python -m ml.build_dataset --countries France --years 2020 2021 2022 --samples 10000
"""

import argparse
import sys
import random
import time
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_sources.fires.firms import COUNTRY_BBOX

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_DIR = Path(__file__).resolve().parent.parent / "data_sources" / "fires" / "historical"
OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"


def load_fires(country, years):
    """Charge les feux FIRMS pour un pays et des années données."""
    frames = []
    for year in years:
        path = HISTORICAL_DIR / str(year) / f"viirs-snpp_{year}_{country}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["latitude", "longitude", "acq_date", "frp", "confidence"])
            df["year"] = year
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Garder les détections de confiance haute et nominale
    df = df[df["confidence"].isin(["h", "high", "H", "n", "nominal", "N"])].copy()
    return df


def sample_fires(fires_df, n_samples):
    """Échantillonne n feux aléatoirement. Si n_samples=0, prend tout."""
    if n_samples == 0 or len(fires_df) <= n_samples:
        return fires_df.copy()
    return fires_df.sample(n=n_samples, random_state=42).copy()


def generate_negatives(country, fires_df, n_samples):
    """
    Génère des points négatifs (pas de feu) dans la bounding box du pays.
    On s'assure qu'ils sont éloignés d'au moins 0.1° de tout feu le même jour.
    """
    bbox = COUNTRY_BBOX.get(country)
    if not bbox:
        return pd.DataFrame()

    dates = fires_df["acq_date"].unique().tolist()
    if not dates:
        return pd.DataFrame()

    negatives = []
    fire_coords = set(zip(
        fires_df["latitude"].round(1),
        fires_df["longitude"].round(1),
        fires_df["acq_date"]
    ))

    attempts = 0
    max_attempts = n_samples * 10

    while len(negatives) < n_samples and attempts < max_attempts:
        lat = round(random.uniform(bbox["lat_min"], bbox["lat_max"]), 2)
        lon = round(random.uniform(bbox["lon_min"], bbox["lon_max"]), 2)
        date = random.choice(dates)

        key = (round(lat, 1), round(lon, 1), date)
        if key not in fire_coords:
            negatives.append({
                "latitude": lat,
                "longitude": lon,
                "acq_date": date,
            })
        attempts += 1

    return pd.DataFrame(negatives)


def _fetch_weather_range(lat, lon, start_date, end_date):
    """
    Récupère la météo historique pour un point sur une plage de dates.
    Open-Meteo Archive accepte des plages de plusieurs mois.
    Retourne un dict {date_str: {temp_max, humidity_min, wind_max, precip_sum}}.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }
    for attempt in range(3):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return {}
            daily = r.json().get("daily", {})
            dates = daily.get("time", [])
            temps = daily.get("temperature_2m_max", [])
            hums = daily.get("relative_humidity_2m_min", [])
            winds = daily.get("wind_speed_10m_max", [])
            precips = daily.get("precipitation_sum", [])

            result = {}
            for i, d in enumerate(dates):
                result[d] = {
                    "temp_max": temps[i] if i < len(temps) else None,
                    "humidity_min": hums[i] if i < len(hums) else None,
                    "wind_max": winds[i] if i < len(winds) else None,
                    "precip_sum": precips[i] if i < len(precips) else None,
                }
            return result
        except Exception:
            time.sleep(2)
    return {}


def enrich_with_weather(df, max_workers=6):
    """
    Ajoute les colonnes météo au DataFrame.

    Stratégie d'optimisation :
    - Regrouper par cellule de grille 0.5°
    - Pour chaque cellule, déterminer la plage de dates min-max
    - Faire UNE requête par cellule × année (au lieu d'une par point × date)
    """
    # Arrondir à la grille 0.5°
    df["grid_lat"] = (df["latitude"] * 2).round() / 2
    df["grid_lon"] = (df["longitude"] * 2).round() / 2
    df["acq_date_str"] = df["acq_date"].astype(str)

    # Trouver les cellules uniques avec leurs plages de dates
    grid_groups = df.groupby(["grid_lat", "grid_lon"]).agg(
        date_min=("acq_date_str", "min"),
        date_max=("acq_date_str", "max"),
    ).reset_index()

    # Découper en chunks par année pour éviter des requêtes trop grosses
    tasks = []
    for _, row in grid_groups.iterrows():
        lat, lon = row["grid_lat"], row["grid_lon"]
        d_min = pd.to_datetime(row["date_min"])
        d_max = pd.to_datetime(row["date_max"])

        # Découper par année
        current = d_min
        while current <= d_max:
            year_end = min(d_max, pd.Timestamp(f"{current.year}-12-31"))
            tasks.append({
                "lat": lat, "lon": lon,
                "start": current.strftime("%Y-%m-%d"),
                "end": year_end.strftime("%Y-%m-%d"),
            })
            current = pd.Timestamp(f"{current.year + 1}-01-01")

    print(f"  Cellules de grille uniques : {len(grid_groups)}")
    print(f"  Requêtes API à faire : {len(tasks)}")
    print(f"  Estimation : ~{len(tasks) // 3 // 60 + 1} minutes")

    # Cache global : (grid_lat, grid_lon, date_str) -> weather dict
    weather_cache = {}
    done = 0

    def fetch_task(task):
        data = _fetch_weather_range(task["lat"], task["lon"], task["start"], task["end"])
        return task["lat"], task["lon"], data

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_task, t): t for t in tasks}
        for fut in as_completed(futures):
            lat, lon, data = fut.result()
            for date_str, weather in data.items():
                weather_cache[(lat, lon, date_str)] = weather
            done += 1
            if done % 50 == 0:
                print(f"  Météo : {done}/{len(tasks)} requêtes terminées ({done*100//len(tasks)}%)")

    print(f"  Cache météo : {len(weather_cache)} entrées (date × cellule)")

    # Mapper les résultats sur chaque ligne du DataFrame
    def get_weather_col(row, col):
        key = (row["grid_lat"], row["grid_lon"], row["acq_date_str"])
        w = weather_cache.get(key)
        if w:
            return w.get(col)
        return None

    for col in ["temp_max", "humidity_min", "wind_max", "precip_sum"]:
        df[col] = df.apply(lambda r, c=col: get_weather_col(r, c), axis=1)

    df.drop(columns=["grid_lat", "grid_lon", "acq_date_str"], inplace=True)
    return df


def add_temporal_features(df):
    """Ajoute les features temporelles."""
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["month"] = df["acq_date"].dt.month
    df["day_of_year"] = df["acq_date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def build_dataset(countries, years, n_samples_per_country=0):
    """Pipeline complet de construction du dataset."""
    all_data = []

    for country in countries:
        print(f"\n{'='*60}")
        print(f"Traitement de {country}...")
        print(f"{'='*60}")

        # 1. Charger les feux
        fires = load_fires(country, years)
        if fires.empty:
            print(f"  Aucune donnée pour {country}, skip.")
            continue
        print(f"  Feux chargés : {len(fires)} détections")

        # 2. Échantillonner les positifs
        positives = sample_fires(fires, n_samples_per_country)
        positives["label"] = 1
        print(f"  Positifs échantillonnés : {len(positives)}")

        # 3. Générer les négatifs (même nombre que les positifs)
        n_neg = len(positives) if n_samples_per_country == 0 else n_samples_per_country
        negatives = generate_negatives(country, fires, n_neg)
        negatives["label"] = 0
        print(f"  Négatifs générés : {len(negatives)}")

        # 4. Combiner
        combined = pd.concat([
            positives[["latitude", "longitude", "acq_date", "label"]],
            negatives[["latitude", "longitude", "acq_date", "label"]],
        ], ignore_index=True)

        # 5. Enrichir avec météo
        print(f"  Récupération météo pour {len(combined)} points...")
        combined = enrich_with_weather(combined)

        combined["country"] = country
        all_data.append(combined)

    if not all_data:
        print("Aucune donnée construite !")
        return

    dataset = pd.concat(all_data, ignore_index=True)
    dataset = add_temporal_features(dataset)

    # Supprimer les lignes sans météo
    before = len(dataset)
    dataset.dropna(subset=["temp_max", "humidity_min", "wind_max"], inplace=True)
    print(f"\nLignes supprimées (météo manquante) : {before - len(dataset)}")
    print(f"Dataset final : {len(dataset)} lignes")
    print(f"  Positifs : {(dataset['label'] == 1).sum()}")
    print(f"  Négatifs : {(dataset['label'] == 0).sum()}")

    # Sauvegarder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    countries_str = "_".join(countries[:3])
    years_str = f"{min(years)}-{max(years)}"
    filename = f"dataset_{countries_str}_{years_str}.csv"
    output_path = OUTPUT_DIR / filename
    dataset.to_csv(output_path, index=False)
    print(f"\nDataset sauvegardé : {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construction du dataset ML pour la prédiction de feux")
    parser.add_argument("--countries", nargs="+", default=["France"],
                        help="Pays à inclure")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2012, 2025)),
                        help="Années à inclure")
    parser.add_argument("--samples", type=int, default=0,
                        help="Nombre d'échantillons positifs par pays (0 = tous)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Nombre de threads pour les requêtes API")
    args = parser.parse_args()

    build_dataset(args.countries, args.years, args.samples)
