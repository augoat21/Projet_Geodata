"""
Calcul du score de risque de feu basé sur un modèle XGBoost entraîné
sur les données historiques FIRMS + météo Open-Meteo (France métropolitaine).

Score de 0 à 100 (probabilité de feu × 100).
"""

import requests
import pandas as pd
import numpy as np
import io
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import shape, Point
from scipy.interpolate import griddata
from data_sources.fires.firms import COUNTRY_BBOX

# ─── Chargement du modèle ML ───────────────────────────────────────────
_ml_model = None
_ml_available = False

def _load_ml_model():
    """Charge le modèle XGBoost s'il existe."""
    global _ml_model, _ml_available
    model_path = Path(__file__).resolve().parent.parent / "ml" / "models" / "fire_risk_xgboost.joblib"
    if model_path.exists():
        try:
            import joblib
            _ml_model = joblib.load(model_path)
            _ml_available = True
        except Exception:
            _ml_available = False
    else:
        _ml_available = False

_load_ml_model()

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Cache du polygone pays
_country_polygon_cache = {}


def _fetch_nominatim_geojson(query):
    """Requête Nominatim avec polygone détaillé."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "polygon_geojson": 1,
                "polygon_threshold": 0.001,
                "limit": 1,
            },
            headers={"User-Agent": "projet_geodata"},
            timeout=30,
        )
        data = r.json()
        if data and "geojson" in data[0]:
            return data[0]["geojson"]
    except Exception:
        pass
    return None


def _get_query_for_country(country):
    """
    Retourne la requête Nominatim adaptée.
    Pour la France, cible la métropole uniquement.
    """
    special = {
        "France": "France métropolitaine",
    }
    return special.get(country, country)


def get_country_polygon(country):
    """
    Récupère le polygone du pays via Nominatim.
    Retourne un objet shapely Polygon/MultiPolygon, ou None.
    """
    if country in _country_polygon_cache:
        return _country_polygon_cache[country]

    query = _get_query_for_country(country)
    geojson = _fetch_nominatim_geojson(query)
    if geojson is None:
        # Fallback sur le nom brut
        geojson = _fetch_nominatim_geojson(country)

    poly = None
    if geojson:
        poly = shape(geojson)
    _country_polygon_cache[country] = poly
    return poly


def get_country_geojson(country):
    """
    Récupère le GeoJSON brut du pays (pour affichage folium).
    """
    query = _get_query_for_country(country)
    geojson = _fetch_nominatim_geojson(query)
    if geojson is None:
        geojson = _fetch_nominatim_geojson(country)
    return geojson


def get_country_grid(country, resolution=0.5):
    """
    Génère une grille de points (lat, lon) couvrant le pays
    avec la résolution donnée (en degrés).
    """
    bbox = COUNTRY_BBOX.get(country)
    if not bbox:
        return []

    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + resolution, resolution)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + resolution, resolution)

    # Filtrer par polygone du pays si disponible
    poly = get_country_polygon(country)

    points = []
    for lat in lats:
        for lon in lons:
            pt = (round(float(lat), 2), round(float(lon), 2))
            if poly is not None:
                if not poly.contains(Point(pt[1], pt[0])):
                    continue
            points.append(pt)
    return points


def _fetch_forecast(lat, lon, days=7):
    """
    Récupère les prévisions météo journalières pour un point.
    Retourne une liste de dicts avec date, temp_max, humidity_min, wind_max.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
        "forecast_days": days,
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=15)
        if r.status_code != 200:
            return []
        daily = r.json().get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])
        hums = daily.get("relative_humidity_2m_min", [])
        winds = daily.get("wind_speed_10m_max", [])
        precips = daily.get("precipitation_sum", [])

        results = []
        for i, d in enumerate(dates):
            results.append({
                "date": d,
                "temp_max": temps[i] if i < len(temps) else None,
                "humidity_min": hums[i] if i < len(hums) else None,
                "wind_max": winds[i] if i < len(winds) else None,
                "precip_sum": precips[i] if i < len(precips) else None,
            })
        return results
    except Exception:
        return []



def _compute_score_ml(lat, lon, temp_max, humidity_min, wind_max, precip_sum, date_str):
    """
    Scoring via le modèle XGBoost.
    Retourne la probabilité de feu × 100 (score 0-100).
    """
    dt = pd.to_datetime(date_str)
    month = dt.month
    day_of_year = dt.timetuple().tm_yday
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    features = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "temp_max": temp_max if temp_max is not None else 20,
        "humidity_min": humidity_min if humidity_min is not None else 50,
        "wind_max": wind_max if wind_max is not None else 10,
        "precip_sum": precip_sum if precip_sum is not None else 0,
        "month": month,
        "day_of_year": day_of_year,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }])

    proba = _ml_model.predict_proba(features)[0][1]  # probabilité de la classe 1 (feu)
    return round(proba * 100, 1)


def _compute_score(temp_max, humidity_min, wind_max,
                   lat=None, lon=None, date_str=None, precip_sum=None):
    """
    Calcule le score de risque 0-100 via le modèle XGBoost.
    """
    if not _ml_available:
        raise RuntimeError("Modèle XGBoost non disponible.")
    return _compute_score_ml(lat, lon, temp_max, humidity_min, wind_max, precip_sum, date_str)


def risk_color(score):
    """Retourne la couleur selon le score."""
    if score < 30:
        return "green"
    elif score < 60:
        return "orange"
    else:
        return "red"


def risk_label(score):
    """Retourne le label de risque."""
    if score < 30:
        return "Faible"
    elif score < 60:
        return "Modéré"
    else:
        return "Élevé"


def generate_risk_image(day_df, country):
    """
    Génère une image PNG interpolée (heatmap de valeurs) à partir du DataFrame d'un jour.
    Retourne (image_base64, bounds) pour folium.ImageOverlay.
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    lats = day_df["latitude"].values
    lons = day_df["longitude"].values
    scores = day_df["score"].values

    if len(lats) < 3:
        return None, None

    # Bornes de l'image : utiliser la bbox du pays pour couvrir tout le territoire
    bbox = COUNTRY_BBOX.get(country)
    if bbox:
        pad = 0.25
        lat_min, lat_max = bbox["lat_min"] - pad, bbox["lat_max"] + pad
        lon_min, lon_max = bbox["lon_min"] - pad, bbox["lon_max"] + pad
    else:
        pad = 0.25
        lat_min, lat_max = lats.min() - pad, lats.max() + pad
        lon_min, lon_max = lons.min() - pad, lons.max() + pad

    # Grille fine pour l'interpolation
    grid_res = 200
    grid_lon = np.linspace(lon_min, lon_max, grid_res)
    grid_lat = np.linspace(lat_min, lat_max, grid_res)
    grid_lon_2d, grid_lat_2d = np.meshgrid(grid_lon, grid_lat)

    # Interpolation des scores sur la grille (linear + nearest pour combler les trous)
    grid_scores = griddata(
        (lons, lats), scores,
        (grid_lon_2d, grid_lat_2d),
        method="linear",
        fill_value=np.nan,
    )
    # Combler les NaN restants (îles, bords) avec le point le plus proche
    nan_mask = np.isnan(grid_scores)
    if nan_mask.any():
        grid_nearest = griddata(
            (lons, lats), scores,
            (grid_lon_2d, grid_lat_2d),
            method="nearest",
        )
        grid_scores[nan_mask] = grid_nearest[nan_mask]
    # Clipper pour éviter les valeurs hors bornes
    grid_scores = np.clip(grid_scores, 0, 100)

    # Masquer les zones hors du polygone pays
    poly = get_country_polygon(country)
    if poly is not None:
        mask = np.zeros_like(grid_scores, dtype=bool)
        for i in range(grid_res):
            for j in range(grid_res):
                if not poly.contains(Point(grid_lon_2d[i, j], grid_lat_2d[i, j])):
                    mask[i, j] = True
        grid_scores = np.ma.array(grid_scores, mask=mask)

    # Colormap aligné sur les seuils de risque : 0-30 vert, 30-60 jaune/orange, 60-100 rouge
    cmap = LinearSegmentedColormap.from_list(
        "fire_risk",
        [
            (0.0,  "#00c853"),   # 0   — vert
            (0.30, "#00c853"),   # 30  — vert (fin Faible)
            (0.35, "#ffd600"),   # 35  — jaune (début Modéré)
            (0.60, "#ff9100"),   # 60  — orange (fin Modéré)
            (0.65, "#dd2c00"),   # 65  — rouge (début Élevé)
            (1.0,  "#b71c1c"),   # 100 — rouge foncé
        ],
        N=256,
    )
    cmap.set_bad(alpha=0)  # transparent pour les zones masquées

    # Générer l'image
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(
        grid_scores,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=100,
        aspect="auto",
        interpolation="bilinear",
    )
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    bounds = [[float(lat_min), float(lon_min)], [float(lat_max), float(lon_max)]]
    return f"data:image/png;base64,{img_b64}", bounds


def compute_fire_risk(country, days=7, resolution=0.5, progress_callback=None):
    """
    Calcule le risque de feu pour un pays sur les prochains jours.

    Retourne un DataFrame avec colonnes :
    latitude, longitude, date, temp_max, humidity_min, wind_max, score, color, label
    """
    points = get_country_grid(country, resolution=resolution)
    if not points:
        return pd.DataFrame()

    all_rows = []
    done = 0

    def fetch_one(lat, lon):
        return lat, lon, _fetch_forecast(lat, lon, days)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, lat, lon): (lat, lon) for lat, lon in points}
        for fut in as_completed(futures):
            lat, lon, forecasts = fut.result()
            for f in forecasts:
                score = _compute_score(
                    f["temp_max"], f["humidity_min"], f["wind_max"],
                    lat=lat, lon=lon, date_str=f["date"],
                    precip_sum=f.get("precip_sum"),
                )
                all_rows.append({
                    "latitude": lat,
                    "longitude": lon,
                    "date": f["date"],
                    "temp_max": f["temp_max"],
                    "humidity_min": f["humidity_min"],
                    "wind_max": f["wind_max"],
                    "score": score,
                    "color": risk_color(score),
                    "label": risk_label(score),
                })
            done += 1
            if progress_callback:
                progress_callback(done / len(points))

    return pd.DataFrame(all_rows)
