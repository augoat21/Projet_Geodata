"""
Module Sentinel-2 via Google Earth Engine
Récupère des images avant/après un feu et calcule le dNBR (zones brûlées)
"""

import ee

GEE_PROJECT = "my-project-geodata-486610"

# Seuils dNBR standard (USGS)
DNBR_THRESHOLDS = {
    "Non brûlé": (-0.1, 0.1),
    "Brûlure faible": (0.1, 0.27),
    "Brûlure modérée-faible": (0.27, 0.44),
    "Brûlure modérée-élevée": (0.44, 0.66),
    "Brûlure sévère": (0.66, 1.3),
}

# Palette de couleurs pour le dNBR
DNBR_VIS = {
    "min": -0.2,
    "max": 0.8,
    "palette": ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"],
}

# Vraies couleurs (RGB)
TRUE_COLOR_VIS = {
    "bands": ["B4", "B3", "B2"],
    "min": 0,
    "max": 3000,
}

# Fausses couleurs (NIR, Red, Green) - met en évidence la végétation
FALSE_COLOR_VIS = {
    "bands": ["B8", "B4", "B3"],
    "min": 0,
    "max": 4000,
}


def init_gee():
    """Initialise Google Earth Engine s'il ne l'est pas déjà"""
    try:
        ee.Number(1).getInfo()
    except Exception:
        ee.Initialize(project=GEE_PROJECT)


def _get_best_image(region, date_start, date_end):
    """
    Récupère la meilleure image Sentinel-2 (moins nuageuse) sur une période et zone.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = collection.size().getInfo()
    if count == 0:
        return None

    return collection.first().clip(region)


def _compute_nbr(image):
    """Calcule le NBR (Normalized Burn Ratio) = (B8 - B12) / (B8 + B12)"""
    return image.normalizedDifference(["B8", "B12"]).rename("NBR")


def _compute_water_mask(image):
    """
    Masque eau via NDWI = (B3 - B8) / (B3 + B8).
    Les pixels avec NDWI > 0 sont considérés comme de l'eau.
    """
    ndwi = image.normalizedDifference(["B3", "B8"])
    return ndwi.lte(0)  # True = non-eau, False = eau


def get_satellite_analysis(lat, lon, fire_date, buffer_km=10, days_before=30, days_after=30):
    """
    Analyse satellite autour d'un point de feu.

    Args:
        lat, lon: coordonnées du feu
        fire_date: date du feu (string 'YYYY-MM-DD' ou datetime)
        buffer_km: rayon autour du point en km
        days_before: jours avant le feu pour l'image "avant"
        days_after: jours après le feu pour l'image "après"

    Returns:
        dict avec les URLs des tuiles pour Folium + stats
    """
    init_gee()

    fire_date_str = str(fire_date)[:10]
    fire_date_ee = ee.Date(fire_date_str)

    # Zone d'analyse (buffer autour du point)
    point = ee.Geometry.Point([lon, lat])
    buffer_m = buffer_km * 1000
    region = point.buffer(buffer_m).bounds()

    # Périodes
    pre_start = fire_date_ee.advance(-days_before, "day")
    pre_end = fire_date_ee.advance(-1, "day")
    post_start = fire_date_ee.advance(1, "day")
    post_end = fire_date_ee.advance(days_after, "day")

    # Images avant et après
    pre_image = _get_best_image(region, pre_start, pre_end)
    post_image = _get_best_image(region, post_start, post_end)

    if pre_image is None or post_image is None:
        return None

    # Calcul NBR et dNBR
    nbr_pre = _compute_nbr(pre_image)
    nbr_post = _compute_nbr(post_image)
    dnbr = nbr_pre.subtract(nbr_post).rename("dNBR")

    # Masque eau : exclure les pixels d'eau (NDWI > 0 sur l'image post)
    water_mask = _compute_water_mask(post_image)
    dnbr = dnbr.updateMask(water_mask)

    # Générer les URLs de tuiles pour Folium
    tiles = {}

    # Zones brûlées par sévérité (masquer tout ce qui n'est pas brûlé)
    burned_mask = dnbr.gt(0.1)
    dnbr_masked = dnbr.updateMask(burned_mask)
    dnbr_severity = dnbr_masked.visualize(
        min=0.1,
        max=0.8,
        palette=["#ffff00", "#ffa500", "#ff4500", "#cc0000", "#660000"],
    )
    severity_map = dnbr_severity.getMapId()
    tiles["burned"] = severity_map["tile_fetcher"].url_format

    # Statistiques sur la zone brûlée
    stats = _compute_burn_stats(dnbr, region)

    # Métadonnées des images
    pre_date = pre_image.date().format("YYYY-MM-dd").getInfo()
    post_date = post_image.date().format("YYYY-MM-dd").getInfo()

    return {
        "tiles": tiles,
        "stats": stats,
        "pre_date": pre_date,
        "post_date": post_date,
        "center": [lat, lon, buffer_km],
    }


def _compute_burn_stats(dnbr, region):
    """
    Calcule les statistiques de surface brûlée à partir du dNBR.
    """
    # Pixel size Sentinel-2 = 20m pour B8/B12
    pixel_area = ee.Image.pixelArea()

    stats = {}

    # Surface totale de la zone
    total_area = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=20,
        maxPixels=1e8,
    ).get("area")

    total_ha = ee.Number(total_area).divide(10000)

    # Surface brûlée (dNBR > 0.1)
    burned_mask = dnbr.gt(0.1)
    burned_area = pixel_area.updateMask(burned_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=20,
        maxPixels=1e8,
    ).get("area")

    burned_ha = ee.Number(burned_area).divide(10000)

    # Surface sévèrement brûlée (dNBR > 0.44)
    severe_mask = dnbr.gt(0.44)
    severe_area = pixel_area.updateMask(severe_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=20,
        maxPixels=1e8,
    ).get("area")

    severe_ha = ee.Number(severe_area).divide(10000)

    try:
        stats = {
            "total_ha": round(total_ha.getInfo(), 1),
            "burned_ha": round(burned_ha.getInfo(), 1),
            "severe_ha": round(severe_ha.getInfo(), 1),
        }
    except Exception:
        stats = {"total_ha": 0, "burned_ha": 0, "severe_ha": 0}

    return stats


def get_ndvi_batch(fires_df):
    """
    Calcule le NDVI MODIS (MOD13Q1, composite 16j, 250m) pour tous les points
    de feu via GEE. Groupé par mois pour minimiser les appels GEE.

    Returns: dict {DataFrame_index: ndvi_float}
    """
    import pandas as pd
    from collections import defaultdict
    import calendar

    init_gee()

    df = fires_df.copy()
    df["_date"] = pd.to_datetime(df["acq_date"]).dt.date

    ndvi_col = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
    results = {}

    # Grouper par mois → 1 image NDVI moyenne par mois
    month_groups = defaultdict(list)
    for idx, row in df.iterrows():
        d = row["_date"]
        month_groups[(d.year, d.month)].append((idx, row))

    BATCH_SIZE = 500  # Limite GEE sampleRegions

    for (year, month), fire_rows in month_groups.items():
        last_day = calendar.monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{last_day}"

        image = ndvi_col.filterDate(start, end).mean()

        # Traiter par batchs de 500 points
        for batch_start in range(0, len(fire_rows), BATCH_SIZE):
            batch = fire_rows[batch_start: batch_start + BATCH_SIZE]
            features = [
                ee.Feature(
                    ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])]),
                    {"fire_idx": str(idx)},
                )
                for idx, row in batch
            ]
            fc = ee.FeatureCollection(features)
            try:
                sampled = image.sampleRegions(collection=fc, scale=250, geometries=False)
                for feat in sampled.getInfo().get("features", []):
                    idx = feat["properties"].get("fire_idx")
                    raw = feat["properties"].get("NDVI")
                    if idx is not None and raw is not None:
                        results[int(idx)] = round(float(raw) * 0.0001, 3)
            except Exception:
                pass  # GEE échec → skip ce batch

    return results
