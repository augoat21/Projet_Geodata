"""
Module pour charger les données FIRMS depuis les fichiers CSV locaux
Structure attendue : data_sources/fires/historical/{year}/viirs-snpp_{year}_{country}.csv

Les pays sont détectés automatiquement à partir des noms de fichiers.
Le filtrage se fait par bounding box géographique.
"""

import pandas as pd
from pathlib import Path
import streamlit as st
import re


# Bounding boxes des pays (coordonnées géographiques)
# Utilisé pour filtrer les données par zone géographique
COUNTRY_BBOX = {
    "Afghanistan": {"lat_min": 29.0, "lat_max": 39.0, "lon_min": 60.0, "lon_max": 75.0},
    "Albania": {"lat_min": 39.5, "lat_max": 42.7, "lon_min": 19.0, "lon_max": 21.1},
    "Algeria": {"lat_min": 19.0, "lat_max": 37.1, "lon_min": -8.7, "lon_max": 12.0},
    "Angola": {"lat_min": -18.1, "lat_max": -4.4, "lon_min": 11.7, "lon_max": 24.1},
    "Argentina": {"lat_min": -55.1, "lat_max": -21.8, "lon_min": -73.6, "lon_max": -53.6},
    "Armenia": {"lat_min": 38.8, "lat_max": 41.3, "lon_min": 43.4, "lon_max": 46.6},
    "Australia": {"lat_min": -44.0, "lat_max": -10.0, "lon_min": 112.0, "lon_max": 154.0},
    "Austria": {"lat_min": 46.4, "lat_max": 49.0, "lon_min": 9.5, "lon_max": 17.2},
    "Azerbaijan": {"lat_min": 38.4, "lat_max": 41.9, "lon_min": 44.8, "lon_max": 50.4},
    "Bangladesh": {"lat_min": 20.7, "lat_max": 26.6, "lon_min": 88.0, "lon_max": 92.7},
    "Belarus": {"lat_min": 51.3, "lat_max": 56.2, "lon_min": 23.2, "lon_max": 32.8},
    "Belgium": {"lat_min": 49.5, "lat_max": 51.5, "lon_min": 2.5, "lon_max": 6.4},
    "Belize": {"lat_min": 15.9, "lat_max": 18.5, "lon_min": -89.2, "lon_max": -87.5},
    "Benin": {"lat_min": 6.2, "lat_max": 12.4, "lon_min": 0.8, "lon_max": 3.9},
    "Bolivia": {"lat_min": -22.9, "lat_max": -9.7, "lon_min": -69.6, "lon_max": -57.5},
    "Bosnia and Herzegovina": {"lat_min": 42.6, "lat_max": 45.3, "lon_min": 15.7, "lon_max": 19.6},
    "Botswana": {"lat_min": -27.0, "lat_max": -17.8, "lon_min": 19.9, "lon_max": 29.4},
    "Brazil": {"lat_min": -34.0, "lat_max": 6.0, "lon_min": -74.0, "lon_max": -34.0},
    "Bulgaria": {"lat_min": 41.2, "lat_max": 44.2, "lon_min": 22.4, "lon_max": 28.6},
    "Burkina Faso": {"lat_min": 9.4, "lat_max": 15.1, "lon_min": -5.5, "lon_max": 2.4},
    "Burundi": {"lat_min": -4.5, "lat_max": -2.3, "lon_min": 29.0, "lon_max": 30.9},
    "Cambodia": {"lat_min": 10.4, "lat_max": 14.7, "lon_min": 102.3, "lon_max": 107.6},
    "Cameroon": {"lat_min": 1.7, "lat_max": 13.1, "lon_min": 8.5, "lon_max": 16.2},
    "Canada": {"lat_min": 42.0, "lat_max": 84.0, "lon_min": -141.0, "lon_max": -52.0},
    "Central African Republic": {"lat_min": 2.2, "lat_max": 11.0, "lon_min": 14.4, "lon_max": 27.5},
    "Chad": {"lat_min": 7.4, "lat_max": 23.5, "lon_min": 13.5, "lon_max": 24.0},
    "Chile": {"lat_min": -56.0, "lat_max": -17.5, "lon_min": -75.7, "lon_max": -66.4},
    "China": {"lat_min": 18.2, "lat_max": 53.6, "lon_min": 73.5, "lon_max": 134.8},
    "Colombia": {"lat_min": -4.2, "lat_max": 13.4, "lon_min": -79.0, "lon_max": -66.9},
    "Congo": {"lat_min": -5.0, "lat_max": 3.7, "lon_min": 11.2, "lon_max": 18.6},
    "Congo (Democratic Republic)": {"lat_min": -13.5, "lat_max": 5.4, "lon_min": 12.2, "lon_max": 31.3},
    "Costa Rica": {"lat_min": 8.0, "lat_max": 11.2, "lon_min": -85.9, "lon_max": -82.6},
    "Croatia": {"lat_min": 42.4, "lat_max": 46.6, "lon_min": 13.5, "lon_max": 19.4},
    "Cuba": {"lat_min": 19.8, "lat_max": 23.3, "lon_min": -85.0, "lon_max": -74.1},
    "Cyprus": {"lat_min": 34.6, "lat_max": 35.7, "lon_min": 32.3, "lon_max": 34.6},
    "Czech Republic": {"lat_min": 48.6, "lat_max": 51.1, "lon_min": 12.1, "lon_max": 18.9},
    "Denmark": {"lat_min": 54.6, "lat_max": 57.8, "lon_min": 8.1, "lon_max": 15.2},
    "Dominican Republic": {"lat_min": 17.5, "lat_max": 20.0, "lon_min": -72.0, "lon_max": -68.3},
    "Ecuador": {"lat_min": -5.0, "lat_max": 1.7, "lon_min": -81.1, "lon_max": -75.2},
    "Egypt": {"lat_min": 22.0, "lat_max": 31.7, "lon_min": 24.7, "lon_max": 36.9},
    "El Salvador": {"lat_min": 13.2, "lat_max": 14.5, "lon_min": -90.1, "lon_max": -87.7},
    "Eritrea": {"lat_min": 12.4, "lat_max": 18.0, "lon_min": 36.4, "lon_max": 43.1},
    "Estonia": {"lat_min": 57.5, "lat_max": 59.7, "lon_min": 21.8, "lon_max": 28.2},
    "Ethiopia": {"lat_min": 3.4, "lat_max": 15.0, "lon_min": 33.0, "lon_max": 48.0},
    "Finland": {"lat_min": 59.8, "lat_max": 70.1, "lon_min": 20.6, "lon_max": 31.6},
    "France": {"lat_min": 41.0, "lat_max": 51.5, "lon_min": -5.5, "lon_max": 10.0},
    "Gabon": {"lat_min": -3.9, "lat_max": 2.3, "lon_min": 8.7, "lon_max": 14.5},
    "Georgia": {"lat_min": 41.0, "lat_max": 43.6, "lon_min": 40.0, "lon_max": 46.7},
    "Germany": {"lat_min": 47.0, "lat_max": 55.5, "lon_min": 5.5, "lon_max": 15.5},
    "Ghana": {"lat_min": 4.7, "lat_max": 11.2, "lon_min": -3.3, "lon_max": 1.2},
    "Greece": {"lat_min": 34.0, "lat_max": 42.0, "lon_min": 19.0, "lon_max": 29.0},
    "Guatemala": {"lat_min": 13.7, "lat_max": 17.8, "lon_min": -92.2, "lon_max": -88.2},
    "Guinea": {"lat_min": 7.2, "lat_max": 12.7, "lon_min": -15.1, "lon_max": -7.6},
    "Honduras": {"lat_min": 13.0, "lat_max": 16.5, "lon_min": -89.4, "lon_max": -83.2},
    "Hungary": {"lat_min": 45.7, "lat_max": 48.6, "lon_min": 16.1, "lon_max": 22.9},
    "Iceland": {"lat_min": 63.4, "lat_max": 66.6, "lon_min": -24.5, "lon_max": -13.5},
    "India": {"lat_min": 6.7, "lat_max": 35.5, "lon_min": 68.2, "lon_max": 97.4},
    "Indonesia": {"lat_min": -11.0, "lat_max": 6.1, "lon_min": 95.0, "lon_max": 141.0},
    "Iran": {"lat_min": 25.1, "lat_max": 39.8, "lon_min": 44.0, "lon_max": 63.3},
    "Iraq": {"lat_min": 29.1, "lat_max": 37.4, "lon_min": 38.8, "lon_max": 48.6},
    "Ireland": {"lat_min": 51.4, "lat_max": 55.4, "lon_min": -10.5, "lon_max": -6.0},
    "Israel": {"lat_min": 29.5, "lat_max": 33.3, "lon_min": 34.3, "lon_max": 35.9},
    "Italy": {"lat_min": 36.0, "lat_max": 47.5, "lon_min": 6.0, "lon_max": 19.0},
    "Ivory Coast": {"lat_min": 4.4, "lat_max": 10.7, "lon_min": -8.6, "lon_max": -2.5},
    "Japan": {"lat_min": 24.4, "lat_max": 45.5, "lon_min": 123.0, "lon_max": 146.0},
    "Jordan": {"lat_min": 29.2, "lat_max": 33.4, "lon_min": 34.9, "lon_max": 39.3},
    "Kazakhstan": {"lat_min": 40.6, "lat_max": 55.4, "lon_min": 46.5, "lon_max": 87.3},
    "Kenya": {"lat_min": -4.7, "lat_max": 5.0, "lon_min": 33.9, "lon_max": 41.9},
    "Laos": {"lat_min": 13.9, "lat_max": 22.5, "lon_min": 100.1, "lon_max": 107.7},
    "Latvia": {"lat_min": 55.7, "lat_max": 58.1, "lon_min": 21.0, "lon_max": 28.2},
    "Lebanon": {"lat_min": 33.1, "lat_max": 34.7, "lon_min": 35.1, "lon_max": 36.6},
    "Libya": {"lat_min": 19.5, "lat_max": 33.2, "lon_min": 9.3, "lon_max": 25.2},
    "Lithuania": {"lat_min": 53.9, "lat_max": 56.5, "lon_min": 21.0, "lon_max": 26.8},
    "Madagascar": {"lat_min": -25.6, "lat_max": -11.9, "lon_min": 43.2, "lon_max": 50.5},
    "Malawi": {"lat_min": -17.1, "lat_max": -9.4, "lon_min": 32.7, "lon_max": 35.9},
    "Malaysia": {"lat_min": 0.9, "lat_max": 7.4, "lon_min": 99.6, "lon_max": 119.3},
    "Mali": {"lat_min": 10.2, "lat_max": 25.0, "lon_min": -12.2, "lon_max": 4.3},
    "Mexico": {"lat_min": 14.5, "lat_max": 32.7, "lon_min": -118.4, "lon_max": -86.7},
    "Mongolia": {"lat_min": 41.6, "lat_max": 52.1, "lon_min": 87.8, "lon_max": 119.9},
    "Morocco": {"lat_min": 27.7, "lat_max": 35.9, "lon_min": -13.2, "lon_max": -1.0},
    "Mozambique": {"lat_min": -26.9, "lat_max": -10.5, "lon_min": 30.2, "lon_max": 40.8},
    "Myanmar": {"lat_min": 9.8, "lat_max": 28.5, "lon_min": 92.2, "lon_max": 101.2},
    "Namibia": {"lat_min": -29.0, "lat_max": -17.0, "lon_min": 11.7, "lon_max": 25.3},
    "Nepal": {"lat_min": 26.4, "lat_max": 30.4, "lon_min": 80.1, "lon_max": 88.2},
    "Netherlands": {"lat_min": 50.8, "lat_max": 53.5, "lon_min": 3.4, "lon_max": 7.2},
    "New Zealand": {"lat_min": -47.3, "lat_max": -34.4, "lon_min": 166.4, "lon_max": 178.6},
    "Nicaragua": {"lat_min": 10.7, "lat_max": 15.0, "lon_min": -87.7, "lon_max": -83.1},
    "Niger": {"lat_min": 11.7, "lat_max": 23.5, "lon_min": 0.2, "lon_max": 16.0},
    "Nigeria": {"lat_min": 4.3, "lat_max": 13.9, "lon_min": 2.7, "lon_max": 14.7},
    "North Korea": {"lat_min": 37.7, "lat_max": 43.0, "lon_min": 124.3, "lon_max": 130.7},
    "Norway": {"lat_min": 58.0, "lat_max": 71.2, "lon_min": 4.6, "lon_max": 31.1},
    "Pakistan": {"lat_min": 23.7, "lat_max": 37.1, "lon_min": 60.9, "lon_max": 77.8},
    "Panama": {"lat_min": 7.2, "lat_max": 9.6, "lon_min": -83.1, "lon_max": -77.2},
    "Papua New Guinea": {"lat_min": -10.7, "lat_max": -1.3, "lon_min": 141.0, "lon_max": 156.0},
    "Paraguay": {"lat_min": -27.6, "lat_max": -19.3, "lon_min": -62.6, "lon_max": -54.3},
    "Peru": {"lat_min": -18.4, "lat_max": -0.0, "lon_min": -81.3, "lon_max": -68.7},
    "Philippines": {"lat_min": 4.6, "lat_max": 21.1, "lon_min": 116.9, "lon_max": 126.6},
    "Poland": {"lat_min": 49.0, "lat_max": 54.8, "lon_min": 14.1, "lon_max": 24.2},
    "Portugal": {"lat_min": 36.5, "lat_max": 42.5, "lon_min": -10.0, "lon_max": -6.0},
    "Romania": {"lat_min": 43.6, "lat_max": 48.3, "lon_min": 20.3, "lon_max": 29.7},
    "Russia": {"lat_min": 41.2, "lat_max": 82.0, "lon_min": 19.6, "lon_max": -169.0},
    "Rwanda": {"lat_min": -2.8, "lat_max": -1.1, "lon_min": 29.0, "lon_max": 30.9},
    "Saudi Arabia": {"lat_min": 16.4, "lat_max": 32.2, "lon_min": 34.5, "lon_max": 55.7},
    "Senegal": {"lat_min": 12.3, "lat_max": 16.7, "lon_min": -17.5, "lon_max": -11.4},
    "Serbia": {"lat_min": 42.2, "lat_max": 46.2, "lon_min": 18.8, "lon_max": 23.0},
    "Sierra Leone": {"lat_min": 6.9, "lat_max": 10.0, "lon_min": -13.3, "lon_max": -10.3},
    "Somalia": {"lat_min": -1.7, "lat_max": 12.0, "lon_min": 41.0, "lon_max": 51.4},
    "South Africa": {"lat_min": -35.0, "lat_max": -22.1, "lon_min": 16.5, "lon_max": 32.9},
    "South Korea": {"lat_min": 33.1, "lat_max": 38.6, "lon_min": 124.6, "lon_max": 130.0},
    "South Sudan": {"lat_min": 3.5, "lat_max": 12.2, "lon_min": 24.1, "lon_max": 35.9},
    "Spain": {"lat_min": 35.0, "lat_max": 44.0, "lon_min": -10.0, "lon_max": 5.0},
    "Sri Lanka": {"lat_min": 5.9, "lat_max": 9.8, "lon_min": 79.7, "lon_max": 81.9},
    "Sudan": {"lat_min": 8.7, "lat_max": 22.2, "lon_min": 21.8, "lon_max": 38.6},
    "Sweden": {"lat_min": 55.3, "lat_max": 69.1, "lon_min": 11.1, "lon_max": 24.2},
    "Switzerland": {"lat_min": 45.8, "lat_max": 47.8, "lon_min": 6.0, "lon_max": 10.5},
    "Syria": {"lat_min": 32.3, "lat_max": 37.3, "lon_min": 35.7, "lon_max": 42.4},
    "Taiwan": {"lat_min": 21.9, "lat_max": 25.3, "lon_min": 120.0, "lon_max": 122.0},
    "Tanzania": {"lat_min": -11.7, "lat_max": -1.0, "lon_min": 29.3, "lon_max": 40.4},
    "Thailand": {"lat_min": 5.6, "lat_max": 20.5, "lon_min": 97.4, "lon_max": 105.6},
    "Togo": {"lat_min": 6.1, "lat_max": 11.1, "lon_min": -0.1, "lon_max": 1.8},
    "Tunisia": {"lat_min": 30.2, "lat_max": 37.4, "lon_min": 7.5, "lon_max": 11.6},
    "Turkey": {"lat_min": 36.0, "lat_max": 42.1, "lon_min": 26.0, "lon_max": 44.8},
    "Uganda": {"lat_min": -1.5, "lat_max": 4.2, "lon_min": 29.6, "lon_max": 35.0},
    "Ukraine": {"lat_min": 44.4, "lat_max": 52.4, "lon_min": 22.1, "lon_max": 40.2},
    "United Arab Emirates": {"lat_min": 22.6, "lat_max": 26.1, "lon_min": 51.6, "lon_max": 56.4},
    "United Kingdom": {"lat_min": 49.5, "lat_max": 61.0, "lon_min": -8.5, "lon_max": 2.0},
    "United States": {"lat_min": 24.0, "lat_max": 50.0, "lon_min": -125.0, "lon_max": -66.0},
    "Uruguay": {"lat_min": -35.0, "lat_max": -30.1, "lon_min": -58.4, "lon_max": -53.1},
    "Uzbekistan": {"lat_min": 37.2, "lat_max": 45.6, "lon_min": 56.0, "lon_max": 73.1},
    "Venezuela": {"lat_min": 0.6, "lat_max": 12.2, "lon_min": -73.4, "lon_max": -59.8},
    "Vietnam": {"lat_min": 8.6, "lat_max": 23.4, "lon_min": 102.1, "lon_max": 109.5},
    "Yemen": {"lat_min": 12.1, "lat_max": 19.0, "lon_min": 42.5, "lon_max": 54.5},
    "Zambia": {"lat_min": -18.1, "lat_max": -8.2, "lon_min": 22.0, "lon_max": 33.7},
    "Zimbabwe": {"lat_min": -22.4, "lat_max": -15.6, "lon_min": 25.2, "lon_max": 33.1},
}


def _extract_country_from_filename(filename):
    """
    Extrait le nom du pays depuis le nom de fichier.
    Format attendu : viirs-snpp_ANNÉE_NomDuPays.csv
    Exemples :
        viirs-snpp_2012_Afghanistan.csv → Afghanistan
        viirs-snpp_2020_United States.csv → United States
        viirs-snpp_2024_South Africa.csv → South Africa
    """
    # Retirer l'extension
    name = filename.replace('.csv', '')
    
    # Pattern : viirs-snpp_YYYY_PAYS ou autre-prefix_YYYY_PAYS
    # On cherche tout ce qui est après le 2e underscore qui suit l'année
    match = re.match(r'^[^_]+_(\d{4})_(.+)$', name)
    if match:
        return match.group(2).strip()
    
    # Fallback : essayer de trouver après le dernier _YYYY_
    match = re.match(r'.*_(\d{4})_(.+)$', name)
    if match:
        return match.group(2).strip()
    
    return None


def get_available_countries():
    """
    Retourne la liste des pays disponibles en scannant les noms de fichiers CSV.
    Détecte automatiquement les pays depuis le format : viirs-snpp_ANNÉE_Pays.csv
    """
    base_path = Path(__file__).parent / "historical"
    
    if not base_path.exists():
        st.error(f"❌ Le dossier {base_path} n'existe pas")
        return []
    
    countries = set()
    
    # Scanner tous les fichiers CSV dans tous les dossiers d'années
    for year_folder in base_path.iterdir():
        if year_folder.is_dir() and year_folder.name.isdigit():
            for csv_file in year_folder.glob("*.csv"):
                country = _extract_country_from_filename(csv_file.name)
                if country:
                    countries.add(country)
    
    if not countries:
        st.warning("⚠️ Aucun pays détecté dans les noms de fichiers CSV.")
        st.info("Format attendu : `viirs-snpp_ANNÉE_NomPays.csv` dans les sous-dossiers de `historical/`")
        return []
    
    return sorted(list(countries))


def _get_files_for_country(country, years):
    """
    Retourne la liste des fichiers CSV correspondant à un pays et une plage d'années.
    Recherche par nom de fichier.
    """
    base_path = Path(__file__).parent / "historical"
    files = []
    
    for year in years:
        year_folder = base_path / str(year)
        if not year_folder.exists():
            continue
        
        for csv_file in year_folder.glob("*.csv"):
            extracted_country = _extract_country_from_filename(csv_file.name)
            if extracted_country and extracted_country.lower() == country.lower():
                files.append(csv_file)
    
    return files


def load_fires(country, start_date, end_date):
    """
    Charge les feux depuis les fichiers CSV locaux.
    
    1. Trouve les fichiers CSV correspondant au pays (par nom de fichier)
    2. Filtre par bounding box géographique si disponible
    3. Filtre par dates
    
    Args:
        country: Nom du pays (str)
        start_date: Date de début (date object)
        end_date: Date de fin (date object)
    
    Returns:
        DataFrame avec les feux détectés
    """
    base_path = Path(__file__).parent / "historical"
    
    if not base_path.exists():
        st.error(f"❌ Le dossier {base_path} n'existe pas")
        return pd.DataFrame()
    
    # Déterminer les années concernées
    years = range(start_date.year, end_date.year + 1)
    
    # Trouver les fichiers CSV pour ce pays
    csv_files = _get_files_for_country(country, years)
    
    if not csv_files:
        st.warning(f"Aucun fichier CSV trouvé pour **{country}** entre {start_date.year} et {end_date.year}")
        st.info(f"📁 Fichiers cherchés dans : `{base_path}`")
        return pd.DataFrame()
    
    # Récupérer la bounding box si disponible
    bbox = COUNTRY_BBOX.get(country)
    
    all_fires = []
    total_loaded = 0
    total_filtered = 0
    
    with st.spinner(f"📦 Chargement des données pour {country} ({len(csv_files)} fichier(s))..."):
        progress_bar = st.progress(0)
        
        for idx, csv_file in enumerate(csv_files):
            try:
                # Charger le CSV
                df = pd.read_csv(csv_file)
                total_loaded += len(df)
                
                # Filtrer par bounding box si disponible
                if bbox and 'latitude' in df.columns and 'longitude' in df.columns:
                    df = df[
                        (df['latitude'] >= bbox['lat_min']) &
                        (df['latitude'] <= bbox['lat_max']) &
                        (df['longitude'] >= bbox['lon_min']) &
                        (df['longitude'] <= bbox['lon_max'])
                    ]
                
                # Convertir et filtrer les dates
                if 'acq_date' in df.columns:
                    df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
                    df = df[
                        (df['acq_date'].dt.date >= start_date) &
                        (df['acq_date'].dt.date <= end_date)
                    ]
                
                if not df.empty:
                    total_filtered += len(df)
                    all_fires.append(df)
            
            except Exception as e:
                st.warning(f"⚠️ Erreur lecture {csv_file.name}: {str(e)}")
                continue
            
            progress_bar.progress((idx + 1) / len(csv_files))
        
        progress_bar.empty()
    
    # Combiner tous les DataFrames
    if not all_fires:
        st.warning(f"Aucun feu trouvé pour {country} entre {start_date} et {end_date}")
        st.info(f"📊 Lignes chargées : {total_loaded:,} | Après filtrage : 0")
        return pd.DataFrame()
    
    result = pd.concat(all_fires, ignore_index=True)
    
    # Garder les colonnes utiles
    expected_columns = [
        'latitude', 'longitude', 'bright_ti4', 'scan', 'track',
        'acq_date', 'acq_time', 'satellite', 'instrument',
        'confidence', 'version', 'bright_ti5', 'frp', 'daynight', 'type'
    ]
    
    available_columns = [col for col in expected_columns if col in result.columns]
    result = result[available_columns]
    
    # Supprimer les doublons
    result = result.drop_duplicates()
    
    st.success(f"✅ {len(result):,} feux chargés pour {country}")
    st.info(f"📊 {total_loaded:,} lignes lues → {len(result):,} feux dans la zone ({start_date} → {end_date})")
    
    return result


def get_data_info():
    """
    Retourne des informations sur les données disponibles
    """
    base_path = Path(__file__).parent / "historical"
    
    if not base_path.exists():
        return {
            'available': False,
            'years': [],
            'total_files': 0,
            'countries': []
        }
    
    years = []
    total_files = 0
    countries = set()
    
    for year_folder in base_path.iterdir():
        if year_folder.is_dir() and year_folder.name.isdigit():
            years.append(int(year_folder.name))
            for csv_file in year_folder.glob("*.csv"):
                total_files += 1
                country = _extract_country_from_filename(csv_file.name)
                if country:
                    countries.add(country)
    
    return {
        'available': True,
        'years': sorted(years),
        'total_files': total_files,
        'countries': sorted(countries),
        'path': str(base_path)
    }