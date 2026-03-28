"""
Module pour accéder aux données FIRMS via l'API officielle
Utilise l'endpoint /api/area/ avec les bounding boxes des pays
"""

import pandas as pd
import requests
from io import StringIO
from datetime import timedelta
import streamlit as st
from data_sources.fires.firms import COUNTRY_BBOX

# Configuration
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api"


def get_available_countries():
    """
    Retourne la liste des pays disponibles via l'API FIRMS
    Inclut l'option "Monde entier"
    """
    return ["Monde entier"] + sorted(COUNTRY_BBOX.keys())


def _build_area_url(api_key, country, days):
    """
    Construit l'URL de l'API FIRMS area.
    Format: /api/area/csv/API_KEY/VIIRS_SNPP_NRT/AREA/DAYS
    AREA = "world" ou "west,south,east,north"
    """
    if country == "Monde entier":
        area = "world"
    else:
        bbox = COUNTRY_BBOX.get(country)
        if not bbox:
            return None
        area = f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}"

    return f"{FIRMS_API_BASE}/area/csv/{api_key}/VIIRS_SNPP_NRT/{area}/{days}"


def load_fires_api(country, start_date, end_date, api_key):
    """
    Charge les feux depuis l'API FIRMS pour un pays/monde et une période donnée
    """
    if not api_key or api_key == "VOTRE_CLE_API":
        st.error("⚠️ Clé API FIRMS non configurée. Voir les instructions ci-dessous.")
        st.info("""
        **Pour obtenir votre clé API gratuite :**
        1. Visitez : https://firms.modaps.eosdis.nasa.gov/api/area/
        2. Remplissez le formulaire (nom + email)
        3. Vous recevrez la clé par email immédiatement
        """)
        return pd.DataFrame()

    # Calculer le nombre de jours
    delta = end_date - start_date
    days = delta.days + 1

    # L'API FIRMS limite à 5 jours par requête
    if days > 5:
        return _load_fires_by_chunks(country, start_date, end_date, api_key, chunk_days=5)

    url = _build_area_url(api_key, country, days)
    if not url:
        st.error(f"Pays '{country}' non trouvé dans la liste des bounding boxes")
        return pd.DataFrame()

    try:
        with st.spinner(f"📡 Chargement des données depuis l'API FIRMS pour {country}..."):
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                text = response.text.strip()

                if "Invalid" in text or "Error" in text:
                    st.error(f"❌ Erreur API : {text}")
                    return pd.DataFrame()

                df = pd.read_csv(StringIO(response.text))

                if df.empty:
                    st.warning(f"Aucun feu détecté pour {country} durant cette période")
                    return pd.DataFrame()

                # Convertir les dates
                df['acq_date'] = pd.to_datetime(df['acq_date'])

                # Filtrer par date
                df = df[
                    (df['acq_date'].dt.date >= start_date) &
                    (df['acq_date'].dt.date <= end_date)
                ]

                # Réorganiser les colonnes
                columns_order = [
                    "latitude", "longitude", "bright_ti4", "scan", "track",
                    "acq_date", "acq_time", "satellite", "instrument",
                    "confidence", "version", "bright_ti5", "frp", "daynight", "type"
                ]
                available_columns = [col for col in columns_order if col in df.columns]
                df = df[available_columns]

                st.success(f"✅ {len(df)} feux chargés depuis l'API FIRMS")
                return df

            elif response.status_code == 401:
                st.error("❌ Clé API invalide. Vérifiez votre clé FIRMS.")
                return pd.DataFrame()

            else:
                st.error(f"❌ Erreur API: {response.status_code}")
                return pd.DataFrame()

    except requests.Timeout:
        st.error("⏱️ Timeout: L'API FIRMS met trop de temps à répondre. Réessayez avec une période plus courte.")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return pd.DataFrame()


def _load_fires_by_chunks(country, start_date, end_date, api_key, chunk_days=10):
    """
    Charge les feux par tranches pour les longues périodes
    """
    all_fires = []
    current_date = start_date

    progress_bar = st.progress(0)
    total_days = (end_date - start_date).days + 1

    while current_date <= end_date:
        chunk_end = min(current_date + timedelta(days=chunk_days - 1), end_date)
        days_in_chunk = (chunk_end - current_date).days + 1

        url = _build_area_url(api_key, country, days_in_chunk)
        if not url:
            break

        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                text = response.text.strip()
                if text and text != "Invalid API call.":
                    df = pd.read_csv(StringIO(response.text))

                    if not df.empty:
                        df['acq_date'] = pd.to_datetime(df['acq_date'])
                        df = df[
                            (df['acq_date'].dt.date >= current_date) &
                            (df['acq_date'].dt.date <= chunk_end)
                        ]
                        all_fires.append(df)

        except Exception as e:
            st.warning(f"Erreur pour la période {current_date} - {chunk_end}: {str(e)}")

        current_date = chunk_end + timedelta(days=1)
        progress = min((current_date - start_date).days / total_days, 1.0)
        progress_bar.progress(progress)

    progress_bar.empty()

    if all_fires:
        result = pd.concat(all_fires, ignore_index=True)

        columns_order = [
            "latitude", "longitude", "bright_ti4", "scan", "track",
            "acq_date", "acq_time", "satellite", "instrument",
            "confidence", "version", "bright_ti5", "frp", "daynight", "type"
        ]
        available_columns = [col for col in columns_order if col in result.columns]
        result = result[available_columns]

        return result

    return pd.DataFrame()


def test_api_key(api_key):
    """
    Teste si la clé API est valide
    """
    if not api_key or api_key == "VOTRE_CLE_API":
        return False

    # Test avec le monde entier, 1 jour
    url = f"{FIRMS_API_BASE}/area/csv/{api_key}/VIIRS_SNPP_NRT/world/1"

    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200 and "Invalid" not in response.text
    except:
        return False
