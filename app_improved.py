import streamlit as st
import datetime
import folium
from pathlib import Path
from folium import plugins
from streamlit_folium import st_folium
import pandas as pd
from data_sources.fires.firms import load_fires, get_available_countries
from data_sources.fires.firms_api import load_fires_api, get_available_countries as get_api_countries
from data_sources.satellite.sentinel2 import get_satellite_analysis, get_ndvi_batch
from data_sources.weather.meteo import fetch_weather_for_fires, wind_direction_label
from processing.fire_risk import compute_fire_risk, get_country_geojson, generate_risk_image

st.set_page_config(page_title="FIRMS Fires", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d32f2f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-archive {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .source-realtime {
        background: linear-gradient(135deg, #b71c1c 0%, #d32f2f 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header"> Catalogue mondial des feux (FIRMS)</h1>', unsafe_allow_html=True)

def obtenir_couleur(frp):
    """Détermine la couleur du marqueur selon la puissance du feu"""
    if pd.isna(frp):
        return 'gray'
    elif frp < 10:
        return 'yellow'
    elif frp < 50:
        return 'orange'
    elif frp < 100:
        return 'red'
    else:
        return 'darkred'

def formater_heure(acq_time):
    """Formate l'heure d'acquisition"""
    if pd.isna(acq_time):
        return "N/A"
    time_str = str(int(acq_time)).zfill(4)
    return f"{time_str[:2]}:{time_str[2:]}"

def creer_popup(row, country):
    """Crée le contenu HTML du popup"""
    heure = formater_heure(row.get('acq_time'))
    frp = row.get('frp', 0)

    html = f"""
    <div style="font-family: Arial; width: 280px;">
        <h4 style="margin: 0 0 10px 0; color: #d32f2f; border-bottom: 2px solid #d32f2f; padding-bottom: 5px;">
            Feu détecté
        </h4>
        <table style="width: 100%; font-size: 13px;">
            <tr style="background-color: #f5f5f5;">
                <td style="padding: 5px;"><strong>Pays:</strong></td>
                <td style="padding: 5px;">{country}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>Date:</strong></td>
                <td style="padding: 5px;">{row['acq_date']}</td>
            </tr>
            <tr style="background-color: #f5f5f5;">
                <td style="padding: 5px;"><strong>Heure:</strong></td>
                <td style="padding: 5px;">{heure}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>Puissance (FRP):</strong></td>
                <td style="padding: 5px;"><strong>{frp:.2f} MW</strong></td>
            </tr>
            <tr style="background-color: #f5f5f5;">
                <td style="padding: 5px;"><strong>Satellite:</strong></td>
                <td style="padding: 5px;">{row.get('satellite', 'N/A')}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>Instrument:</strong></td>
                <td style="padding: 5px;">{row.get('instrument', 'N/A')}</td>
            </tr>
            <tr style="background-color: #f5f5f5;">
                <td style="padding: 5px;"><strong>Jour/Nuit:</strong></td>
                <td style="padding: 5px;">{'Jour ' if row.get('daynight') == 'D' else 'Nuit '}</td>
            </tr>
            <tr>
                <td style="padding: 5px;"><strong>Coordonnées:</strong></td>
                <td style="padding: 5px;">{row['latitude']:.4f}, {row['longitude']:.4f}</td>
            </tr>
        </table>
        {_meteo_html(row)}
        <div style="margin-top: 8px; padding: 6px; background-color: #e3f2fd; border-left: 3px solid #1565c0; font-size: 11px;">
            Activez la couche <strong>Occupation des sols</strong> pour identifier le type de terrain
        </div>
    </div>
    """
    return html


def _meteo_html(row):
    """Génère le bloc HTML météo + NDVI si les données sont disponibles."""
    import pandas as pd

    ws = row.get("wind_speed")
    wd = row.get("wind_dir")
    hum = row.get("humidity")
    temp = row.get("temperature")
    ndvi = row.get("ndvi")

    # N'afficher le bloc que si les colonnes météo ont été chargées
    if "wind_speed" not in row.index:
        return ""

    rows_html = ""

    if pd.notna(ws) and ws is not None:
        dir_label = wind_direction_label(wd) if pd.notna(wd) and wd is not None else "N/A"
        rows_html += f"""
        <tr style="background-color: #e3f2fd;">
            <td style="padding: 5px;">&#127788; <strong>Vent:</strong></td>
            <td style="padding: 5px;">{float(ws):.1f} km/h &mdash; {dir_label}</td>
        </tr>"""
    else:
        rows_html += """
        <tr style="background-color: #e3f2fd;">
            <td style="padding: 5px;">&#127788; <strong>Vent:</strong></td>
            <td style="padding: 5px; color: #999; font-style: italic;">Non disponible</td>
        </tr>"""

    if pd.notna(hum) and hum is not None:
        rows_html += f"""
        <tr>
            <td style="padding: 5px;">&#128167; <strong>Humidité:</strong></td>
            <td style="padding: 5px;">{float(hum):.0f} %</td>
        </tr>"""
    else:
        rows_html += """
        <tr>
            <td style="padding: 5px;">&#128167; <strong>Humidité:</strong></td>
            <td style="padding: 5px; color: #999; font-style: italic;">Non disponible</td>
        </tr>"""

    if pd.notna(temp) and temp is not None:
        rows_html += f"""
        <tr style="background-color: #fff3e0;">
            <td style="padding: 5px;">&#127777; <strong>Température:</strong></td>
            <td style="padding: 5px;">{float(temp):.1f} °C</td>
        </tr>"""
    else:
        rows_html += """
        <tr style="background-color: #fff3e0;">
            <td style="padding: 5px;">&#127777; <strong>Température:</strong></td>
            <td style="padding: 5px; color: #999; font-style: italic;">Non disponible</td>
        </tr>"""

    if pd.notna(ndvi) and ndvi is not None:
        ndvi_val = float(ndvi)
        if ndvi_val < 0.1:
            ndvi_desc = "Sol nu / eau"
        elif ndvi_val < 0.3:
            ndvi_desc = "Végétation clairsemée"
        elif ndvi_val < 0.5:
            ndvi_desc = "Végétation modérée"
        else:
            ndvi_desc = "Végétation dense"
        rows_html += f"""
        <tr style="background-color: #e8f5e9;">
            <td style="padding: 5px;">&#127807; <strong>NDVI:</strong></td>
            <td style="padding: 5px;">{ndvi_val:.3f} &mdash; {ndvi_desc}</td>
        </tr>"""
    else:
        rows_html += """
        <tr style="background-color: #e8f5e9;">
            <td style="padding: 5px;">&#127807; <strong>NDVI:</strong></td>
            <td style="padding: 5px; color: #999; font-style: italic;">Non disponible</td>
        </tr>"""

    return f"""
    <table style="width: 100%; font-size: 13px; margin-top: 8px; border-top: 2px solid #1565c0;">
        <tr>
            <td colspan="2" style="padding: 6px 5px 3px 5px; font-weight: bold; color: #1565c0; font-size: 12px;">
                Météo &amp; Végétation au moment du feu
            </td>
        </tr>
        {rows_html}
    </table>"""


def creer_legende_esa():
    """
    Crée une légende HTML pour la couche ESA WorldCover 2021
    avec les couleurs exactes du dataset
    """
    legend_html = """
    <div id="esa-legend" style="
        position: fixed;
        bottom: 30px;
        left: 10px;
        z-index: 1000;
        background-color: white;
        border: 2px solid #333;
        border-radius: 6px;
        padding: 10px 14px;
        font-family: Arial, sans-serif;
        color: #000000;
        font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        max-width: 220px;
    ">
        <div style="font-weight: bold; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid #ccc; padding-bottom: 4px;">
            Occupation des sols
            <span style="font-size: 10px; color: #000000;">(ESA 2021)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#006400; border:1px solid #000000; margin-right:6px;"></span>
            Couverture arborée / Forêts
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#ffbb22; border:1px solid #000000; margin-right:6px;"></span>
            Zone Arbustive
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#ffff4c; border:1px solid #000000; margin-right:6px;"></span>
            Prairies
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#f096ff; border:1px solid #000000; margin-right:6px;"></span>
            Terres cultivées
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#fa0000; border:1px solid #000000; margin-right:6px;"></span>
            Zones bâties
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#b4b4b4; border:1px solid #000000; margin-right:6px;"></span>
            Végétation clairsemée
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#f0f0f0; border:1px solid #000000; margin-right:6px;"></span>
            Neige et glace permanentes
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#0064c8; border:1px solid #000000; margin-right:6px;"></span>
            Plans d'eau permanents
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#0096a0; border:1px solid #000000; margin-right:6px;"></span>
            Zones humides herbacées
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#00cf75; border:1px solid #000000; margin-right:6px;"></span>
            Mangroves
        </div>
        <div style="display: flex; align-items: center; margin: 3px 0;">
            <span style="display:inline-block; width:18px; height:14px; background-color:#fae6a0; border:1px solid #000000; margin-right:6px;"></span>
            Mousses et lichens
        </div>
    </div>
    """
    return legend_html


def creer_carte(fires_df, country, satellite_tiles=None, sat_center=None):
    """
    Crée la carte interactive avec les feux, les frontières et la couche ESA WorldCover
    + couches Sentinel-2 si disponibles
    """
    if fires_df.empty:
        return None

    if sat_center:
        lat_centre = sat_center[0]
        lon_centre = sat_center[1]
        buf = sat_center[2] if len(sat_center) > 2 else 10
        if buf <= 5:
            zoom = 14
        elif buf <= 10:
            zoom = 12
        elif buf <= 20:
            zoom = 11
        elif buf <= 30:
            zoom = 10
        else:
            zoom = 9
    else:
        lat_centre = fires_df['latitude'].mean()
        lon_centre = fires_df['longitude'].mean()
        zoom = 6

    m = folium.Map(
        location=[lat_centre, lon_centre],
        zoom_start=zoom,
        tiles=None
    )


    folium.TileLayer(
        tiles='OpenStreetMap', name='Carte standard',
        overlay=False, control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Image satellite',
        overlay=False, control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap', name='Carte topographique',
        overlay=False, control=True, max_zoom=17
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Terrain et relief',
        overlay=False, control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='CartoDB positron', name='Carte claire',
        overlay=False, control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite + étiquettes',
        overlay=False, control=True
    ).add_to(m)

    folium.WmsTileLayer(
        url='https://services.terrascope.be/wms/v2',
        layers='WORLDCOVER_2021_MAP',
        fmt='image/png',
        transparent=True,
        name='Occupation des sols (ESA WorldCover)',
        overlay=True,
        control=True,
        show=False,
        attr='ESA WorldCover 2021',
        opacity=0.7
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Frontières et noms',
        overlay=True,
        control=True,
        show=True,
        opacity=0.9
    ).add_to(m)

    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner-labels/{z}/{x}/{y}{r}.png',
        attr='Stamen', name='',
        overlay=True, control=False, opacity=0.7
    ).add_to(m)

    groupe_feux = folium.FeatureGroup(name='Feux détectés', show=True)
    groupe_feux.add_to(m)

    max_points = 2000
    if len(fires_df) > max_points:
        df_affichage = fires_df.sample(n=max_points, random_state=42)
        st.info(f"ℹAffichage de {max_points} points sur {len(fires_df)} total pour les performances")
    else:
        df_affichage = fires_df

    for idx, row in df_affichage.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        frp = row.get('frp', 0)

        couleur = obtenir_couleur(frp)
        popup_html = creer_popup(row, country)
        popup = folium.Popup(popup_html, max_width=300)

        heure = formater_heure(row.get('acq_time'))
        date_str = row['acq_date'].strftime('%Y-%m-%d') if isinstance(row['acq_date'], pd.Timestamp) else str(row['acq_date'])

        tooltip = f"<b>{frp:.1f} MW</b><br>📅 {date_str} {heure}"

        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=min(5 + frp / 20, 15),
            popup=popup,
            tooltip=tooltip,
            color='black',
            fillColor=couleur,
            fillOpacity=0.7,
            weight=1
        )
        marker.add_to(groupe_feux)

    if satellite_tiles and sat_center:
        folium.TileLayer(
            tiles=satellite_tiles["burned"],
            attr="Google Earth Engine - Sentinel-2",
            name="Zones brûlées",
            overlay=True, control=True, show=True, opacity=0.6
        ).add_to(m)

        burn_legend_html = """
        <div style="
            position: fixed;
            bottom: 30px;
            right: 10px;
            z-index: 1000;
            background-color: white;
            border: 2px solid #333;
            border-radius: 6px;
            padding: 10px 14px;
            font-family: Arial, sans-serif;
            color: #000000;
            font-size: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            max-width: 200px;
        ">
            <div style="font-weight: bold; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid #ccc; padding-bottom: 4px;">
                Sévérité des brûlures
            </div>
            <div style="display: flex; align-items: center; margin: 3px 0;">
                <span style="display:inline-block; width:18px; height:14px; background-color:#ffff00; border:1px solid #000; margin-right:6px;"></span>
                Faible
            </div>
            <div style="display: flex; align-items: center; margin: 3px 0;">
                <span style="display:inline-block; width:18px; height:14px; background-color:#ffa500; border:1px solid #000; margin-right:6px;"></span>
                Modérée
            </div>
            <div style="display: flex; align-items: center; margin: 3px 0;">
                <span style="display:inline-block; width:18px; height:14px; background-color:#ff4500; border:1px solid #000; margin-right:6px;"></span>
                Élevée
            </div>
            <div style="display: flex; align-items: center; margin: 3px 0;">
                <span style="display:inline-block; width:18px; height:14px; background-color:#cc0000; border:1px solid #000; margin-right:6px;"></span>
                Sévère
            </div>
            <div style="display: flex; align-items: center; margin: 3px 0;">
                <span style="display:inline-block; width:18px; height:14px; background-color:#660000; border:1px solid #000; margin-right:6px;"></span>
                Très sévère
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(burn_legend_html))

        buf_km = sat_center[2] if len(sat_center) > 2 else 10
        delta_lat = buf_km / 111.0
        delta_lon = buf_km / (111.0 * abs(max(0.1, __import__('math').cos(__import__('math').radians(sat_center[0])))))
        bounds = [
            [sat_center[0] - delta_lat, sat_center[1] - delta_lon],
            [sat_center[0] + delta_lat, sat_center[1] + delta_lon],
        ]
        folium.Rectangle(
            bounds=bounds,
            color="#ff6600",
            weight=3,
            fill=False,
            dash_array="10",
            tooltip=f"Zone analysée : {buf_km*2} x {buf_km*2} km",
        ).add_to(m)

    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    minimap = plugins.MiniMap(
        toggle_display=True,
        tile_layer=folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Minimap'
        )
    )
    m.add_child(minimap)

    # Plein écran
    plugins.Fullscreen(
        position='topleft', title='Plein écran',
        title_cancel='Quitter le plein écran', force_separate_button=True
    ).add_to(m)

    folium.plugins.MeasureControl(
        position='bottomleft',
        primary_length_unit='kilometers', secondary_length_unit='miles',
        primary_area_unit='hectares', secondary_area_unit='acres'
    ).add_to(m)

    m.get_root().html.add_child(folium.Element(creer_legende_esa()))

    return m



with st.sidebar:
    st.header("Navigation")
    app_mode = st.radio(
        "Mode",
        options=["Catalogue des feux", "Prédiction de risque"],
        index=0,
        help="**Catalogue** : Explorer les feux passés\n\n**Prédiction** : Score de risque futur"
    )

    st.markdown("---")
    st.header("Filtres de recherche")

    st.markdown("### Source de données")

    data_source = st.radio(
        "Choisissez la source",
        options=["Archives (2012-2024)", "Temps réel (5 derniers jours)"],
        index=0,
        help="**Archives** : Données CSV locales (2012-2024)\n\n**Temps réel** : API NASA FIRMS (5 derniers jours)"
    )

    st.markdown("---")

    if data_source == "Archives (2012-2024)":
        st.markdown('<div class="source-archive">Mode Archives<br><small>Données locales CSV · 2012 à 2024</small></div>', unsafe_allow_html=True)

        all_countries = get_available_countries()

        if not all_countries:
            st.error("❌ Aucun pays détecté dans les fichiers CSV")
            st.stop()

        country = st.selectbox(
            "Sélection du pays", all_countries,
            help="Pays détectés automatiquement depuis les noms de fichiers CSV"
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Date de début",
                value=datetime.date(2024, 1, 1),
                min_value=datetime.date(2012, 1, 1),
                max_value=datetime.date(2024, 12, 31),
                key="archive_start"
            )
        with col2:
            end_date = st.date_input(
                "Date de fin",
                value=datetime.date(2024, 12, 31),
                min_value=datetime.date(2012, 1, 1),
                max_value=datetime.date(2024, 12, 31),
                key="archive_end"
            )

        st.markdown("---")
        search_button = st.button("Rechercher dans les archives", type="primary", use_container_width=True)
    else:
        st.markdown('<div class="source-realtime">Mode Temps Réel<br><small>API NASA FIRMS · 5 derniers jours</small></div>', unsafe_allow_html=True)

        api_key = st.text_input(
            "Clé API FIRMS", type="password",
            help="Obtenez votre clé gratuite sur https://firms.modaps.eosdis.nasa.gov/api/area/"
        )

        if api_key:
            st.success("Clé API renseignée")
        else:
            st.warning("Entrez votre clé API FIRMS")
            st.markdown("""
            **Clé gratuite :**
            1. [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/api/area/)
            2. Formulaire (nom + email)
            3. Clé reçue par email
            """)

        all_api_countries = get_api_countries()

        country = st.selectbox(
            "Sélection du pays", all_api_countries,
            index=all_api_countries.index("France") if "France" in all_api_countries else 0,
            help="Tous les pays via l'API FIRMS + 'Monde entier'"
        )

        nb_jours = st.slider(
            "Nombre de jours", min_value=1, max_value=5, value=5,
            help="Nombre de jours en arrière (max 5, limite API FIRMS)"
        )

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=nb_jours - 1)

        st.info(f"Du **{start_date.strftime('%d/%m/%Y')}** au **{end_date.strftime('%d/%m/%Y')}**")

        st.markdown("---")
        search_button = st.button("Rechercher les feux récents", type="primary", use_container_width=True)

if 'fires' not in st.session_state:
    st.session_state.fires = None
    st.session_state.fires_country = None
    st.session_state.fires_source = None
    st.session_state.fires_start = None
    st.session_state.fires_end = None
if 'satellite_result' not in st.session_state:
    st.session_state.satellite_result = None
if 'meteo_loaded' not in st.session_state:
    st.session_state.meteo_loaded = False

if app_mode == "Catalogue des feux" and search_button:
    if not country:
        st.warning("Veuillez sélectionner un pays")
    elif start_date > end_date:
        st.warning("La date de début doit être avant la date de fin")
    else:
        if data_source == "Archives (2012-2024)":
            with st.spinner(f"Recherche dans les archives pour {country}..."):
                fires = load_fires(country=country, start_date=start_date, end_date=end_date)
        else:
            if not api_key:
                st.error("Veuillez entrer votre clé API FIRMS")
                st.stop()
            with st.spinner(f"Chargement temps réel pour {country}..."):
                fires = load_fires_api(country=country, start_date=start_date, end_date=end_date, api_key=api_key)

        if fires.empty:
            st.warning(f"Aucun feu détecté pour {country} entre {start_date} et {end_date}")
            st.session_state.fires = None
        else:
            if 'Pays' not in fires.columns:
                fires.insert(0, "Pays", country)
            st.session_state.fires = fires
            st.session_state.fires_country = country
            st.session_state.fires_source = data_source
            st.session_state.fires_start = start_date
            st.session_state.fires_end = end_date
            st.session_state.satellite_result = None
            st.session_state.meteo_loaded = False

if app_mode == "Catalogue des feux" and st.session_state.fires is not None:
    fires = st.session_state.fires
    country = st.session_state.fires_country
    s_start = st.session_state.fires_start
    s_end = st.session_state.fires_end

    jours_delta = (s_end - s_start).days + 1
    if st.session_state.fires_source == "Archives (2012-2024)":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 0.7rem 1.2rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <strong>Source : Archives locales</strong> · {country} · {s_start} → {s_end} · <strong>{len(fires):,} feux</strong> détectés
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #b71c1c, #d32f2f); color: white; padding: 0.7rem 1.2rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <strong>Source : API Temps réel NASA FIRMS</strong> · {country} · {jours_delta} jour(s) · <strong>{len(fires):,} feux</strong> détectés
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Statistiques")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total de feux", value=f"{len(fires):,}")
    with col2:
        if 'frp' in fires.columns:
            st.metric(label="Puissance moyenne", value=f"{fires['frp'].mean():.2f} MW")
        else:
            st.metric(label="Puissance moyenne", value="N/A")
    with col3:
        if 'frp' in fires.columns:
            st.metric(label="Puissance maximale", value=f"{fires['frp'].max():.2f} MW")
        else:
            st.metric(label="Puissance maximale", value="N/A")

    st.markdown("---")

    # --- Météo & NDVI ---
    with st.expander("Météo & NDVI au moment des feux", expanded=not st.session_state.meteo_loaded):
        if not st.session_state.meteo_loaded:
            st.info(
                "Enrichit chaque point de feu avec : **vitesse du vent**, "
                "**direction du vent**, **humidité de l'air**, **température** (Open-Meteo ERA5) "
                "et **NDVI** (MODIS 250m via GEE). Visible dans les popups de la carte."
            )
            nb_locs = len(
                st.session_state.fires[["latitude", "longitude"]]
                .round(0)
                .drop_duplicates()
            )
            st.caption(
                f"{len(st.session_state.fires):,} feux · "
                f"~{nb_locs} cellules météo uniques (grille 1°) · "
                f"~{len(st.session_state.fires['acq_date'].dt.to_period('M').unique() if hasattr(st.session_state.fires['acq_date'], 'dt') else [1])} mois NDVI"
            )
            meteo_btn = st.button("Charger météo & NDVI", type="primary", use_container_width=True)

            if meteo_btn:
                fires_work = st.session_state.fires.copy()

                # 1. Météo Open-Meteo
                progress_bar = st.progress(0, text="Chargement météo (Open-Meteo ERA5)...")
                def update_progress(p):
                    progress_bar.progress(min(p * 0.9, 0.9), text=f"Météo… {int(p*100)}%")

                fires_work = fetch_weather_for_fires(fires_work, progress_callback=update_progress)
                progress_bar.progress(0.9, text="Calcul NDVI MODIS (GEE)...")

                # 2. NDVI MODIS via GEE
                try:
                    ndvi_dict = get_ndvi_batch(fires_work)
                    fires_work["ndvi"] = fires_work.index.map(ndvi_dict)
                except Exception as e:
                    st.warning(f"NDVI non disponible : {e}")
                    fires_work["ndvi"] = None

                progress_bar.progress(1.0, text="Terminé !")
                progress_bar.empty()

                st.session_state.fires = fires_work
                st.session_state.meteo_loaded = True
                st.rerun()
        else:
            fires_check = st.session_state.fires
            ws_ok = fires_check["wind_speed"].notna().sum()
            hum_ok = fires_check["humidity"].notna().sum()
            ndvi_ok = fires_check["ndvi"].notna().sum() if "ndvi" in fires_check.columns else 0
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Points météo", f"{ws_ok:,} / {len(fires_check):,}")
            col_b.metric("Points humidité", f"{hum_ok:,} / {len(fires_check):,}")
            col_c.metric("Points NDVI", f"{ndvi_ok:,} / {len(fires_check):,}")
            st.success("Données météo & NDVI chargées — visibles dans les popups de la carte.")

    st.subheader("Carte interactive des feux")

    sat_tiles = st.session_state.satellite_result['tiles'] if st.session_state.satellite_result else None
    sat_center = st.session_state.satellite_result['center'] if st.session_state.satellite_result else None
    carte = creer_carte(fires, country, satellite_tiles=sat_tiles, sat_center=sat_center)

    if carte is not None:
        with st.expander("Légende des feux (puissance FRP)", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("🟡 **Faible** : < 10 MW")
            with col2:
                st.markdown("🟠 **Moyen** : 10-50 MW")
            with col3:
                st.markdown("🔴 **Élevé** : 50-100 MW")
            with col4:
                st.markdown("⚫ **Très élevé** : > 100 MW")

        st_folium(carte, width=1400, height=600, returned_objects=[])

        st.info("**Astuce** : Activez la couche **'Occupation des sols'** dans le menu en haut à droite de la carte pour voir le type de terrain (forêt, cultures, zones bâties...). La légende s'affiche en bas à gauche de la carte.")

    st.markdown("---")

    st.subheader("Analyse satellite Sentinel-2")
    st.markdown("Sélectionnez un feu pour analyser les zones brûlées avec des images Sentinel-2 (avant/après).")


    if 'frp' in fires.columns:
        top_fires = fires.nlargest(20, 'frp').reset_index(drop=True)
    else:
        top_fires = fires.head(20).reset_index(drop=True)

    fire_options = []
    for i, row in top_fires.iterrows():
        date_str = str(row['acq_date'])[:10]
        frp_val = row.get('frp', 0)
        fire_options.append(
            f"#{i+1} - {date_str} - FRP: {frp_val:.1f} MW - ({row['latitude']:.3f}, {row['longitude']:.3f})"
        )

    col_sel, col_buf = st.columns([3, 1])
    with col_sel:
        selected_fire_idx = st.selectbox(
            "Choisir un feu à analyser (top 20 par puissance)",
            range(len(fire_options)),
            format_func=lambda x: fire_options[x],
            key="sentinel_fire_select"
        )
    with col_buf:
        buffer_km = st.number_input(
            "Rayon d'analyse (km)", min_value=5, max_value=50, value=10,
            help="Zone autour du feu à analyser"
        )

    analyse_button = st.button("Lancer l'analyse Sentinel-2", type="primary", use_container_width=True)

    if analyse_button:
        selected_row = top_fires.iloc[selected_fire_idx]
        fire_lat = selected_row['latitude']
        fire_lon = selected_row['longitude']
        fire_date = str(selected_row['acq_date'])[:10]

        with st.spinner(f"Chargement des images Sentinel-2 autour de ({fire_lat:.3f}, {fire_lon:.3f})... Cela peut prendre 10-20 secondes."):
            try:
                result = get_satellite_analysis(
                    lat=fire_lat, lon=fire_lon,
                    fire_date=fire_date,
                    buffer_km=buffer_km
                )

                if result is None:
                    st.warning("Pas d'images Sentinel-2 disponibles pour ce feu (trop de nuages ou zone non couverte). Essayez un autre feu.")
                else:
                    st.session_state.satellite_result = result
                    st.rerun()

            except Exception as e:
                st.error(f"Erreur lors de l'analyse satellite : {str(e)}")

    if st.session_state.satellite_result:
        result = st.session_state.satellite_result
        stats = result['stats']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Zone analysée", f"{stats['total_ha']:.0f} ha")
        with col2:
            st.metric("Surface brûlée", f"{stats['burned_ha']:.0f} ha")
        with col3:
            st.metric("Brûlure sévère", f"{stats['severe_ha']:.0f} ha")

    st.markdown("---")

    with st.expander("Données détaillées", expanded=False):
        st.dataframe(fires, use_container_width=True, height=400)

        csv = fires.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les données (CSV)",
            data=csv,
            file_name=f"feux_{country.replace(' ', '_')}_{s_start}_{s_end}.csv",
            mime="text/csv",
        )

    st.subheader("Analyses")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Nombre de feux par jour**")
        fires_copy = fires.copy()
        fires_copy['acq_date'] = pd.to_datetime(fires_copy['acq_date'])
        feux_par_jour = fires_copy.groupby(fires_copy['acq_date'].dt.date).size()
        st.line_chart(feux_par_jour)

    with col2:
        if 'frp' in fires.columns:
            st.write("**Distribution de la puissance (FRP)**")
            bins = [0, 10, 50, 100, 500, float('inf')]
            labels = ['0-10 MW', '10-50 MW', '50-100 MW', '100-500 MW', '500+ MW']
            frp_bins = pd.cut(fires['frp'].dropna(), bins=bins, labels=labels, right=False)
            frp_counts = frp_bins.value_counts().reindex(labels).fillna(0)
            st.bar_chart(frp_counts)

    with st.expander("Statistiques détaillées"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Par satellite:**")
            if 'satellite' in fires.columns:
                st.dataframe(
                    fires['satellite'].value_counts().reset_index().rename(
                        columns={'index': 'Satellite', 'satellite': 'Nombre'}
                    ),
                    hide_index=True
                )

        with col2:
            st.write("**Jour/Nuit:**")
            if 'daynight' in fires.columns:
                daynight_map = {'D': 'Jour', 'N': 'Nuit'}
                daynight_counts = fires['daynight'].map(daynight_map).value_counts()
                st.dataframe(
                    daynight_counts.reset_index().rename(
                        columns={'index': 'Période', 'daynight': 'Nombre'}
                    ),
                    hide_index=True
                )

elif app_mode == "Catalogue des feux":
    st.info("Utilisez le panneau latéral pour rechercher des feux")

    st.markdown("""
    ### Guide d'utilisation

    #### Deux sources de données disponibles

    | | **Archives** | **Temps réel** |
    |---|---|---|
    | **Période** | 2012 → 2024 | 7 derniers jours |
    | **Source** | Fichiers CSV locaux | API NASA FIRMS |
    | **Clé API** | Non requise | Requise (gratuite) |
    | **Pays** | Détectés depuis les CSV | Tous les pays + Monde entier |
    | **Usage** | Analyse historique | Surveillance en cours |

    #### Couches de carte disponibles

    - **Image satellite** · **Terrain et relief** · **Carte topographique**
    - **Satellite + étiquettes** · **Carte standard** · **Carte claire**
    - **Occupation des sols (ESA WorldCover 2021)** — avec légende intégrée
    - **Frontières et noms de pays** — traits apparents

    #### Légende des feux (puissance FRP)

    - 🟡 Faible (< 10 MW) · 🟠 Moyen (10-50 MW) · 🔴 Élevé (50-100 MW) · ⚫ Très élevé (> 100 MW)
    """)

# =============================================
# MODE PRÉDICTION DE RISQUE
# =============================================
if app_mode == "Prédiction de risque":
    st.subheader("Prédiction du risque de feu — France")

    from processing.fire_risk import _ml_available
    if not _ml_available:
        st.error(
            "Le modèle XGBoost n'est pas disponible. "
            "Lancez `python -m ml.build_dataset` puis `python -m ml.train_model` pour l'entraîner."
        )
        st.stop()

    st.markdown(
        "Prédiction basée sur un **modèle XGBoost** entraîné sur **61 490 observations** "
        "(feux FIRMS 2012-2024 + météo historique Open-Meteo) sur la **France métropolitaine**. "
        "Le score (0-100) représente la **probabilité de feu** estimée par le modèle."
    )

    # Charger les métriques du modèle
    import json
    _metrics_path = Path(__file__).resolve().parent / "ml" / "models" / "fire_risk_xgboost_metrics.json"
    if _metrics_path.exists():
        with open(_metrics_path) as _mf:
            _ml_metrics = json.load(_mf)

        with st.expander("Performances du modèle ML", expanded=False):
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("AUC-ROC", f"{_ml_metrics.get('auc_roc', 0):.4f}")
            col_m2.metric("Accuracy", "93%")
            col_m3.metric("Cross-validation", f"{_ml_metrics.get('cv_auc_mean', 0):.4f}")
            col_m4.metric("Données d'entraînement", f"{_ml_metrics.get('n_train', 0) + _ml_metrics.get('n_test', 0):,}")

            st.markdown("**Importance des features :**")
            import pandas as pd
            feat_imp = _ml_metrics.get("feature_importances", {})
            if feat_imp:
                feat_df = pd.DataFrame(
                    sorted(feat_imp.items(), key=lambda x: -float(x[1])),
                    columns=["Feature", "Importance"]
                )
                feat_df["Feature"] = feat_df["Feature"].replace({
                    "humidity_min": "Humidité min",
                    "month": "Mois",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                    "month_sin": "Mois (sin)",
                    "precip_sum": "Précipitations",
                    "temp_max": "Temp. max",
                    "day_of_year": "Jour de l'année",
                    "month_cos": "Mois (cos)",
                    "wind_max": "Vent max",
                })
                st.bar_chart(feat_df.set_index("Feature"), horizontal=True)

            st.markdown("**Matrice de confusion (test) :**")
            cm = _ml_metrics.get("confusion_matrix", [[0,0],[0,0]])
            col_cm1, col_cm2 = st.columns(2)
            col_cm1.metric("Vrais négatifs", f"{cm[0][0]:,}")
            col_cm1.metric("Faux négatifs", f"{cm[1][0]:,}")
            col_cm2.metric("Faux positifs", f"{cm[0][1]:,}")
            col_cm2.metric("Vrais positifs", f"{cm[1][1]:,}")

    # Légende des scores
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.markdown(
        '<div style="background:#c8e6c9;padding:8px;border-radius:5px;text-align:center;">'
        '<strong>0-30 : Faible</strong></div>', unsafe_allow_html=True
    )
    col_l2.markdown(
        '<div style="background:#ffe0b2;padding:8px;border-radius:5px;text-align:center;">'
        '<strong>30-60 : Modéré</strong></div>', unsafe_allow_html=True
    )
    col_l3.markdown(
        '<div style="background:#ffcdd2;padding:8px;border-radius:5px;text-align:center;">'
        '<strong>60-100 : Élevé</strong></div>', unsafe_allow_html=True
    )

    st.markdown("")

    # France uniquement (modèle entraîné sur ces données)
    pred_country = "France"
    st.info("Analyse limitée à la **France métropolitaine** (périmètre d'entraînement du modèle).")

    pred_days = st.slider("Nombre de jours de prévision", min_value=1, max_value=7, value=7, key="pred_days")

    resolution = st.select_slider(
        "Résolution de la grille",
        options=[0.5, 0.25],
        value=0.5,
        format_func=lambda x: f"{x}° — {'Standard (~rapide)' if x == 0.5 else 'Fine (~4x plus long)'}",
        key="pred_resolution",
    )

    from data_sources.fires.firms import COUNTRY_BBOX
    bbox = COUNTRY_BBOX.get(pred_country, {})
    if bbox:
        import numpy as np
        n_lats = len(np.arange(bbox["lat_min"], bbox["lat_max"] + resolution, resolution))
        n_lons = len(np.arange(bbox["lon_min"], bbox["lon_max"] + resolution, resolution))
        n_points = n_lats * n_lons
        st.caption(f"~{n_points} points de grille (résolution {resolution}°) sur {pred_days} jours")

    pred_btn = st.button("Calculer le risque", type="primary", use_container_width=True, key="pred_btn")

    if 'risk_df' not in st.session_state:
        st.session_state.risk_df = None
        st.session_state.risk_country = None

    if pred_btn:
        progress = st.progress(0, text="Chargement des prévisions météo...")

        def update_pred_progress(p):
            progress.progress(min(p, 1.0), text=f"Prévisions météo… {int(p*100)}%")

        risk_df = compute_fire_risk(pred_country, days=pred_days, resolution=resolution, progress_callback=update_pred_progress)
        progress.progress(1.0, text="Terminé !")
        progress.empty()

        st.session_state.risk_df = risk_df
        st.session_state.risk_country = pred_country
        st.rerun()

    if st.session_state.risk_df is not None and not st.session_state.risk_df.empty:
        risk_df = st.session_state.risk_df

        # Sélection du jour
        st.markdown("### Résultats")
        available_dates = sorted(risk_df['date'].unique())
        selected_date = st.select_slider(
            "Jour de prévision",
            options=available_dates,
            value=available_dates[0],
            key="pred_date_slider"
        )
        day_df = risk_df[risk_df['date'] == selected_date]

        # Statistiques du jour sélectionné
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Score moyen", f"{day_df['score'].mean():.1f} / 100")
        col_s2.metric("Score max", f"{day_df['score'].max():.1f} / 100")
        col_s3.metric("Points analysés", f"{len(day_df):,}")
        n_eleve = len(day_df[day_df['score'] >= 60])
        col_s4.metric("Zones à risque élevé", f"{n_eleve}")

        # Carte
        st.markdown(f"### Carte du risque — {selected_date}")
        bbox_data = COUNTRY_BBOX.get(st.session_state.risk_country, {})
        center_lat = (bbox_data.get("lat_min", 0) + bbox_data.get("lat_max", 0)) / 2
        center_lon = (bbox_data.get("lon_min", 0) + bbox_data.get("lon_max", 0)) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

        # Image interpolée du risque (fond coloré lisse)
        with st.spinner("Génération de la carte de risque..."):
            img_data, img_bounds = generate_risk_image(day_df, st.session_state.risk_country)

        if img_data and img_bounds:
            folium.raster_layers.ImageOverlay(
                image=img_data,
                bounds=img_bounds,
                opacity=0.65,
                name="Risque de feu",
                interactive=False,
            ).add_to(m)

        # Contour du pays
        country_geojson = get_country_geojson(st.session_state.risk_country)
        if country_geojson:
            folium.GeoJson(
                country_geojson,
                name="Contour du pays",
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "color": "#333",
                    "weight": 2.5,
                },
            ).add_to(m)

        # Marqueurs pour tooltip au survol
        for _, row in day_df.iterrows():
            score = row['score']
            tooltip_text = (
                f"<b>Score : {score:.0f}/100 — {row['label']}</b><br>"
                f"Temp. max : {row['temp_max']:.1f} °C<br>"
                f"Humidité min : {row['humidity_min']:.0f} %<br>"
                f"Vent max : {row['wind_max']:.1f} km/h"
            )
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=20,
                tooltip=tooltip_text,
                color="transparent",
                fillColor="transparent",
                fill=True,
                fillOpacity=0.01,
                weight=0,
            ).add_to(m)

        # Légende
        legend_html = """
        <div style="position:fixed;bottom:30px;right:10px;z-index:1000;background:white;
            border:2px solid #333;border-radius:6px;padding:10px 14px;font-family:Arial;font-size:12px;
            color:#000000;box-shadow:0 2px 8px rgba(0,0,0,0.3);">
            <div style="font-weight:bold;margin-bottom:6px;color:#000000;">Risque de feu</div>
            <div style="color:#000000;"><span style="display:inline-block;width:16px;height:12px;background:#00c853;margin-right:6px;"></span>Faible (0-30)</div>
            <div style="color:#000000;"><span style="display:inline-block;width:16px;height:12px;background:#ffd600;margin-right:6px;"></span>Modéré (30-60)</div>
            <div style="color:#000000;"><span style="display:inline-block;width:16px;height:12px;background:#dd2c00;margin-right:6px;"></span>Élevé (60-100)</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl().add_to(m)
        st_folium(m, width=1400, height=600, returned_objects=[])

        # Tableau détaillé
        with st.expander("Données détaillées", expanded=False):
            display_df = day_df[['latitude', 'longitude', 'temp_max', 'humidity_min', 'wind_max', 'score', 'label']].copy()
            display_df.columns = ['Latitude', 'Longitude', 'Temp. max (°C)', 'Humidité min (%)', 'Vent max (km/h)', 'Score', 'Risque']
            st.dataframe(display_df.sort_values('Score', ascending=False), use_container_width=True, hide_index=True)

        # Graphique évolution du risque
        st.markdown("### Évolution du risque moyen sur la période")
        daily_avg = risk_df.groupby('date')['score'].mean()
        st.line_chart(daily_avg, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Données: NASA FIRMS · Occupation des sols: ESA WorldCover 2021 · Développé avec Streamlit, Folium et Python</p>
    </div>
    """,
    unsafe_allow_html=True
)