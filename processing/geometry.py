from shapely.geometry import Point, box
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="projet_geodata")

def geometry_from_location(location_input):
    location_input = location_input.strip()

    if "," in location_input:
        parts = location_input.split(",")
        if len(parts) == 2:
            try:
                lat, lon = map(float, parts)
                return Point(lon, lat)
            except ValueError:
                pass

        if len(parts) == 4:
            try:
                lonmin, latmin, lonmax, latmax = map(float, parts)
                return box(lonmin, latmin, lonmax, latmax)
            except ValueError:
                pass

    location = geolocator.geocode(location_input)
    if location:
        return Point(location.longitude, location.latitude)

    raise ValueError("Lieu non reconnu")
