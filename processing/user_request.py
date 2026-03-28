from processing.geometry import geometry_from_location
from processing.time import parse_date

class UserRequest:
    def __init__(self, location, start_date, end_date):
        self.location = location
        self.geometry = geometry_from_location(location)
        self.start_date = parse_date(start_date)
        self.end_date = parse_date(end_date)

        if self.start_date > self.end_date:
            raise ValueError("La date de début doit être avant la date de fin")
