"""
Minimal GTFS writer shared by the matrix builders (text_bndes).

A feed is assembled from plain lists of dicts and written as a zip. Times are seconds after midnight.
Synthetic lines (metro branches, ENMU projects) are timetabled with explicit trips at a fixed headway, not
frequencies.txt, so that r5py results are deterministic.
"""
import io
import zipfile

import pandas as pd

SERVICE_ID = 'WK'


def hms(seconds):
    seconds = int(round(seconds))
    return '%02d:%02d:%02d' % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


class Feed:
    def __init__(self, agency_id, agency_name, start_date, end_date):
        self.agency = [dict(agency_id=agency_id, agency_name=agency_name, agency_url='https://example.org',
                            agency_timezone='America/Sao_Paulo')]
        self.calendar = [dict(service_id=SERVICE_ID, monday=1, tuesday=1, wednesday=1, thursday=1, friday=1,
                              saturday=0, sunday=0, start_date=start_date, end_date=end_date)]
        self.agency_id = agency_id
        self.stops, self.routes, self.trips, self.stop_times = {}, [], [], []

    def add_stop(self, stop_id, lon, lat, name=''):
        self.stops.setdefault(stop_id, dict(stop_id=stop_id, stop_name=name or stop_id, stop_lat=lat, stop_lon=lon))

    def add_route(self, route_id, short_name, long_name, route_type):
        self.routes.append(dict(route_id=route_id, agency_id=self.agency_id, route_short_name=short_name,
                                route_long_name=long_name, route_type=route_type))

    def add_trip(self, route_id, trip_id, direction_id, stop_ids, offsets, departure):
        """offsets: seconds from the first stop, one per stop, non-decreasing."""
        self.trips.append(dict(route_id=route_id, service_id=SERVICE_ID, trip_id=trip_id, direction_id=direction_id))
        for seq, (stop_id, off) in enumerate(zip(stop_ids, offsets)):
            t = hms(departure + off)
            self.stop_times.append(dict(trip_id=trip_id, arrival_time=t, departure_time=t, stop_id=stop_id,
                                        stop_sequence=seq))

    def add_headway_line(self, route_id, stop_ids, offsets, headway_min, start, end, direction_id, prefix):
        """Trips every headway_min minutes departing the first stop from start to end (seconds)."""
        step = headway_min * 60.0
        k, dep = 0, float(start)
        while dep <= end:
            self.add_trip(route_id, '%s_%d_%05d' % (prefix, direction_id, k), direction_id, stop_ids, offsets, dep)
            k += 1
            dep = start + k * step

    def write(self, path):
        tables = dict(agency=self.agency, calendar=self.calendar, stops=list(self.stops.values()),
                      routes=self.routes, trips=self.trips, stop_times=self.stop_times)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, rows in tables.items():
                buf = io.StringIO()
                pd.DataFrame(rows).to_csv(buf, index=False)
                z.writestr(name + '.txt', buf.getvalue())
        return {k: len(v) for k, v in tables.items()}


def offsets_from_speed(positions_m, speed_kmh):
    """Seconds from the first station, running at a commercial speed that already includes dwell."""
    v = speed_kmh / 3.6
    return [(p - positions_m[0]) / v for p in positions_m]
