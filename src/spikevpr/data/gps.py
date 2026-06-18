"""
GPS helpers shared by the dataset loaders and the recall metric.

Two distance conventions appear across the three datasets:

  * Brisbane uses geographic (lat, lon) coordinates. Distances are geodesic
    metres (``calculate_gps_distance``); a vectorised haversine is provided for
    the O(N^2) positive-mining step.
  * NSAVP (EDEF) and NYC (UTM) use planar metre coordinates; distances are plain
    Euclidean (``euclidean_distance``).

``coordinate_distance`` dispatches on magnitude (|first component| <= 90 => geo),
which lets the same recall code serve all three datasets.
"""
import numpy as np
import pynmea2
from geopy.distance import geodesic

_EARTH_RADIUS_M = 6_371_000.0


# ── geographic distance ─────────────────────────────────────────────────────────

def calculate_gps_distance(coord1, coord2):
    """Geodesic distance in metres between two (lat, lon) points."""
    return geodesic(coord1, coord2).meters


def euclidean_distance(p1, p2):
    """Planar Euclidean distance in the XY plane (EDEF / UTM coordinates)."""
    p1, p2 = np.asarray(p1), np.asarray(p2)
    return float(np.linalg.norm(p2[:2] - p1[:2]))


def coordinate_distance(c1, c2):
    """Distance in metres, auto-selecting geodesic vs Euclidean by coordinate type."""
    if abs(c1[0]) <= 90:
        return calculate_gps_distance(c1, c2)
    return euclidean_distance(c1, c2)


def haversine_matrix(lats, lons):
    """
    Vectorised pairwise haversine distance matrix (metres) for arrays of
    latitudes/longitudes. Used to build positive indices without an O(N^2) loop
    of geopy calls.
    """
    lat = np.radians(np.asarray(lats, dtype=np.float64))
    lon = np.radians(np.asarray(lons, dtype=np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── NMEA parsing (Brisbane) ──────────────────────────────────────────────────────

def get_gps(nmea_file_path):
    """
    Parse an NMEA file into an (N, 3) array of [lat, lon, seconds-from-start].

    Adapted from the ensemble-event-vpr reference implementation
    (https://github.com/Tobias-Fischer/ensemble-event-vpr).
    """
    latitudes, longitudes, timestamps = [], [], []
    first_timestamp = None
    previous_lat, previous_lon = 0, 0

    with open(nmea_file_path, encoding="utf-8") as nmea_file:
        for line in nmea_file.readlines():
            try:
                msg = pynmea2.parse(line)
                if first_timestamp is None:
                    first_timestamp = msg.timestamp
                if msg.sentence_type in ("GSV", "VTG", "GSA"):
                    continue
                dist = np.linalg.norm(np.array([msg.latitude, msg.longitude])
                                      - np.array([previous_lat, previous_lon]))
                if (msg.latitude != 0 and msg.longitude != 0
                        and msg.latitude != previous_lat and msg.longitude != previous_lon
                        and dist > 0.0001):
                    diff = ((msg.timestamp.hour - first_timestamp.hour) * 3600
                            + (msg.timestamp.minute - first_timestamp.minute) * 60
                            + (msg.timestamp.second - first_timestamp.second))
                    latitudes.append(msg.latitude)
                    longitudes.append(msg.longitude)
                    timestamps.append(diff)
                    previous_lat, previous_lon = msg.latitude, msg.longitude
            except pynmea2.ParseError:
                continue

    return np.vstack((latitudes, longitudes, timestamps)).T


def match_x1_to_x2(x1, x2):
    """
    Align traverse x1 to the nearest points of traverse x2 by lat/lon.

    Adapted from the Brisbane-Event-VPR reference notebook
    (https://github.com/Tobias-Fischer/ensemble-event-vpr).
    """
    matched = []
    for idx1, (latlon, _t) in enumerate(zip(x1[:, 0:2], x1[:, 2])):
        if len(matched) < 6:
            lo, hi = 0, int(0.25 * len(x2))
        elif idx1 > 0.5 * len(x1):
            lo, hi = matched[-5], len(x2)
        else:
            lo, hi = matched[-5], int(0.75 * len(x2))
        best = np.linalg.norm(x2[lo:hi, 0:2] - latlon, axis=1).argmin() + lo
        matched.append(best)
    return np.array(matched)


def get_timestamp_matches(timestamps, timestamps_to_match):
    return np.array([np.abs(timestamps - ts).argmin() for ts in timestamps_to_match])


def get_raw_gps_timestamps(x, mask=None):
    return x[:, 2] if mask is None else x[mask, 2]


def get_positions(x, mask=None):
    """Return lat/lon pairs as an array of (lat, lon) tuples."""
    coords = x[:, 0:2] if mask is None else x[mask, 0:2]
    return np.array([tuple(c) for c in coords])
