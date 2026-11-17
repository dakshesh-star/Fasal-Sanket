"""Small WGS84 GeoJSON helpers for field area and location metadata."""
import math

# WGS84 authalic sphere: areas computed on this sphere equal ellipsoidal surface
# areas after transforming geodetic latitude to authalic latitude.
WGS84_A = 6_378_137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
WGS84_E = math.sqrt(WGS84_E2)


def _authalic_latitude(latitude):
    phi = math.radians(latitude)
    sin_phi = math.sin(phi)
    q = (1 - WGS84_E2) * (sin_phi / (1 - WGS84_E2 * sin_phi * sin_phi)
        - math.log((1 - WGS84_E * sin_phi) / (1 + WGS84_E * sin_phi)) / (2 * WGS84_E))
    qp = 1 - (1 - WGS84_E2) * math.log((1 - WGS84_E) / (1 + WGS84_E)) / (2 * WGS84_E)
    return math.asin(max(-1.0, min(1.0, q / qp)))


def _authalic_radius():
    qp = 1 - (1 - WGS84_E2) * math.log((1 - WGS84_E) / (1 + WGS84_E)) / (2 * WGS84_E)
    return WGS84_A * math.sqrt(qp / 2)


AUTHALIC_RADIUS_M = _authalic_radius()


def geometry_from_geojson(boundary):
    if not boundary:
        return None
    kind = boundary.get("type")
    if kind == "Feature":
        return boundary.get("geometry")
    if kind == "FeatureCollection":
        features = boundary.get("features") or []
        return features[0].get("geometry") if len(features) == 1 and isinstance(features[0], dict) else None
    return boundary if kind in {"Polygon", "MultiPolygon"} else None


def polygon_rings(boundary):
    geometry = geometry_from_geojson(boundary)
    if not geometry:
        return []
    coords = geometry["coordinates"]
    return [coords] if geometry["type"] == "Polygon" else coords


def ring_area_m2(ring):
    total = 0.0
    for p1, p2 in zip(ring, ring[1:]):
        # GeoJSON positions are [longitude, latitude] in WGS84 degrees.
        lon1, lat1 = math.radians(p1[0]), _authalic_latitude(p1[1])
        lon2, lat2 = math.radians(p2[0]), _authalic_latitude(p2[1])
        dlon = lon2 - lon1
        if dlon > math.pi:
            dlon -= 2 * math.pi
        elif dlon < -math.pi:
            dlon += 2 * math.pi
        total += dlon * (2 + math.sin(lat1) + math.sin(lat2))
    return abs(total * AUTHALIC_RADIUS_M * AUTHALIC_RADIUS_M / 2.0)


def is_simple_ring(ring):
    """Reject self-intersecting rings using planar segment checks in local WGS84 degrees."""
    edges = list(zip(ring, ring[1:]))
    def orientation(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def intersects(a, b, c, d):
        o1, o2 = orientation(a, b, c), orientation(a, b, d)
        o3, o4 = orientation(c, d, a), orientation(c, d, b)
        return o1 * o2 < 0 and o3 * o4 < 0
    edge_count = len(edges)
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, edge_count):
            if j == i + 1 or (i == 0 and j == edge_count - 1):
                continue
            c, d = edges[j]
            if intersects(a, b, c, d):
                return False
    return True


def area_hectares(boundary):
    polygons = polygon_rings(boundary)
    if not polygons:
        return None
    total = 0.0
    for rings in polygons:
        total += max(0.0, ring_area_m2(rings[0]) - sum(ring_area_m2(r) for r in rings[1:]))
    return round(total / 10_000.0, 4)


def polygon_centroid(boundary):
    polygons = polygon_rings(boundary)
    coords = polygons[0][0][:-1] if polygons and polygons[0] and polygons[0][0] else []
    if not coords:
        return None
    return (sum(p[1] for p in coords) / len(coords), sum(p[0] for p in coords) / len(coords))
