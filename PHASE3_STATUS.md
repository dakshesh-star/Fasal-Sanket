# Fasal Sanket Phase 3 status

Implementation status, 2026-11-16. “Tested” means local software checks, not scientific validation.

## Complete and tested locally

- Authentication: registration, login, logout, `/auth/me`, JWT bearer access, rotating HttpOnly refresh cookies, scrypt password hashes, and owner-scoped field routes.
- SQLite persistence for accounts, sessions, fields, observations, and saved crop-identification reports.
- Polygon-first field workflow: fields default to **Unknown / Not identified**; users may provide GeoJSON geometry without point coordinates. The API derives centroid and area, validates WGS84 ranges, ring closure, nonzero area, and simple-ring intersections. Point-only fields remain supported.
- Polygon-based Sentinel-2 L2A STAC scene discovery. A live catalog query against a test polygon returned 14 matching scene records. The endpoint marks these as metadata discovery; it does not claim downloaded pixels.
- Experimental crop candidate route: requires a polygon and at least one usable stored observation because the current model accepts a 20-bin temporal tensor. Missing seasonal bins are imputed and disclosed. Any output is experimental, empirical reliability is unavailable, and taxonomy stays at level 0.
- Synthetic demonstration workflow remains separate and labeled as such.
- Dashboard and field history are wired to persisted data, with empty states when there are no observations.
- `pytest backend/tests -q`: **28 passed**, with one Starlette deprecation warning.

## Implemented, with external data access still blocked

- Direct Copernicus Data Space STAC/S3 L2A asset acquisition, BOA reflectance scaling from product metadata, SCL cloud/shadow masking, polygon extraction, and NDVI/NDWI/EVI processing are implemented. Live STAC discovery confirms scenes/assets exist for the saved polygon. Actual pixels were not acquired: no CDSE S3 credentials are configured, and local rasterio/boto3 dependencies are absent. No imagery-derived observations or crop candidates were generated in this session.
- Browser UI includes polygon drawing/editing, GeoJSON upload, spatial scene search, processing controls, and evidence-gated identification. Fixed two JavaScript syntax errors that stopped the frontend bundle from executing; changed Leaflet loading so an unavailable CDN no longer blocks the rest of the app. Live browser smoke check confirmed the dashboard and Field Management, Satellite Explorer, Live Crop Intelligence, Resolution Calendar, Crop Analytics, Model Performance, and Settings all render. Polygon map interactions still depend on Leaflet and network access to its CDN; GeoJSON and coordinate entry remain available if that library cannot load.

## Not available in this checkout

- Real labeled crop observations, geographic/temporal holdouts, model calibration, independent validation, or operational performance claims.
- Empirical reliability and out-of-distribution assessment for the supplied synthetic-trained models.
- Password reset delivery (no mail provider configured), NDVI raster rendering/download, and downloadable source imagery assets.

## Setup for real imagery processing

Generate CDSE S3 credentials from the [S3 credentials panel](https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials), add `CDSE_S3_ACCESS_KEY_ID` and `CDSE_S3_SECRET_ACCESS_KEY` to the private `.env`, install `backend/requirements.txt`, and restart the backend. Check saved polygon boundaries and areas before imagery retrieval, especially unusually large polygons. Real labeled ground truth is required before claims of crop-identification accuracy or taxonomic unlocks can be made.
