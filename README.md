# Seyd Yar LCS — coast-aware rolling FTLE and interactive map

**Seyd Yar LCS** analyzes attracting FTLE and Lagrangian coherent structures in Copernicus global-ocean surface currents. Its derived accumulation-potential index is **not** a calibrated prediction of tuna catches.

This release adds rolling `-14` historical to up to `+10` forecast output dates (as data permits), per-target **7-day backward** integrations, shared Copernicus subsets, adaptive forecast fallback, conservative land/island crossing rejection, manual bounding box inputs in GitHub Actions, Leaflet map/interactive per-cell popups, and a daily LCS-map-change MP4.

Run via **GitHub Actions → Run Seyd Yar LCS → Run workflow**. Enter all four `lon_min, lon_max, lat_min, lat_max` fields for a custom rectangle, or leave all blank to use the saved `config/aoi/current.geojson`. Configure Copernicus username/password as repository secrets. The output report lists successful and skipped dates and reasons; GitHub Pages serves the map dashboard.

See [README_fa.md](README_fa.md) for full Persian operating instructions, validation caveats and architecture. See [CHANGELOG_vNext.md](CHANGELOG_vNext.md) for completed work and tests. Offline smoke tests: `python -m unittest discover -s tests -v`.

**Release caveat:** Remote Copernicus downloads, actual NumbaCS production execution and live Pages deployment could not be exercised in the isolated preparation environment; do a first GitHub Actions test run before using outputs operationally.
