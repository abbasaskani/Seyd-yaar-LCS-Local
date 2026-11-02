"""Coast-aware advection: reject trajectories crossing the model's land mask.

This is a conservative *model-grid* boundary, not high-resolution coastline
physics. Nearshore cells are excluded rather than presenting false FTLE ridges.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates


def native_water_mask(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(time, lon, lat) -> stable ocean cells. Intermittent NaNs are invalid."""
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError('Expected matching (time, longitude, latitude) velocity arrays')
    return np.isfinite(u).all(axis=0) & np.isfinite(v).all(axis=0)


def fill_ocean_velocity(data: np.ndarray, water: np.ndarray) -> np.ndarray:
    """Nearest-ocean numerical fill; filled land values are NEVER valid output."""
    if not np.any(water):
        raise ValueError('No usable ocean cells in selected area')
    idx = distance_transform_edt(~water, return_distances=False, return_indices=True)
    return np.asarray(data[:, idx[0], idx[1]], dtype=np.float64)


def _grid_coords(xp, yp, x, y):
    return ((xp - x[0]) / (x[1] - x[0]), (yp - y[0]) / (y[1] - y[0]))


def coast_safe_flowmap(t_hours, x_native, y_native, u_kmh, v_kmh,
                        x_out, y_out, backward_hours=168., max_step_hours=1.,
                        coast_guard_native_cells=1):
    """Advect back from t=0, validating ocean at every RK4 evaluation.

    Returns (flowmap[nx,ny,2], valid[nx,ny], native_water[nx,ny]).
    Velocity is in km/h and native axes in km. No unphysical land crossing.
    """
    t = np.asarray(t_hours, dtype=float)
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError('Time samples must be strictly increasing')
    if t[0] > -backward_hours + 1.01 or t[-1] < -0.01:
        raise ValueError('Insufficient temporal coverage for complete backward FTLE')
    if np.nanmax(np.diff(t)) > 3.01:
        raise ValueError('A gap in ocean currents is too large for continuous particle tracking')
    water = native_water_mask(u_kmh, v_kmh)
    if min(water.shape) < 3 or not water.any():
        raise ValueError('Area too small or entirely land')
    safe_water = binary_erosion(water, iterations=int(coast_guard_native_cells), border_value=0) if coast_guard_native_cells else water
    if not safe_water.any():
        raise ValueError('No safe offshore grid cells after coast margin')
    uf = fill_ocean_velocity(u_kmh, water)
    vf = fill_ocean_velocity(v_kmh, water)
    X, Y = np.meshgrid(x_out, y_out, indexing='ij')
    xp, yp = X.ravel().copy(), Y.ravel().copy()
    initial = np.column_stack((xp.copy(), yp.copy()))
    valid = np.ones(xp.size, dtype=bool)
    step_count = int(np.ceil(float(backward_hours) / max(float(max_step_hours), 0.1)))
    h = -float(backward_hours) / step_count

    def sample(xp_, yp_, time_):
        ix, iy = _grid_coords(xp_, yp_, x_native, y_native)
        inside = ((ix >= 0) & (ix <= len(x_native)-1) & (iy >= 0) & (iy <= len(y_native)-1))
        cells = np.vstack((ix, iy))
        wet = map_coordinates(safe_water.astype(np.float32), cells, order=0, mode='constant', cval=0.0) >= 0.5
        ti = np.interp(time_, t, np.arange(len(t), dtype=float))
        coords = np.vstack((np.full(xp_.shape, ti), ix, iy))
        uu = map_coordinates(uf, coords, order=1, mode='nearest')
        vv = map_coordinates(vf, coords, order=1, mode='nearest')
        return uu, vv, inside & wet & np.isfinite(uu) & np.isfinite(vv)

    for step in range(step_count):
        now = step * h
        u1, v1, ok1 = sample(xp, yp, now)
        u2, v2, ok2 = sample(xp + h*u1/2, yp + h*v1/2, now+h/2)
        u3, v3, ok3 = sample(xp + h*u2/2, yp + h*v2/2, now+h/2)
        u4, v4, ok4 = sample(xp + h*u3, yp + h*v3, now+h)
        nx = xp + (h / 6) * (u1 + 2*u2 + 2*u3 + u4)
        ny = yp + (h / 6) * (v1 + 2*v2 + 2*v3 + v4)
        _, _, ok_end = sample(nx, ny, now+h)
        valid &= ok1 & ok2 & ok3 & ok4 & ok_end
        xp, yp = nx, ny
        if not valid.any():
            raise ValueError('All particle trajectories intersect land/edge; shrink the region or coast guard')
    flowmap = np.stack((xp, yp), axis=-1).reshape(len(x_out), len(y_out), 2)
    return flowmap, valid.reshape(len(x_out),len(y_out)), water
