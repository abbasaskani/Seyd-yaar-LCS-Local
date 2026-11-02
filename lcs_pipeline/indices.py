from __future__ import annotations

from datetime import datetime
import numpy as np

from .ftle import FTLEOutputs, robust_normalize


def centered_persistence(normalized_fields: list[np.ndarray], index: int, half_window: int,
                         calendar_days: list | None = None) -> np.ndarray:
    """Calendar-day-aware 3/5-day mean; never borrows a distant day across gaps."""
    if calendar_days is None:
        nearby = range(max(0,index-half_window),min(len(normalized_fields),index+half_window+1))
    else:
        nearby = [k for k,d in enumerate(calendar_days)
                  if abs((d-calendar_days[index]).days) <= half_window]
    if not nearby:
        return np.full_like(normalized_fields[index], np.nan)
    stack=np.stack([normalized_fields[k] for k in nearby],axis=0)
    count=np.isfinite(stack).sum(axis=0)
    return np.divide(np.nansum(stack,axis=0), count,
                     out=np.full_like(stack[0], np.nan), where=count>0)


def attach_persistence_and_scores(outputs: list[FTLEOutputs], config_raw: dict) -> list[FTLEOutputs]:
    if not outputs:
        return []
    q_low,q_high=config_raw['persistence'].get('robust_quantiles',[5,95])
    ftle_norms=[robust_normalize(out.ftle_smooth,q_low=q_low,q_high=q_high) for out in outputs]
    dates=[]
    for out in outputs:
        meta=out.metadata or {}
        raw=meta.get('requested_target_date_local') or meta.get('actual_selected_local')
        dates.append(datetime.fromisoformat(raw[:10]).date() if raw else None)
    if any(d is None for d in dates):
        dates=None
    balanced_w=config_raw['composites']['balanced']
    physics_w=config_raw['composites']['physics_first']
    for i,out in enumerate(outputs):
        p3=centered_persistence(ftle_norms,i,half_window=1,calendar_days=dates)
        p5=centered_persistence(ftle_norms,i,half_window=2,calendar_days=dates)
        out.persistence_3d=p3;out.persistence_5d=p5
        ftle_norm=ftle_norms[i]
        ridge=np.clip(out.ridge_support,0,1)
        out.balanced_composite=np.clip(
            balanced_w['ftle']*ftle_norm+balanced_w['ridge_support']*ridge+balanced_w['persistence']*p5,0,1)
        out.physics_first_composite=np.clip(
            physics_w['ftle']*ftle_norm+physics_w['ridge_support']*ridge+physics_w['persistence']*p5,0,1)
        out.accumulation_potential_balanced=out.balanced_composite*100
        out.accumulation_potential_physics_first=out.physics_first_composite*100
    return outputs
