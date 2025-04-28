from typing import Any

import tensorflow as tf
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.services.sde import simulate_heston_jump_diffusion

router = APIRouter(prefix="/futures_simulate", tags=["futures_simulate"])


@router.post("", response_class=JSONResponse)
def simulate_sde_endpoint(
        initial_price: float = Query(...),
        init_vol: float = Query(...),
        mu: float = Query(...),
        kappa: float = Query(...),
        theta: float = Query(...),
        xi: float = Query(...),
        rho: float = Query(...),
        jump_intensity: float = Query(...),
        jump_loc: float = Query(...),
        jump_scale: float = Query(...),
        dt: float = Query(3600.0),
        steps: int = Query(60),
        paths: int = Query(100)
) -> Any:
    traj = simulate_heston_jump_diffusion(
        S0=tf.constant(initial_price),
        v0=tf.constant(init_vol),
        mu=tf.constant(mu), kappa=tf.constant(kappa),
        theta=tf.constant(theta), xi=tf.constant(xi),
        rho=tf.constant(rho),
        jump_intensity=tf.constant(jump_intensity),
        jump_loc=tf.constant(jump_loc),
        jump_scale=tf.constant(jump_scale),
        dt=tf.constant(dt, tf.float32),
        steps=steps, paths=paths
    ).numpy().tolist()
    return {"simulation": traj}
