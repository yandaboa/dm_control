import numpy as np

def ff_pd_action_2d(
    *,
    obs_position,   # (2,)
    obs_velocity,   # (2,)
    pr, vr, ar,     # (2,) references at current timestep
    Kp=8.0,
    Kd=4.0,
    mass=0.3,
    gear=0.1,
    control_noise_std=0.0,
    rng=None,
):
    """
    Returns action u in [-10,10]^2 for the two motors (t1, t2).

    Model: pddot = (gear*u)/mass  (approx; matches point_mass well)
    """
    p = np.asarray(obs_position, dtype=float)
    v = np.asarray(obs_velocity, dtype=float)

    action_min = -10.0
    action_max = 10.0

    pr = np.asarray(pr, dtype=float)
    vr = np.asarray(vr, dtype=float)
    ar = np.asarray(ar, dtype=float)

    a_cmd = ar + Kp * (pr - p) + Kd * (vr - v)
    F_cmd = mass * a_cmd
    u = F_cmd / gear
    u = np.clip(u, action_min, action_max)

    if control_noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        u = np.clip(u + rng.normal(0.0, control_noise_std, size=(2,)), action_min, action_max)

    return u