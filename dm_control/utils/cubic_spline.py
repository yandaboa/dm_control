import numpy as np


def _hermite_segment_2d(p0, p1, m0, m1, s, T):
    """
    Cubic Hermite interpolation for one segment in 2D.
    p0,p1: (2,) endpoints
    m0,m1: (2,) endpoint velocities (units: pos/sec)
    s:     normalized time in [0,1]
    T:     segment duration (sec)
    Returns: p, v, a each (2,)
    """
    s2 = s * s
    s3 = s2 * s

    h00 = 2*s3 - 3*s2 + 1
    h10 = s3 - 2*s2 + s
    h01 = -2*s3 + 3*s2
    h11 = s3 - s2

    p = h00*p0 + h10*(T*m0) + h01*p1 + h11*(T*m1)

    dh00 = 6*s2 - 6*s
    dh10 = 3*s2 - 4*s + 1
    dh01 = -6*s2 + 6*s
    dh11 = 3*s2 - 2*s

    d2h00 = 12*s - 6
    d2h10 = 6*s - 4
    d2h01 = -12*s + 6
    d2h11 = 6*s - 2

    v = (dh00*p0 + dh10*(T*m0) + dh01*p1 + dh11*(T*m1)) / T
    a = (d2h00*p0 + d2h10*(T*m0) + d2h01*p1 + d2h11*(T*m1)) / (T*T)

    return p, v, a


def generate_spline_noise_traj_2d(
    *,
    dt=0.02,
    horizon_sec=10.0,
    bounds_low=(-0.28, -0.28),
    bounds_high=(0.28, 0.28),
    n_anchors=10,
    velocity_scale=0.30,   # ↑ = more aggressive / wiggly
    seed=0,
):
    """
    Returns:
      t:  (N,)
      pr: (N,2)
      vr: (N,2)
      ar: (N,2)
    """
    rng = np.random.default_rng(seed)
    low = np.array(bounds_low, dtype=float)
    high = np.array(bounds_high, dtype=float)
    box = high - low

    T_total = float(horizon_sec)
    N = int(np.round(T_total / dt)) + 1
    t = np.linspace(0.0, T_total, N)

    anchor_times = np.linspace(0.0, T_total, n_anchors)
    P = rng.uniform(low, high, size=(n_anchors, 2))

    # Velocities scaled relative to box size and segment duration.
    Tseg = T_total / (n_anchors - 1)
    M = rng.normal(size=(n_anchors, 2)) * (velocity_scale * box / max(Tseg, 1e-6))
    M[0] *= 0.2
    M[-1] *= 0.2

    pr = np.zeros((N, 2), dtype=float)
    vr = np.zeros((N, 2), dtype=float)
    ar = np.zeros((N, 2), dtype=float)

    seg_idx = np.clip(np.searchsorted(anchor_times, t, side="right") - 1, 0, n_anchors - 2)

    for k in range(N):
        i = int(seg_idx[k])
        t0, t1 = anchor_times[i], anchor_times[i + 1]
        T = t1 - t0
        s = 0.0 if T <= 0 else (t[k] - t0) / T
        p, v, a = _hermite_segment_2d(P[i], P[i + 1], M[i], M[i + 1], s, T)
        pr[k], vr[k], ar[k] = p, v, a

    # keep reference inside the walls
    pr = np.clip(pr, low, high)
    return t, pr, vr, ar


def scale_to_accel_limit_2d(vr, ar, amax=0.25):
    """
    Global scaling so ||a|| max <= amax (good to avoid constant saturation).
    Returns scale factor applied.
    """
    max_norm = np.max(np.linalg.norm(ar, axis=1))
    if max_norm <= amax:
        return 1.0
    s = amax / max_norm
    vr *= s
    ar *= s
    return s