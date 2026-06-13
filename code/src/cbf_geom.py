import numpy as np

# Pendulum model and discrete CBF geometry, imported by cbf.py, the training
# script, and the visualization script.

# Physical model (Pendulum-v1 with modified actuation limits)
DT = 0.05
GRAV = 10.0
MASS = 2.0
LENGTH = 2.0

# Actuation limits
U_MAX = 15.0
SPEED_MAX = 60.0

# Safe set: |theta + LAM * theta_dot| <= F_LIM, encoded by four half-planes
LAM = 0.01
F_LIM = 1.0
H_ROWS = (
    np.array([1.0, LAM]),
    np.array([1.0, -LAM]),
    np.array([-1.0, LAM]),
    np.array([-1.0, -LAM]),
)

# Discrete CBF parameters
GAMMA = 0.5  # decrease rate gamma_b
KD = 1.5  # multiplier on the GP std margin


# Nominal one-step model x' = f(x) + g u
def nominal_dynamics(obs, u=0.0):
    # f is affine in u: passing u folds g*u into f. Pass u=0 for the CBF
    # bounds, where the control enters through g.
    obs = np.squeeze(obs)
    theta = np.arctan2(obs[1], obs[0])
    theta_dot = obs[2]
    f = np.array(
        [
            -3 * GRAV / (2 * LENGTH) * np.sin(theta + np.pi) * DT**2
            + theta_dot * DT
            + theta
            + 3 / (MASS * LENGTH**2) * u * DT**2,
            theta_dot
            - 3 * GRAV / (2 * LENGTH) * np.sin(theta + np.pi) * DT
            + 3 / (MASS * LENGTH**2) * u * DT,
        ]
    )
    g = np.array([3 / (MASS * LENGTH**2) * DT**2, 3 / (MASS * LENGTH**2) * DT])
    x = np.array([theta, theta_dot])
    return np.squeeze(f), np.squeeze(g), np.squeeze(x)


# Admissible interval [lo, hi] for the total control u
def u_bounds(f, g, x, std=None, kd=KD):
    # Intersects, over the four barrier rows, the constraints
    #   H_i (f + g u) >= (1 - GAMMA) H_i x - GAMMA F_LIM + kd |H_i std|
    # with the torque and next-step speed limits. std=None gives the nominal
    # bounds; the GP posterior std gives the robust bounds enforced in cbf.py.
    lo, hi = -U_MAX, U_MAX

    for h_vec in H_ROWS:
        margin = kd * abs(float(np.dot(h_vec, std))) if std is not None else 0.0
        rhs = (
            GAMMA * F_LIM
            + float(np.dot(h_vec, f))
            - (1 - GAMMA) * float(np.dot(h_vec, x))
            - margin
        )
        coef = -float(np.dot(h_vec, g))
        if abs(coef) < 1e-12:
            continue
        if coef > 0:
            hi = min(hi, rhs / coef)
        else:
            lo = max(lo, rhs / coef)

    f1, g1 = float(f[1]), float(g[1])
    if abs(g1) > 1e-12:
        b1 = (SPEED_MAX - f1) / g1
        b2 = (-SPEED_MAX - f1) / g1
        hi = min(hi, max(b1, b2))
        lo = max(lo, min(b1, b2))

    return lo, hi


# Nominal bounds straight from a state (theta, theta_dot)
def nominal_u_bounds(theta, theta_dot):
    obs = np.array([np.cos(theta), np.sin(theta), theta_dot])
    f, g, x = nominal_dynamics(obs, 0.0)
    return u_bounds(f, g, x, std=None)


# Check that the nominal bounds and the robust bounds at zero std are the same
# algebra. Run with: python code/src/cbf_geom.py
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        th = rng.uniform(-1.4, 1.4)
        td = rng.uniform(-30.0, 30.0)
        obs = np.array([np.cos(th), np.sin(th), td])
        f, g, x = nominal_dynamics(obs, 0.0)
        lo_n, hi_n = nominal_u_bounds(th, td)
        lo_r, hi_r = u_bounds(f, g, x, std=np.zeros(2))
        worst = max(worst, abs(lo_n - lo_r), abs(hi_n - hi_r))
        assert lo_n == lo_r and hi_n == hi_r, (th, td)
    print(f"consistency check passed, max bound mismatch {worst:.2e}")
