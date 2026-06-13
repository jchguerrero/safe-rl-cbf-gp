import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

from src.cbf_geom import nominal_dynamics


# Nominal dynamics (inaccurate model)
def get_nominal_dynamics(obs, u_rl):
    f, g, x = nominal_dynamics(obs, u_rl)
    return [f, g, x]


# GP Model
def build_GP_model(obs_size):
    GP_list = []
    noise = 0.01
    for i in range(obs_size - 1):
        kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kern, alpha=noise, n_restarts_optimizer=10)
        GP_list.append(gp)
    return GP_list


# Update GP dynamics
def update_GP_dynamics(GP_model, X_obs, U_control, obs_size):
    L = X_obs.shape[0]
    err = np.zeros((L - 1, obs_size - 1))
    S = np.zeros((L - 1, 2))
    for i in range(L - 1):
        f, _, _ = get_nominal_dynamics(X_obs[i, :], U_control[i])
        theta_p = np.arctan2(X_obs[i, 1], X_obs[i, 0])
        theta_dot_p = X_obs[i, 2]
        theta = np.arctan2(X_obs[i + 1, 1], X_obs[i + 1, 0])
        theta_dot = X_obs[i + 1, 2]
        S[i, :] = np.array([theta_p, theta_dot_p])
        err[i, :] = np.array([theta, theta_dot]) - f
    GP_model[0].fit(S, err[:, 0])
    GP_model[1].fit(S, err[:, 1])


# One-step RMSE: all states and large-angle states
def gp_prediction_error(GP_model, X_obs, U_control, obs_size, large_angle=30.0):
    L = X_obs.shape[0]
    if L < 2:
        return None
    nom = np.zeros((L - 1, obs_size - 1))
    gp = np.zeros((L - 1, obs_size - 1))
    theta_prev = np.zeros(L - 1)
    for i in range(L - 1):
        f, _, _ = get_nominal_dynamics(X_obs[i, :], U_control[i])
        theta_p = np.arctan2(X_obs[i, 1], X_obs[i, 0])
        theta_dot_p = X_obs[i, 2]
        theta = np.arctan2(X_obs[i + 1, 1], X_obs[i + 1, 0])
        theta_dot = X_obs[i + 1, 2]
        true_next = np.array([theta, theta_dot])
        s = np.array([theta_p, theta_dot_p]).reshape(1, -1)
        gp_corr = np.array(
            [GP_model[0].predict(s).flat[0], GP_model[1].predict(s).flat[0]]
        )
        nom[i, :] = true_next - f
        gp[i, :] = true_next - (f + gp_corr)
        theta_prev[i] = theta_p

    large = np.abs(theta_prev) > np.radians(large_angle)
    return {
        "nom_rmse": float(np.sqrt((nom**2).mean())),
        "gp_rmse": float(np.sqrt((gp**2).mean())),
        "nom_rmse_large": (
            float(np.sqrt((nom[large] ** 2).mean())) if large.any() else np.nan
        ),
        "gp_rmse_large": (
            float(np.sqrt((gp[large] ** 2).mean())) if large.any() else np.nan
        ),
    }


# GP dynamics (inference)
def get_GP_dynamics(GP_model, obs):
    # drift only (u = 0); control enters through g, not f
    f_nom, g, x = nominal_dynamics(obs, 0.0)
    f = np.zeros(2)
    [m1, std1] = GP_model[0].predict(x.reshape(1, -1), return_std=True)
    [m2, std2] = GP_model[1].predict(x.reshape(1, -1), return_std=True)
    f[0] = f_nom[0] + m1.flat[0]
    f[1] = f_nom[1] + m2.flat[0]
    return [
        np.squeeze(f),
        np.squeeze(g),
        np.squeeze(x),
        np.array([np.squeeze(std1), np.squeeze(std2)]),
    ]
