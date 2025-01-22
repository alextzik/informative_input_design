"""
    The main loop
"""

###############################
# Dependencies
import numpy as np
import cvxpy as cp
import random 

import matplotlib.pyplot as plt

from dynamics import true_dynamics, a, b, model
from method import compute_map_estimate, compute_next_input, fit_model
from helper_funcs import plot_confidence_ellipse

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

###############################
# Parameters
theta_prior = np.array([7., 4.])
Sigma_prior = 0.1*np.eye(2)

xs = [np.array([1., 1.])]
ys = [true_dynamics(x) for x in xs]

Sigma_obs = 10*np.eye(2)

num_timesteps = 20
NUMBER_OF_SINS = 30

###############################
# Methods
def run_proposed_method(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    
    theta_est = theta_prior.copy()
    dists = [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]
    Sigmas_obs = [Sigma_obs for _x in range(len(xs))]

    ax = plot_confidence_ellipse(theta_est, Sigma_prior)

    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],#Sigmas_obs,
                                            )

        delta_ys = []
        for _ in range(len(ys)):
            delta_ys += [(model(xs[_], theta_est) - ys[_]).reshape(-1,1)]
        Sigmas_obs = [0.8*1/len(delta_ys) * np.sum([_y @ _y.T for _y in delta_ys], axis=0) + 0.2*Sigma_obs 
                        for _ in range(len(xs)+1)]

        x_next, Sigma_post = compute_next_input(theta_est=theta_est,
                                    Sigma_prior=Sigma_prior,
                                    xs=xs,
                                    Sigmas_obs=Sigmas_obs)
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        ax = plot_confidence_ellipse(theta_est, Sigma_post, ax)

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]
    
    ax.set_aspect('equal', adjustable='box')  # Keep aspect ratio equal
    ax.grid(True)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    plt.show()

    return dists


def run_proposed_method_no_Sigma_update(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    
    theta_est = theta_prior.copy()
    dists = [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],
                                            )
        
        x_next, Sigma_post = compute_next_input(theta_est=theta_est,
                                    Sigma_prior=Sigma_prior,
                                    xs=xs,
                                    Sigmas_obs=[Sigma_obs for _x in range(len(xs)+1)])
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    return dists


def run_random_selection(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    dists = []
    theta_est = theta_prior.copy()
    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],
                                            )
        
        x_next = 1*np.random.normal(size=xs[0].shape)
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    return dists

def run_prbs(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    dists = []
    theta_est = theta_prior.copy()
    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],
                                            )
        
        x_next = np.array([random.choice([0, 1]) for _ in range(len(xs[0]))]).reshape(xs[0].shape)
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    return dists

def run_multisine(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    dists = []
    theta_est = theta_prior.copy()

    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],
                                            )
        
        x_next = np.array([
            np.sum([np.random.uniform(-1,1)*np.sin(2*np.pi*(2*j+0)*(timestep+1)/num_timesteps + np.random.uniform(0, 2*np.pi)) for j in range(NUMBER_OF_SINS)]),
            np.sum([np.random.uniform(-1,1)*np.sin(2*np.pi*(2*j+1)*(timestep+1)/num_timesteps + np.random.uniform(0, 2*np.pi)) for j in range(NUMBER_OF_SINS)])
        ]).reshape(xs[0].shape)
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    return dists

###############################
# Main loop
dists_proposed = run_proposed_method(theta_prior=theta_prior,
                                        Sigma_prior=Sigma_prior,
                                        xs=xs,
                                        ys=ys,
                                        Sigma_obs=Sigma_obs)

dists_proposed_no_Sigma_update = run_proposed_method_no_Sigma_update(theta_prior=theta_prior,
                                        Sigma_prior=Sigma_prior,
                                        xs=xs,
                                        ys=ys,
                                        Sigma_obs=Sigma_obs)
dists_random = run_random_selection(theta_prior=theta_prior,
                                        Sigma_prior=Sigma_prior,
                                        xs=xs,
                                        ys=ys,
                                        Sigma_obs=Sigma_obs)
dists_prbs = run_prbs(theta_prior=theta_prior,
                                        Sigma_prior=Sigma_prior,
                                        xs=xs,
                                        ys=ys,
                                        Sigma_obs=Sigma_obs)
dists_multisine = run_multisine(theta_prior=theta_prior,
                                        Sigma_prior=Sigma_prior,
                                        xs=xs,
                                        ys=ys,
                                        Sigma_obs=Sigma_obs)
plt.plot(range(len(dists_proposed)), dists_proposed, label="proposed")
plt.plot(range(len(dists_proposed_no_Sigma_update)), dists_proposed_no_Sigma_update, label="proposed (no finetuning)")
plt.plot(range(len(dists_random)), dists_random, label="random")
plt.plot(range(len(dists_prbs)), dists_prbs, label="PRBS")
plt.plot(range(len(dists_multisine)), dists_multisine, label="multlisine")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel(r'$||\hat{\theta} - \theta_\mathrm{true}||_\infty$')
plt.show()