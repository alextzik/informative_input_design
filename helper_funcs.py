"""
    Helper functions
"""

###############################
# Dependencies
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def plot_confidence_ellipse(mean, cov, ax=None, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if ax is None:
        ax = plt.gca()

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor="None", edgecolor='blue', **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], color='red', marker='x', label="Mean", zorder=5)

    return ax

# def plot_confidence_ellipse(mean, covariance, ax=None, confidence_level=0.95):
    """
    Plots a confidence ellipse (default: 95%) given a mean vector and covariance matrix.
    
    Parameters:
    - mean: The mean vector (2D array).
    - covariance: The covariance matrix (2D array).
    - ax: The axis to plot on (matplotlib Axes object, optional). If None, it will use the current axis.
    - confidence_level: The confidence level for the ellipse (default is 95%).
    
    Returns:
    - ax: The axis with the ellipse added.
    """
    # Default axis if none is provided
    if ax is None:
        ax = plt.gca()
    
    # Eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = eig(covariance)
    
    # Scaling factor based on confidence level (approx. 2.4477 for 95%)
    scaling_factor = np.sqrt(eigvals) * np.sqrt(2.0 * np.log(1.0 / (1.0 - confidence_level)))
    
    # Angle of rotation (orientation of the ellipse)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    
    # Generate the ellipse points (parametric form)
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_points = np.array([scaling_factor[0] * np.cos(theta), scaling_factor[1] * np.sin(theta)])
    
    # Rotate the ellipse based on the eigenvectors
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ellipse_points_rot = np.dot(rotation_matrix, ellipse_points)
    
    # Shift the ellipse to the mean
    ellipse_points_rot[0, :] += mean[0]
    ellipse_points_rot[1, :] += mean[1]
    
    # Plot the ellipse
    ax.plot(ellipse_points_rot[0, :], ellipse_points_rot[1, :], label=f'{int(confidence_level * 100)}% Confidence Ellipse', color='blue')
    ax.scatter(mean[0], mean[1], color='red', marker='x', label="Mean", zorder=5)
    
    ax.set_aspect('equal', adjustable='box')  # Keep aspect ratio equal
    ax.grid(True)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    
    return ax
