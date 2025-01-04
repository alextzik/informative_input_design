"""
    Helper functions
"""

###############################
# Dependencies
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

def plot_confidence_ellipse(mean, covariance, ax=None, confidence_level=0.95):
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
    
    return ax
