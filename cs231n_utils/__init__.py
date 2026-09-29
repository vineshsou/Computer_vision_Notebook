"""Shared helpers for the CS231n notebook series."""
from .data import get_cifar10, load_cifar10_raw, CLASSES
from .gradcheck import (rel_error, eval_numerical_gradient,
                        eval_numerical_gradient_array, grad_check_sparse)
from .checks import grade, show_solution, list_solutions, use
from .viz import (show_images, plot_loss_curves, plot_weight_templates,
                  plot_series, setup_plots, SERIES,
                  plot_norm_balls, plot_distance_concentration, plot_knn_regions,
                  plot_score_geometry, plot_hyperplane_regions,
                  SURFACE, INK, INK_2, INK_MUTED, GRID)

__all__ = [
    "get_cifar10", "load_cifar10_raw", "CLASSES",
    "rel_error", "eval_numerical_gradient", "eval_numerical_gradient_array",
    "grad_check_sparse", "grade", "show_solution", "list_solutions", "use",
    "show_images", "plot_loss_curves", "plot_weight_templates", "plot_series",
    "setup_plots", "SERIES",
    "plot_norm_balls", "plot_distance_concentration", "plot_knn_regions",
    "plot_score_geometry", "plot_hyperplane_regions",
    "SURFACE", "INK", "INK_2", "INK_MUTED", "GRID",
]
