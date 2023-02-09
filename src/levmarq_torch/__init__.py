import functorch
import torch
from torch import Tensor


def minimize_levmarq(
    x,
    y,
    get_y_hat,
    eps_metric=0.1,
    lam=1e-2,
    eps_grad=None,
    eps_x=None,
    eps_reduced_chi2=None,
    lam_decrease_factor=9.0,
    lam_increase_factor=11.0,
    max_iters=10,
    small_number=0.0,
    lam_min=1e-7,
    lam_max=1e7,
) -> Tensor:
    """
    Levenberg-Marquardt minimizer, based on implementation 1 from `Gavin 2022 <https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf>`_.

    This minimizes :math:`|\\mathbf{y} - \\mathbf{\\hat{y}}(\\mathbf{x})|^2`, and is
    batched over parameters `x` and datapoints `y`.

    Args:
        x: batch of initial parameters.
        y: corresponding batch of target data.
        get_y_hat: function taking components of `x` as arguments and returning a
            prediction of `y`. Must be `vmap`able.
        eps_metric: tolerance for deciding when to decrease :math:`\\lambda`.
        lam: damping parameter :math:`\\lambda`.
        eps_grad: gradient convergence threshold. If `None`, doesn't check this
            convergence metric.
        eps_x: parameter convergence threshold. If `None`, doesn't check this
            convergence metric.
        eps_reduced_chi2: reduced :math:`\\chi^2` convergence threshold. If `None`,
            doesn't check this convergence metric.
        lam_decrease_factor: factor by which to decrease :math:`\\lambda` after accepting
            an update.
        lam_increase_factor: factor by which to increase :math:`\\lambda` after rejecting
            an update.
        max_iters: maximum number of iterations.
        small_number: this is added to the diagonal of the approximate Hessian and the
            parameter values to avoid inverting a singular matrix or dividing by zero.
        lam_min: minimum value permitted for :math:`\\lambda`.
        lam_max: maximum value permitted for :math:`\\lambda`.

    Notes:
        - Switch to jacfwd
        - Should be able to get value and Jacobian simultaneously
    """
    if len(x) != len(y):
        raise ValueError("x and y must having matching batch dimension")

    b, n = x.shape
    m = y.shape[1]
    if m - n + 1 <= 0:
        raise ValueError(
            "number of data points per batch must be at least the number of parameters "
            "minus 1"
        )

    # Make vmapd functions
    argnums = tuple(range(n))
    get_y_hat_v = functorch.vmap(lambda x: get_y_hat(*x))
    get_jac = functorch.vmap(lambda x: functorch.jacrev(get_y_hat, argnums)(*x))

    hess_perturbation = small_number * torch.eye(n, device=x.device).repeat(b, 1, 1)
    chi2_prev = ((y - get_y_hat_v(x)) ** 2).sum(-1)

    for _ in range(max_iters):
        # Reduced chi^2 convergence check
        if (
            eps_reduced_chi2 is not None
            and chi2_prev.max() / (m - n + 1) < eps_reduced_chi2
        ):
            break

        J = torch.stack(get_jac(x), -1)
        y_hat = get_y_hat_v(x)
        dy = (y - y_hat)[:, :, None]
        JT_dy = J.transpose(-2, -1) @ dy  # [:, :, 0]

        # Gradient convergence check
        if eps_grad is not None and JT_dy.max() < eps_grad:
            break

        # Propose an update
        JTJ = J.transpose(-2, -1) @ J
        hess_approx = (
            lam * torch.diag_embed(torch.diagonal(JTJ, dim1=-2, dim2=-1))
            + hess_perturbation
        )
        dx = torch.linalg.inv(JTJ + hess_approx) @ JT_dy

        # Parameter convergence check
        if eps_x is not None and (dx / (x + small_number)).abs().max() < eps_x:
            break

        x_new = x + dx[:, :, 0]
        chi2 = ((y - get_y_hat_v(x_new)) ** 2).sum()
        # Decide whether to accept update
        metric = (chi2_prev - chi2) / (
            (dx * JT_dy).sum(-2)[:, 0] + (dx * (hess_approx @ dx)).sum(-2)[:, 0]
        ).abs()
        if metric.max() > eps_metric:
            x = x_new
            chi2_prev = chi2
            lam = max(lam / lam_decrease_factor, lam_min)
        else:
            lam = min(lam * lam_increase_factor, lam_max)

    return x
