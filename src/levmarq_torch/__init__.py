import functorch
import torch
from torch import Tensor


def minimize_levmarq(
    xs,
    ys,
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
    if len(xs) != len(ys):
        raise ValueError("x and y must having matching batch dimension")

    b, n = xs.shape
    m = ys.shape[1]
    if m - n + 1 <= 0:
        raise ValueError(
            "number of data points per batch must be at least the number of parameters "
            "minus 1"
        )

    # Make vmapd functions
    argnums = tuple(range(n))
    get_y_hat_v = functorch.vmap(lambda x: get_y_hat(*x))
    get_jac = functorch.vmap(lambda x: functorch.jacrev(get_y_hat, argnums)(*x))

    H_perturbation = small_number * torch.eye(n, device=xs.device).repeat(b, 1, 1)
    chi2_prevs = ((ys - get_y_hat_v(xs)) ** 2).sum(-1)

    for _ in range(max_iters):
        # Reduced chi^2 convergence check
        if (
            eps_reduced_chi2 is not None
            and chi2_prevs.max() / (m - n + 1) < eps_reduced_chi2
        ):
            break

        Js = torch.stack(get_jac(xs), -1)
        y_hats = get_y_hat_v(xs)
        dys = (ys - y_hats)[:, :, None]
        JT_dys = Js.transpose(-2, -1) @ dys  # [:, :, 0]

        # Gradient convergence check
        if eps_grad is not None and JT_dys.max() < eps_grad:
            break

        # Propose an update
        JTJs = Js.transpose(-2, -1) @ Js
        H_approxs = (
            lam * torch.diag_embed(torch.diagonal(JTJs, dim1=-2, dim2=-1))
            + H_perturbation
        )
        # print(H_approxs)
        dxs = torch.linalg.inv(JTJs + H_approxs) @ JT_dys

        # Parameter convergence check
        if eps_x is not None and (dxs / (xs + small_number)).abs().max() < eps_x:
            break

        x_news = xs + dxs[:, :, 0]
        chi2s = ((ys - get_y_hat_v(x_news)) ** 2).sum()
        # Decide whether to accept update
        metrics = (chi2_prevs - chi2s) / (
            (dxs * JT_dys).sum(-2)[:, 0] + (dxs * (H_approxs @ dxs)).sum(-2)[:, 0]
        ).abs()
        if metrics.max() > eps_metric:
            xs = x_news
            chi2_prevs = chi2s
            lam = max(lam / lam_decrease_factor, lam_min)
        else:
            lam = min(lam * lam_increase_factor, lam_max)

    return xs


# def minimize_levmarq_new(
#     xs,
#     ys,
#     get_y_hat,
#     eps_metric=0.1,
#     lam=1e-2,
#     eps_grad=None,
#     eps_x=None,
#     eps_reduced_chi2=None,
#     lam_decrease_factor=9.0,
#     lam_increase_factor=11.0,
#     max_iters=10,
#     small_number=0.0,
#     lam_min=1e-7,
#     lam_max=1e7,
# ) -> Tensor:
#     """
#     Levenberg-Marquardt minimizer, based on implementation 1 from `Gavin 2022 <https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf>`_.
#     This minimizes :math:`|\\mathbf{y} - \\mathbf{\\hat{y}}(\\mathbf{x})|^2`.
# 
#     The implementation is batched over parameters `x` and datapoints `y`. It separately
#     varies the parameter :math:`\\lambda` for each element in the batch, but only
#     declares convergence when all the minimization problems have converged. If the
#     problems in the batch take wildly different amounts of time to converge, it
#     may be more efficient to solve the minimization problems serially (i.e., by
#     looping over each problem in the batch one at a time).
# 
#     Args:
#         x: batch of initial parameters.
#         y: corresponding batch of target data.
#         get_y_hat: function taking components of `x` as arguments and returning a
#             prediction of `y`. Must be `vmap`able.
#         eps_metric: tolerance for deciding when to decrease :math:`\\lambda`.
#         lam: damping parameter :math:`\\lambda`.
#         eps_grad: gradient convergence threshold. If `None`, doesn't check this
#             convergence metric.
#         eps_x: parameter convergence threshold. If `None`, doesn't check this
#             convergence metric.
#         eps_reduced_chi2: reduced :math:`\\chi^2` convergence threshold. If `None`,
#             doesn't check this convergence metric.
#         lam_decrease_factor: factor by which to decrease :math:`\\lambda` after accepting
#             an update.
#         lam_increase_factor: factor by which to increase :math:`\\lambda` after rejecting
#             an update.
#         max_iters: maximum number of iterations.
#         small_number: this is added to the diagonal of the approximate Hessian and the
#             parameter values to avoid inverting a singular matrix or dividing by zero.
#         lam_min: minimum value permitted for :math:`\\lambda`.
#         lam_max: maximum value permitted for :math:`\\lambda`.
# 
#     Notes:
#         - Switch to jacfwd
#         - Should be able to get value and Jacobian simultaneously
#     """
#     if len(xs) != len(ys):
#         raise ValueError("x and y must having matching batch dimension")
# 
#     b, n = xs.shape
#     m = ys.shape[1]
#     if m - n + 1 <= 0:
#         raise ValueError(
#             "number of data points per batch must be at least the number of parameters "
#             "minus 1"
#         )
# 
#     # Make vmapd functions
#     argnums = tuple(range(n))
#     get_y_hat_v = functorch.vmap(lambda x: get_y_hat(*x))
#     get_jac = functorch.vmap(lambda x: functorch.jacrev(get_y_hat, argnums)(*x))
# 
#     lams = torch.full((b,), lam, device=xs.device)
#     lam_min = torch.as_tensor(lam_min, device=xs.device)
#     lam_max = torch.as_tensor(lam_max, device=xs.device)
# 
#     H_perturbation = small_number * torch.eye(n, device=xs.device).repeat(b, 1, 1)
#     chi2s_prev = ((ys - get_y_hat_v(xs)) ** 2).sum(-1)
# 
#     for _ in range(max_iters):
#         # Reduced chi^2 convergence check
#         if (
#             eps_reduced_chi2 is not None
#             and chi2s_prev.max() / (m - n + 1) < eps_reduced_chi2
#         ):
#             break
# 
#         Js = torch.stack(get_jac(xs), -1)
#         y_hats = get_y_hat_v(xs)
#         dys = (ys - y_hats)[:, :, None]
#         JT_dys = Js.transpose(-2, -1) @ dys
# 
#         # Gradient convergence check
#         if eps_grad is not None and JT_dys.max() < eps_grad:
#             break
# 
#         # Propose an update
#         JTJs = Js.transpose(-2, -1) @ Js
#         H_approxs = (
#             lams[:, None, None]
#             * torch.diag_embed(torch.diagonal(JTJs, dim1=-2, dim2=-1))
#             + H_perturbation
#         )
#         # print(H_approxs)
#         dxs = torch.linalg.inv(JTJs + H_approxs) @ JT_dys
# 
#         # Parameter convergence check
#         if eps_x is not None and (dxs / (xs + small_number)).abs().max() < eps_x:
#             break
# 
#         # Decide which lams to decrease and which updates to accept
#         x_news = xs + dxs[:, :, 0]
#         chi2s = ((ys - get_y_hat_v(x_news)) ** 2).sum()
#         metrics = (chi2s_prev - chi2s) / (
#             (dxs * JT_dys).sum(-2)[:, 0] + (dxs * (H_approxs @ dxs)).sum(-2)[:, 0]
#         ).abs()
#         accept = metrics > eps_metric
#         # accept = (metrics.max() > eps_metric).repeat(b)
#         lams = torch.where(
#             accept,
#             torch.maximum(lams / lam_decrease_factor, lam_min),
#             torch.minimum(lams * lam_increase_factor, lam_max),
#         )
#         xs = torch.where(accept[:, None], x_news, xs)
#         chi2s_prev = torch.where(accept, chi2s, chi2s_prev)
#         # print(metrics)
#         # print(lams)
#         # print()
# 
#         # if metrics.max() > eps_metric:
#         #     xs = x_news
#         #     chi2s_prev = chi2s
#         #     lams = torch.full((b,), max(lams[0].item() / lam_decrease_factor, lam_min.item()), device=xs.device)
#         # else:
#         #     lams = torch.full((b,), min(lams[0].item() * lam_increase_factor, lam_max.item()), device=xs.device)
#         # print(lams[0].item())
# 
#         # accept = metrics > eps_metric
#         # lams = torch.where(
#         #     accept,
#         #     torch.maximum(lams / lam_decrease_factor, lam_min),
#         #     torch.minimum(lams * lam_increase_factor, lam_max),
#         # )
#         # xs = torch.where(accept[:, None], x_news, xs)
#         # chi2s_prev = torch.where(accept, chi2s, chi2s_prev)
# 
#     return xs
