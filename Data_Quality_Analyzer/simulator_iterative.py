"""Conjugate-gradient solver for label-weighted implicit Euler diffusion.

Uses a four-neighbour grid Laplacian and resets boundary pixels after each
step when leaky boundaries are enabled.
"""
import numpy as np
import time as clock
from inspect import signature
from scipy.sparse.linalg import LinearOperator, cg
from scipy.fft import dctn, idctn

if 'rtol' not in signature(cg).parameters:
    raise ImportError(
        'The CPU solver requires SciPy >= 1.12. Install the root requirements.txt.')


def diffuse_mask_iterative(mask, source, total_time, dt=0.1,
                           diffusion_coeff=1.0, barrier_alpha=0.1,
                           leaky_outer_boundary=True, rtol=1e-8, maxiter=None,
                           progress=False, preconditioner_type="auto", callback=None):
    """Diffuse one channel with matrix-free conjugate gradients.

    Each step targets ||A*u - rhs|| <= rtol*||rhs|| for a symmetric positive
    definite system, verified using the true residual. Already converged
    iterates skip CG. The default iteration budget is the SciPy default;
    residual correction continues while the true residual improves.
    Nonconvergence raises RuntimeError.
    """
    labels = np.asarray(mask)
    u = np.asarray(source, dtype=np.float64).copy()
    if labels.ndim != 2 or u.shape != labels.shape:
        raise ValueError("mask and source must have the same 2D shape")
    if dt <= 0 or total_time < 0 or diffusion_coeff < 0 or not 0 <= barrier_alpha <= 1:
        raise ValueError("invalid diffusion parameters")
    if not 0 < rtol < 1 or (maxiter is not None and maxiter < 1):
        raise ValueError("require 0 < rtol < 1 and maxiter >= 1")
    if total_time <= dt:
        return u
    h, w = labels.shape
    # Store dt-scaled horizontal and vertical edge weights.
    horizontal = np.where(labels[:, :-1] == labels[:, 1:],
                          dt * diffusion_coeff, dt * diffusion_coeff * barrier_alpha)
    vertical = np.where(labels[:-1, :] == labels[1:, :],
                        dt * diffusion_coeff, dt * diffusion_coeff * barrier_alpha)
    if preconditioner_type not in ("auto", "jacobi", "dct"):
        raise ValueError("preconditioner_type must be auto, jacobi or dct")
    chosen = preconditioner_type
    if chosen == "auto":
        chosen = "dct" if dt * diffusion_coeff >= 5 and barrier_alpha >= .01 else "jacobi"
    if chosen == "dct":
        # Inverse of the constant-coefficient Neumann operator.
        # For alpha > 0, the preconditioned condition number is bounded by 1/alpha.
        fy = 2 - 2 * np.cos(np.pi * np.arange(h) / h)
        fx = 2 - 2 * np.cos(np.pi * np.arange(w) / w)
        inverse_spectrum = 1.0 / (1.0 + dt * diffusion_coeff *
                                  (fy[:, None] + fx[None, :]))

        def precondition(vector):
            spectrum = dctn(vector.reshape(h, w), type=2, norm="ortho", workers=2)
            spectrum *= inverse_spectrum
            return idctn(spectrum, type=2, norm="ortho", workers=2, overwrite_x=True).ravel()
    else:
        diagonal = np.ones(labels.shape, dtype=np.float64)
        diagonal[:, :-1] += horizontal
        diagonal[:, 1:] += horizontal
        diagonal[:-1, :] += vertical
        diagonal[1:, :] += vertical
        inverse_diagonal = 1.0 / diagonal
        del diagonal

        def precondition(vector):
            return inverse_diagonal.ravel() * vector

    def multiply(vector):
        image = vector.reshape(h, w)
        result = image.copy()
        flux = horizontal * (image[:, :-1] - image[:, 1:])
        result[:, :-1] += flux
        result[:, 1:] -= flux
        del flux
        flux = vertical * (image[:-1, :] - image[1:, :])
        result[:-1, :] += flux
        result[1:, :] -= flux
        return result.ravel()

    size = labels.size
    operator = LinearOperator((size, size), matvec=multiply, dtype=np.float64)
    preconditioner = LinearOperator(
        (size, size), matvec=precondition, dtype=np.float64)
    if progress:
        print(f"Simulator preconditioner: {chosen}", flush=True)
    steps = int(np.round(total_time / dt))
    for step in range(steps):
        rhs = u.ravel()
        iterations = 0
        last_notice = clock.monotonic()

        def count_iteration(_):
            nonlocal iterations, last_notice
            iterations += 1
            if callback is not None and clock.monotonic() - last_notice >= 5.:
                callback(step + 1, steps, iterations)
                last_notice = clock.monotonic()

        tolerance = rtol * np.linalg.norm(rhs)
        solved = rhs.copy()
        residual = np.linalg.norm(rhs - multiply(solved))
        if not np.isfinite(residual):
            raise RuntimeError("Simulator initial residual is nonfinite")
        while residual > tolerance:
            previous_residual = residual
            solved, info = cg(operator, rhs, x0=solved, M=preconditioner,
                              rtol=rtol, atol=0.0, maxiter=maxiter,
                              callback=count_iteration)
            residual = np.linalg.norm(rhs - multiply(solved))
            if np.isfinite(residual) and residual <= tolerance:
                break
            if (not np.isfinite(residual) or info < 0 or residual >= previous_residual
                    or (maxiter is not None and info > 0)):
                raise RuntimeError(f"Simulator CG did not converge at step "
                                   f"{step + 1}/{steps} (info={info}, "
                                   f"residual={residual:.6g}, tolerance={
                                       tolerance:.6g}); "
                                   "no final image was accepted")
        u = solved.reshape(h, w)
        # Reset boundary values after each timestep.
        if leaky_outer_boundary:
            u[0, :] = u[-1, :] = 0.0
            u[:, 0] = u[:, -1] = 0.0
        if progress:
            print(f"Simulator step {
                  step + 1}/{steps}: {iterations} CG iterations", flush=True)
        if callback is not None:
            callback(step + 1, steps, iterations)
    return u
