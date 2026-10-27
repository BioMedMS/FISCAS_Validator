"""CUDA conjugate-gradient solver for label-weighted implicit Euler diffusion.

Uses float64 arithmetic with DCT or Jacobi preconditioning.
"""
import numpy as np
import time as clock
from inspect import signature


_KERNEL = r'''
extern "C" __global__
void apply_grid(const double* x, const int* mask, double* y,
                const long long h, const long long w,
                const double coefficient, const double alpha,
                const int diagonal_only) {
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= h*w) return;
    long long row = i/w, col = i%w;
    double value = diagonal_only ? 1.0 : x[i];
    double result = value;
    int label = mask[i];
    if (col > 0) {
        double edge = coefficient * (label == mask[i-1] ? 1.0 : alpha);
        result += edge * (diagonal_only ? 1.0 : value-x[i-1]);
    }
    if (col+1 < w) {
        double edge = coefficient * (label == mask[i+1] ? 1.0 : alpha);
        result += edge * (diagonal_only ? 1.0 : value-x[i+1]);
    }
    if (row > 0) {
        double edge = coefficient * (label == mask[i-w] ? 1.0 : alpha);
        result += edge * (diagonal_only ? 1.0 : value-x[i-w]);
    }
    if (row+1 < h) {
        double edge = coefficient * (label == mask[i+w] ? 1.0 : alpha);
        result += edge * (diagonal_only ? 1.0 : value-x[i+w]);
    }
    y[i] = result;
}
'''


class GPUSimulator:
    def __init__(self, mask, rtol=1e-8):
        import cupy as cp
        from cupyx.scipy.fft import dctn, idctn
        from cupyx.scipy.sparse.linalg import LinearOperator, cg
        if 'rtol' not in signature(cg).parameters:
            raise ImportError(
                'The CUDA solver requires CuPy >= 14. Install cupy-cuda12x[ctk]>=14,<15.')
        self.cp, self.dctn, self.idctn = cp, dctn, idctn
        self.LinearOperator, self.cg = LinearOperator, cg
        self.mask = cp.asarray(mask, dtype=cp.int32)
        self.shape = self.mask.shape
        self.rtol = float(rtol)
        self.kernel = cp.RawKernel(_KERNEL, 'apply_grid')
        h, w = self.shape
        fy = 2 - 2 * cp.cos(cp.pi * cp.arange(h, dtype=cp.float64) / h)
        fx = 2 - 2 * cp.cos(cp.pi * cp.arange(w, dtype=cp.float64) / w)
        self.spectrum = fy[:, None] + fx[None, :]

    def source(self, array):
        return self.cp.asarray(array, dtype=self.cp.float64)

    def diffuse(self, source, coefficient, alpha, time, dt, leaky=True,
                callback=None, return_device=False):
        cp = self.cp
        source = cp.asarray(source, dtype=cp.float64)
        if source.shape != self.shape:
            raise ValueError("Source and mask shapes differ")
        if time < 0 or dt <= 0 or coefficient < 0 or not 0 <= alpha <= 1:
            raise ValueError("Invalid simulator parameters")
        u = source.copy()
        if time <= dt or coefficient == 0:
            # Apply the leaky-boundary reset for zero diffusion.
            if time > dt and leaky:
                u[0, :] = u[-1, :] = 0
                u[:, 0] = u[:, -1] = 0
            return u if return_device else cp.asnumpy(u)
        h, w = self.shape
        size = self.mask.size
        blocks = ((size + 255) // 256,)
        args = (np.int64(h), np.int64(w), np.float64(
            dt * coefficient), np.float64(alpha))

        def apply(vector):
            result = cp.empty_like(vector)
            self.kernel(blocks, (256,), (vector, self.mask, result, *args, np.int32(0)))
            return result

        if dt * coefficient >= 5 and alpha >= .01:
            inverse = 1.0 / (1.0 + dt * coefficient * self.spectrum)

            def precondition(vector):
                transformed = self.dctn(vector.reshape(h, w), type=2, norm='ortho')
                transformed *= inverse
                return self.idctn(transformed, type=2, norm='ortho', overwrite_x=True).ravel()
        else:
            inverse = cp.empty_like(u).ravel()
            self.kernel(blocks, (256,), (u, self.mask, inverse, *args, np.int32(1)))
            cp.reciprocal(inverse, out=inverse)

            def precondition(vector):
                return vector * inverse

        operator = self.LinearOperator((size, size), matvec=apply, dtype=cp.float64)
        preconditioner = self.LinearOperator(
            (size, size), matvec=precondition, dtype=cp.float64)
        steps = int(np.round(time / dt))
        for step in range(steps):
            rhs = u.ravel()
            iterations = 0
            last_notice = clock.monotonic()

            def iteration(_):
                nonlocal iterations, last_notice
                iterations += 1
                if callback is not None and clock.monotonic() - last_notice >= 5.:
                    callback(step + 1, steps, iterations)
                    last_notice = clock.monotonic()

            solution, info = self.cg(operator, rhs, x0=rhs.copy(), M=preconditioner,
                                     rtol=self.rtol, atol=0., maxiter=size, callback=iteration)
            if info != 0:
                raise RuntimeError(f"GPU simulator failed to converge: {info}")
            u = solution.reshape(h, w)
            if leaky:
                u[0, :] = u[-1, :] = 0
                u[:, 0] = u[:, -1] = 0
            if callback is not None:
                callback(step + 1, steps, iterations)
        return u if return_device else cp.asnumpy(u)

    def msi(self, source, coefficient, alpha, time, dt, scale, leaky=True, callback=None):
        cp = self.cp
        field = self.diffuse(source, coefficient, alpha, time, dt, leaky,
                             callback=callback, return_device=True)
        h, w = field.shape
        pad_h, pad_w = (-h) % scale, (-w) % scale
        if pad_h or pad_w:
            field = cp.pad(field, ((0, pad_h), (0, pad_w)))
        result = field.reshape(field.shape[0] // scale, scale,
                               field.shape[1] // scale, scale).sum(axis=(1, 3))
        return cp.asnumpy(result)
