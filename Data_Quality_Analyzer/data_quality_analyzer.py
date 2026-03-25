from parameter_estimator import estimate_cell_parameters
import numpy as np

mask = np.load("input/seg.npy")#[0:1068,0:996]
target_msi = np.load("input/msi_data.npy")[:, :, 0]#[0:1068,0:996]

result = estimate_cell_parameters(
    mask_canvas=mask,
    target_output=target_msi,
    mic_spacing=1.0,
    diffusion_total_time=0,
    diffusion_dt=0,
    diffusion_coeff=3.0,
    barrier_alpha=0.5,
    leaky_outer_boundary=True,
    finite_diff_eps=1.0,
    verbose=True,
)

print(result.optimized_intensities)
print(result.optimized_sigmas)
print(result.optimized_output.shape)
print(result.optimizer_result.success, result.optimizer_result.message)