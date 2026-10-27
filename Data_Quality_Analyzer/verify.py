"""Verify an aligned MALDI block from cell labels, with rendering on by default."""
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

if __package__:
    from .parameter_optimization import prepare_data, run_pipeline
else:
    from parameter_optimization import prepare_data, run_pipeline


@dataclass
class VerificationResult:
    output_dir: Path
    parameters: list

    @property
    def clean_msi(self):
        return np.load(self.output_dir / "simulator_msi.npy")

    @property
    def final_msi(self):
        return np.load(self.output_dir / "generated_final.npy")


def verify(cell_mask, maldi_block, scaling_factor=None, *, backend="auto",
           render=True, output_dir=None, mass_list=None, channels=None,
           flatten=True, crop=(0, 0, 0, 0), total_time=1., dt=.1,
           rtol=1e-8, apply_noise=True, noise_seed=42,
           score_rtol=.005, d_rtol=.25, alpha_atol=.1,
           d_values=None, alpha_values=None):
    """Estimate local cell totals, then global D/alpha using the shared simulator.

    Input arrays must already be aligned. MALDI can be HxW or HxWxC; mass labels
    are optional. An integer scaling factor can be inferred from array shapes.
    All results are saved before the optional rendering stage.
    """
    if not np.isfinite(total_time + dt) or total_time < 0 or dt <= 0:
        raise ValueError("Require finite time >= 0 and dt > 0")
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError("backend must be auto, gpu or cpu")
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must lie between zero and one")
    if noise_seed is not None and (not isinstance(noise_seed, (int, np.integer)) or noise_seed < 0):
        raise ValueError("noise_seed must be a nonnegative integer or None")
    if noise_seed is not None:
        noise_seed = int(noise_seed)
    if (not np.isfinite([score_rtol, d_rtol, alpha_atol]).all()
            or not 0 < score_rtol < 1 or d_rtol <= 0 or alpha_atol <= 0):
        raise ValueError("Invalid convergence tolerances")
    if scaling_factor is not None and (not np.isfinite(scaling_factor)
                                       or scaling_factor < 1 or int(scaling_factor) != scaling_factor):
        raise ValueError("scaling_factor must be a positive integer")
    if scaling_factor is not None:
        scaling_factor = int(scaling_factor)
    for values, name in ((d_values, "D"), (alpha_values, "alpha")):
        if values is not None:
            values = np.asarray(values, dtype=float)
            if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
                    or np.any(values < 0) or (name == "alpha" and np.any(values > 1))):
                raise ValueError(f"Invalid explicit {name} candidates")
    data = prepare_data(cell_mask, maldi_block, mass_list, crop=crop,
                        scale=scaling_factor, flatten=flatten, channel_indexes=channels)
    output = Path(output_dir) if output_dir is not None else Path(
        __file__).resolve().parent / "output" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    winners = run_pipeline(*data, output, backend=backend, render=render,
                           flatten=flatten, crop=crop, total_time=total_time, dt=dt, rtol=rtol,
                           apply_noise=apply_noise, noise_seed=noise_seed, score_rtol=score_rtol,
                           d_rtol=d_rtol, alpha_atol=alpha_atol, ds=d_values, alphas=alpha_values)
    return VerificationResult(output, winners)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent / 'input',
                        help='Folder containing cell_mask.npy and maldi_block.npy')
    parser.add_argument('--cell-mask', type=Path,
                        help='Override the mask file in --input-dir')
    parser.add_argument('--maldi-block', type=Path,
                        help='Override the MALDI file in --input-dir')
    parser.add_argument('--mass-list', type=Path)
    parser.add_argument('--channels', type=int, nargs='+')
    parser.add_argument('--scale', type=int)
    parser.add_argument('--backend', choices=['auto', 'gpu', 'cpu'], default='auto')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--flatten', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--noise', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--noise-seed', type=int, default=42)
    parser.add_argument('--time', type=float, default=1.)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--rtol', type=float, default=1e-8)
    parser.add_argument('--crop', type=int, nargs=4, default=[0, 0, 0, 0])
    parser.add_argument('--score-rtol', type=float, default=.005)
    parser.add_argument('--d-rtol', type=float, default=.25)
    parser.add_argument('--alpha-atol', type=float, default=.1)
    parser.add_argument('--d-values', type=float, nargs='+')
    parser.add_argument('--alpha-values', type=float, nargs='+')
    args = parser.parse_args(argv)
    cell_mask = args.cell_mask or args.input_dir / 'cell_mask.npy'
    maldi_block = args.maldi_block or args.input_dir / 'maldi_block.npy'
    mass_list = args.mass_list
    if mass_list is None and args.maldi_block is None and (args.input_dir / 'mass_list.npy').is_file():
        mass_list = args.input_dir / 'mass_list.npy'
    for path in (cell_mask, maldi_block, mass_list):
        if path is not None and not path.is_file():
            parser.error(f'Input file missing: {
                         path}. Place cell_mask.npy and maldi_block.npy in --input-dir or provide explicit file paths.')
    result = verify(np.load(cell_mask, mmap_mode='r'),
                    np.load(maldi_block, mmap_mode='r'), args.scale,
                    mass_list=np.load(mass_list) if mass_list else None,
                    channels=args.channels, backend=args.backend, output_dir=args.output_dir,
                    render=args.render, flatten=args.flatten, apply_noise=args.noise,
                    noise_seed=args.noise_seed, total_time=args.time, dt=args.dt,
                    rtol=args.rtol, crop=args.crop, score_rtol=args.score_rtol,
                    d_rtol=args.d_rtol, alpha_atol=args.alpha_atol,
                    d_values=args.d_values, alpha_values=args.alpha_values)
    print(f"Verification complete: {result.output_dir}")
    return result


if __name__ == '__main__':
    main()
