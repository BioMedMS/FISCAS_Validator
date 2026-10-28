"""Render masks, measured MALDI signals and diffusion reconstructions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(directory, cell_mask=None):
    """Save one 2x2 comparison per channel from a completed DQA run."""
    if __package__:
        from .parameter_optimization import clean_l1, channel_label
    else:
        from parameter_optimization import clean_l1, channel_label
    directory = Path(directory)
    target = np.load(directory / 'target_msi.npy')
    clean = np.load(directory / 'simulator_msi.npy')
    final = np.load(directory / 'generated_final.npy')
    with np.load(directory / 'simulator_parameters.npz') as parameters:
        masses = parameters['masses']
        channels = parameters['channels']
        noise_applied = bool(parameters['noise_applied']
                             ) if 'noise_applied' in parameters else True
    if cell_mask is None:
        cell_mask = np.load(directory / 'cell_mask_preview.npy')
    cell_mask = np.asarray(cell_mask)
    if cell_mask.ndim != 2:
        raise ValueError('Cell mask must be a 2D label image')
    stride = max(1, int(np.ceil(max(cell_mask.shape) / 1600)))
    labels = cell_mask[::stride, ::stride]
    labels = np.ma.masked_where(labels == 0, labels)
    mask_cmap = plt.get_cmap('prism').copy()
    mask_cmap.set_bad('black')
    output = directory / 'renderings'
    output.mkdir(exist_ok=True)
    paths = []
    for channel, source_channel in enumerate(channels):
        mass = float(masses[channel]) if len(masses) else None
        measured = target[..., channel]
        simulated = clean[..., channel]
        noisy = final[..., channel]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout='constrained')
        try:
            axes[0, 0].imshow(labels, cmap=mask_cmap, interpolation='nearest')
            axes[0, 0].set_title('Cell masks')
            maximum = max(float(measured.max()), float(
                simulated.max()), float(noisy.max()), 1e-12)
            noisy_title = 'Diffusion with targeted noise' if noise_applied else 'Diffusion (noise disabled)'
            for ax, array, title in (
                (axes[0, 1], measured, 'Original MALDI signal'),
                (axes[1, 0], simulated, f'Diffusion without noise\nL1 = {
                 clean_l1(simulated, measured):.6f}'),
                    (axes[1, 1], noisy, f'{noisy_title}\nL1 = {clean_l1(noisy, measured):.6f}')):
                im = ax.imshow(array, cmap='viridis', vmin=0,
                               vmax=maximum, interpolation='nearest')
                ax.set_title(title)
            for ax in axes.flat:
                ax.axis('off')
            fig.colorbar(im, ax=[axes[0, 1], axes[1, 0], axes[1, 1]],
                         shrink=.8, label='Intensity')
            fig.suptitle(channel_label(source_channel, mass))
            filename = (f'comparison_channel{int(source_channel)}.png' if mass is None
                        else f'comparison_channel{channel}_mz{mass:g}.png')
            path = output / filename
            fig.savefig(path, dpi=180)
            paths.append(path)
        finally:
            plt.close(fig)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--cell-mask', type=Path,
                        help='Aligned label mask for older runs')
    args = parser.parse_args()
    mask = None
    if args.cell_mask:
        mask = np.load(args.cell_mask, mmap_mode='r')
        with np.load(args.run_dir / 'simulator_parameters.npz') as parameters:
            top, bottom, left, right = parameters['crop'].astype(int)
            scale = int(parameters['scaling_factor'])
        h, w = mask.shape
        mask = mask[top * scale:h - bottom * scale if bottom else h,
                    left * scale:w - right * scale if right else w]
    for path in render(args.run_dir, mask):
        print(path)
