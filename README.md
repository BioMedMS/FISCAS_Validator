# FISCAS Validator

Generate synthetic single-cell MALDI images and evaluate registration, segmentation and cell spectra. The Data Quality Analyzer (DQA) fits simulation parameters to an aligned cell mask and measured MALDI block. Both use the shared canvas generator.

## Install

Use Python 3.12 or newer on Windows, Linux or macOS. Run the examples from the repository root.

Create a virtual environment locally. Depending on your Python installation, use `python3` (Linux/macOS) or `py -3` (Windows) in place of `python` for this step.

```text
python -m venv .venv
```

Activate it in your shell:

| Shell | Command |
|---|---|
| Linux/macOS Bash or Zsh | `source .venv/bin/activate` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

After activation, the following commands use `python` on all three systems:

```text
python -m pip install -r requirements.txt
```

For NVIDIA CUDA support on Windows or Linux:

```text
python -m pip install "cupy-cuda12x[ctk]>=14,<15"
```

CUDA requires a compatible NVIDIA driver and enough GPU memory for the input size. `cpu` selects CPU computation; `gpu` requires CUDA; `auto` uses CUDA when available and otherwise CPU. Backend selection controls diffusion computation.

## Simulator

Supply a cell library with matching cell images and segmentation masks. Image filenames must contain `cell`; mask filenames must contain `seg`. The two sorted file lists are paired by position. Cell images need at least three channels; microscopy channels 1 and 2 are used. Masks contain positive integer cell IDs and zero background.

```text
Cell_Library/example/
  001_cell.tif
  001_seg.tif
  002_cell.tif
  002_seg.tif
```

Edit the cell-library paths and settings in the `__main__` block of `FISCAS_Validator.py`, then run:

```text
python -c "from pathlib import Path; Path('output').mkdir(exist_ok=True)"
python FISCAS_Validator.py
```

For a custom configuration:

```python
from pathlib import Path
from FISCAS_Validator import CellType, Validator

masses = [369.35, 744.55, 760.58, 885.79]
cells = CellType(
    name="example",
    parameters={
        "path": "Cell_Library/example",
        **{mz: ["normal", 20, 5] for mz in masses},
    },
    population=1.0,
)
output = Path("output/simulation")
output.mkdir(parents=True, exist_ok=True)

simulation = Validator(
    cell_types=[cells],
    mass_list=masses,
    output=str(output),
    canvas_size=500,
    ms_spacing=5,
    cell_density=0.1,
    diffusion_total_time=10,
    diffusion_coeff=[1, 2, 3, 4],
    diffusion_barrier_alpha=[0.1, 0.2, 0.3, 0.4],
    computation_backend="auto",
    use_perfect_registration=True,
    use_perfect_segmentation=True,
    texture=None,
    plot=False,
    save_img=True,
)
metrics = simulation.main()
```

`CellType` takes a name, a parameter dictionary and a relative population weight. Each m/z maps to `[distribution, location, scale]`: `normal` uses mean and standard deviation; `uniform` uses centre and half-width. The sampled absolute value multiplies the cell's distance-transform profile. `parameters["path"]` selects its cell-library directory.

### Simulator settings

These are the `Validator` constructor defaults. Diffusion lists follow `mass_list` order; scalar diffusion values apply to every channel.

| Setting | Default | Meaning |
|---|---|---|
| `cell_types` | required | List of `CellType` objects to sample. |
| `output` | `./output` | Existing directory for results and saved images. |
| `canvas_size` | `2000` | Square microscopy canvas side length in µm. |
| `mic_spacing` | `0.6897` | Requested microscopy pixel size in µm; adjusted to an integer MSI-to-microscopy ratio. |
| `ms_spacing` | `1` | MALDI pixel size in µm. Must be at least `mic_spacing`. |
| `cell_density` | `0.2` | Target fraction of the microscopy canvas covered by cells. |
| `max_cell_placement_iterations` | `1000` | Placement attempt limit. |
| `mass_list` | `[369.35, 760.58, 744.55, 958.78]` | Simulated m/z channels. Each cell type needs parameters for each mass. |
| `seed` | `0` | Random seed for cell sampling and placement. |
| `rotation` | `0` | Geometric perturbation angle in degrees. |
| `scale` | `1.0` | Geometric perturbation scale factor. |
| `use_perfect_registration` | `False` | Recover alignment from the known perturbation transform. |
| `teaching_imprecision` | `10` | Maximum landmark-placement offset in µm. |
| `use_perfect_segmentation` | `True` | Use ground-truth masks for spectrum extraction. `False` runs Cellpose. |
| `cell_diameter` | `0` | Cellpose diameter in pixels; `0` requests automatic estimation. |
| `remove_ambiguous` | `True` | Exclude MALDI pixels covering multiple cells from spectrum extraction. |
| `diffusion_total_time` | `100.0` | Total diffusion time; `0` disables diffusion. |
| `diffusion_dt` | `1` | Diffusion timestep. Use a value smaller than the total time. |
| `diffusion_coeff` | `[0.1, 1, 50, 10]` | Per-channel D in microscopy-grid pixel² per time unit. |
| `diffusion_barrier_alpha` | `[0.1, 0.3, 0.7, 0.5]` | Per-channel boundary permeability: `0` blocks crossing, `1` permits full crossing. |
| `diffusion_leaky_outer_boundary` | `True` | Reset outer canvas pixels to zero after each diffusion step. |
| `computation_backend` | `cpu` | Diffusion backend: `cpu`, `gpu` or `auto`. |
| `texture` | `registration` | Apply texture to the registration image; `all` affects all MALDI channels; `None` disables it. |
| `texture_path` | `./textures/texture.jpg` | Texture image file. |
| `texture_epsilon_add` | `0` | Additive texture scale, relative to the signal's 99th-percentile intensity. |
| `texture_epsilon_multiply` | `0` | Offset applied to the normalized multiplicative texture. |
| `plot` | `True` | Display result figures. |
| `save_img` | `True` | Save result figures. |
| `render_animation` | `False` | Save frames of pixel-to-cell spectrum assignment. |

`main()` returns dictionaries for canvas, registration, segmentation, spectra and clustering metrics. It also saves cell spectra, ground truth and mask mappings as pickle files in `output`.

## Data Quality Analyzer

Place aligned arrays in `Data_Quality_Analyzer/input/`:

| File | Contents |
|---|---|
| `cell_mask.npy` | A 2D integer label image. Zero is background; each cell has a positive ID. At least one cell must remain after cropping. |
| `maldi_block.npy` | Finite, nonnegative measured intensities, shaped `(height, width, channels)` or `(height, width)` for one channel. |
| `mass_list.npy` | Optional 1D array with one m/z per channel. Without it, results use channel numbers. |

The mask and MALDI block must already be aligned, with an integer microscopy-to-MALDI pixel ratio. Partial blocks at the image edges are supported.

```text
python Data_Quality_Analyzer/verify.py
python Data_Quality_Analyzer/verify.py --backend cpu --no-render
python Data_Quality_Analyzer/verify.py --backend gpu --channels 0 2
```

Explicit paths:

```text
python Data_Quality_Analyzer/verify.py --cell-mask "path/to/cell_mask.npy" --maldi-block "path/to/maldi_block.npy" --mass-list "path/to/mass_list.npy" --scale 23 --output-dir "Data_Quality_Analyzer/output/my_run"
```

The DQA estimates cell intensities first, then refines per-channel D and α against full-image L1, fitting a global intensity multiplier for every candidate. Targeted Gaussian noise is fitted to the residual afterward. Its spatial assignment uses the measured signal. Clean and noise-adjusted L1 scores are reported separately.

### DQA settings

| Option | Default | Meaning |
|---|---|---|
| `--input-dir` | DQA `input/` | Input folder; the default is beside the verifier script. |
| `--cell-mask` | `input/cell_mask.npy` | Override the mask path. |
| `--maldi-block` | `input/maldi_block.npy` | Override the MALDI path. |
| `--mass-list` | optional | Mass labels; detected automatically in folder mode. Supply explicitly when overriding the MALDI path. |
| `--channels` | all | Zero-based input channel indices. |
| `--scale` | inferred | Number of mask pixels per MALDI pixel along each axis. |
| `--backend` | `auto` | `cpu`, `gpu` or `auto`. |
| `--output-dir` | DQA `output/<timestamp>/` | New directory for the run. Existing directories are rejected. |
| `--render` / `--no-render` | on | Save a 2×2 comparison per channel: masks, measurement, clean diffusion and targeted-noise reconstruction. |
| `--flatten` / `--no-flatten` | on | Clip each selected measurement channel at its 99th percentile before fitting. |
| `--noise` / `--no-noise` | on | Apply residual-based targeted Gaussian noise after optimization. |
| `--noise-seed` | `42` | Noise sampling seed; saved with the parameters. |
| `--time` | `1.0` | Fixed total diffusion time. D is estimated for this time. `0` disables diffusion. |
| `--dt` | `0.1` | Diffusion timestep; use a value smaller than `--time` for diffusion. |
| `--rtol` | `1e-8` | Relative residual tolerance of the numerical diffusion solver. |
| `--crop` | `0 0 0 0` | Remove top, bottom, left and right borders before fitting, in MALDI pixels. |
| `--score-rtol` | `0.005` | Relative L1 improvement threshold for convergence. |
| `--d-rtol` | `0.25` | Relative D-bracket width used by the convergence check. |
| `--alpha-atol` | `0.1` | Absolute α-bracket width used by the convergence check. |
| `--d-values` | automatic | Explicit nonnegative D candidates; disables automatic interior refinement. |
| `--alpha-values` | automatic | Explicit α candidates in `[0, 1]`; disables automatic interior refinement. |

The automatic search starts from the measured cell geometry and signal distribution, expands an improving upper edge and refines between candidates. It stops after two small-improvement sweeps with sufficiently narrow or locally flat D and α neighbourhoods.

### DQA results

Parameters, arrays and L1 scores are saved before rendering:

- `parameters_by_mass.csv`: channel, optional mass, D, α, intensity multiplier and clean L1.
- `simulator_parameters.npz`: cell intensities, diffusion settings, preprocessing settings and noise seed.
- `simulator_comparison.csv`: clean and noise-adjusted L1.
- `simulator_msi.npy` and `generated_final.npy`: clean and final reconstructed MALDI blocks.
- `renderings/`: 2×2 comparisons with L1 scores on both reconstruction panels.
- `progress.json`, `candidates.csv` and `convergence_channel*.csv`: run status and search history.
