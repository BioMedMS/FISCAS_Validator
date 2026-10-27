"""Estimate cell intensities and optimize channel-level diffusion parameters."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canvas_generation import CanvasGenerator, render_simulator_source
try:
    from .diffusion_estimator import estimate_mu_from_distribution
except ImportError:
    from diffusion_estimator import estimate_mu_from_distribution


def clean_l1(prediction, target):
    denominator = np.abs(prediction).sum() + np.abs(target).sum()
    return float(np.abs(prediction - target).sum() / denominator) if denominator else 0.


def channel_label(channel, mass):
    return f"Channel {int(channel)}" if mass is None else f"m/z {float(mass):.8f}"


def fit_global_gain(prediction, target):
    """Exact nonnegative minimizer of symmetric normalized L1 in scalar gain.

    Breakpoints are target/prediction. The derivative changes sign where the
    cumulative normalized prediction and target weights first sum to one.
    """
    p, y = np.maximum(prediction, 0).ravel(), np.asarray(target).ravel()
    ps, ys = float(p.sum()), float(y.sum())
    if ys == 0:
        return 0.
    positive = p > 0
    if ps == 0 or not np.any(y[positive] > 0):
        return 1.
    ratios = y[positive] / p[positive]
    order = np.argsort(ratios, kind="stable")
    weights = p[positive][order] / ps + y[positive][order] / ys
    index = min(int(np.searchsorted(np.cumsum(weights), 1.)), len(order) - 1)
    gain = float(ratios[order[index]])
    return 1. if clean_l1(p, y) <= clean_l1(gain * p, y) + 1e-14 else gain


def prepare_data(raw, msi, all_masses=None, crop=(0, 0, 0, 0),
                 scale=None, flatten=True, channel_indexes=None):
    """Validate aligned inputs, select channels and compact cell labels."""
    raw, msi = np.asarray(raw), np.asarray(msi)
    if msi.ndim == 2:
        msi = msi[..., None]
    if raw.ndim != 2 or msi.ndim != 3:
        raise ValueError("Expected a 2D segmentation and 3D MSI")
    if not raw.size or 0 in msi.shape:
        raise ValueError("Cell mask and MALDI block must be nonempty")
    if all_masses is not None:
        all_masses = np.asarray(all_masses, dtype=float).ravel()
        if len(all_masses) != msi.shape[-1] or not np.isfinite(all_masses).all():
            raise ValueError("Provide one finite mass per MSI channel")
    if scale is None:
        scale = int(
            round((raw.shape[0] / msi.shape[0] + raw.shape[1] / msi.shape[1]) / 2))
    if scale < 1 or tuple(int(np.ceil(s / scale)) for s in raw.shape) != msi.shape[:2]:
        raise ValueError(
            "Mask/MSI dimensions do not align at the integer scale; provide aligned inputs and --scale")
    t, b, l, r = crop
    h, w = msi.shape[:2]
    if min(crop) < 0 or t + b >= h or l + r >= w:
        raise ValueError("Crop must leave a nonempty image")
    if channel_indexes is not None:
        supplied = np.asarray(channel_indexes)
        if (supplied.ndim != 1 or supplied.dtype.kind not in "iuf"
                or not np.isfinite(supplied).all()
                or not np.all(supplied == np.floor(supplied))):
            raise ValueError("Channel indices must be a 1D sequence of finite integers")
        if np.any(supplied < 0) or np.any(supplied >= msi.shape[-1]):
            raise ValueError("Selected channel index is outside the input mass list")
        channels = supplied.astype(np.intp)
    else:
        channels = np.arange(msi.shape[-1], dtype=int)
    if len(channels) == 0:
        raise ValueError("Select at least one channel")
    if len(set(channels)) != len(channels):
        raise ValueError("Selected channel indices must be unique")
    target = np.asarray(msi[t:h - b, l:w - r, :][..., channels], dtype=np.float64)
    raw = np.asarray(raw[t * scale:min((h - b) * scale, raw.shape[0]),
                         l * scale:min((w - r) * scale, raw.shape[1])])
    if not np.isfinite(raw).all() or np.any(raw < 0) or not np.all(raw == np.floor(raw)):
        raise ValueError("Segmentation labels must be finite nonnegative integers")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("MSI must contain finite nonnegative intensities")
    if flatten:
        target = np.minimum(target, np.percentile(
            target, 99, axis=(0, 1), keepdims=True))
    ids, counts = np.unique(raw, return_counts=True)
    ids, counts = ids[ids > 0], counts[ids > 0]
    if not len(ids):
        raise ValueError("Cell mask must contain at least one cell after cropping")
    # Compact labels preserve equality/barriers while supporting arbitrary IDs.
    mask = np.zeros(raw.shape, dtype=np.int32)
    for top in range(0, raw.shape[0], 128):
        block = raw[top:top + 128]
        mask[top:top + 128] = np.where(block > 0, np.searchsorted(ids, block) + 1, 0)
    selected_masses = None if all_masses is None else all_masses[channels]
    return mask, target, ids, counts, selected_masses, channels, scale


def initial_cell_totals(mask, target, scale, n_cells):
    """Estimate cell totals from mask fractions in row chunks."""
    totals = np.zeros((target.shape[-1], n_cells), dtype=float)
    for top in range(0, mask.shape[0], 128):
        bottom = min(mask.shape[0], top + 128)
        labels = mask[top:bottom].ravel()
        yy = np.arange(top, bottom) // scale
        xx = np.arange(mask.shape[1]) // scale
        for c in range(target.shape[-1]):
            weights = target[yy[:, None], xx[None, :], c] / float(scale * scale)
            totals[c] += np.bincount(labels, weights=weights.ravel(),
                                     minlength=n_cells + 1)[1:]
    return totals


def data_estimates(mask, target, counts, scale):
    h, w = target.shape[:2]
    padded = np.zeros((h * scale, w * scale), dtype=np.float32)
    padded[:mask.shape[0], :mask.shape[1]] = mask > 0
    fraction = padded.reshape(h, scale, w, scale).mean(axis=(1, 3))
    radius = float(np.median(np.sqrt(counts / np.pi))) if len(counts) else 0.
    estimates = [estimate_mu_from_distribution(target[..., c], fraction,
                 scaling_factor=scale, median_cell_radius_ms=radius / scale)
                 for c in range(target.shape[-1])]
    return radius, estimates


class CandidateSearch:
    """Score and retain full-image parameter candidates."""

    def __init__(self, forward, target, source, total_time, dt, scale, leaky,
                 output, channel, mass, progress, label=None):
        self.forward, self.target, self.source = forward, target, source
        self.time, self.dt, self.scale, self.leaky = total_time, dt, scale, leaky
        self.output, self.channel, self.mass, self.progress = output, channel, mass, progress
        self.label = label or channel_label(channel, mass)
        self.cache, self.best = {}, None
        self.no_diffusion_image = None

    def evaluate(self, d, alpha):
        d, alpha = float(d), float(alpha)
        if d < 0 or not 0 <= alpha <= 1 or not np.isfinite(d + alpha):
            raise ValueError("Invalid candidate")
        key = (d, alpha)
        if key in self.cache:
            return self.cache[key]
        start = time.perf_counter()
        last_print = start

        def step(index, total, iterations):
            nonlocal last_print
            self.progress(phase="simulating", channel=self.channel, mass=self.mass,
                          candidate_D=d, candidate_alpha=alpha,
                          timestep=index, timesteps=total, iterations=iterations,
                          candidates_evaluated=len(self.cache),
                          best_l1=None if self.best is None else self.best["l1"])
            if time.perf_counter() - last_print >= 10.:
                print(f"{self.label}: D={d:.6g}, alpha={alpha:.5g}, step {
                      index}/{total}, iterations={iterations}; candidate still computing", flush=True)
                last_print = time.perf_counter()

        no_diffusion = d == 0 or self.time <= self.dt
        if no_diffusion and self.no_diffusion_image is not None:
            image = self.no_diffusion_image
        else:
            image = self.forward.msi(self.source, d, alpha, self.time, self.dt,
                                     self.scale, self.leaky, callback=step)
            image = np.maximum(np.asarray(image, dtype=float), 0.)
            if no_diffusion:
                self.no_diffusion_image = image
        gain = fit_global_gain(image, self.target)
        score = clean_l1(gain * image, self.target)
        seconds = time.perf_counter() - start
        record = dict(channel=self.channel, mass=self.mass, D=d, alpha=alpha,
                      gain=gain, l1=score, seconds=seconds,
                      objective="clean_full_image_symmetric_L1", solver_rtol=self.forward.rtol)
        self.cache[key] = record
        candidate_path = self.output / "candidates" / \
            f"channel{self.channel}_candidate{len(self.cache):04d}.npy"
        np.save(candidate_path, gain * image)
        record["image_file"] = str(candidate_path.relative_to(self.output))
        # Break score ties by D, then alpha.
        if self.best is None or (score, d, alpha) < (self.best["l1"], self.best["D"], self.best["alpha"]):
            self.best = record.copy()
            np.save(self.output / f"best_channel{self.channel}.npy", gain * image)
            (self.output /
             f"best_channel{self.channel}.json").write_text(json.dumps(self.best, indent=2))
        with (self.output / "candidates.csv").open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(record)).writerow(record)
        self.progress(phase="candidate_complete", **record,
                      best_l1=self.best["l1"], candidates_evaluated=len(self.cache))
        print(f"{self.label}: D={d:.6g}, alpha={alpha:.5g}, gain={gain:.5g}, full L1={
              score:.6f}; best={self.best['l1']:.6f}; {seconds:.1f}s", flush=True)
        return record


def _local_profile(search, reference_d, score_rtol, d_bounds=None):
    """Bracket the incumbent along D and alpha, using actual scored points."""
    while True:
        d, alpha = search.best["D"], search.best["alpha"]
        ds = sorted(set(k[0] for k in search.cache))
        aa = sorted(set([0., 1., *[k[1] for k in search.cache]]))
        below = [x for x in ds if x < d]
        above = [x for x in ds if x > d]
        lo = max(below) if below else 0.
        hi = min(above) if above else (d * 4 if d else reference_d)
        if d_bounds is not None:
            lo = max(lo, d_bounds[0])
            hi = min(hi, d_bounds[1])
        if hi < d or (hi == d and d_bounds is None) or not np.isfinite(hi):
            raise RuntimeError("No representable upper diffusion bracket")
        alo = max([x for x in aa if x < alpha], default=alpha)
        ahi = min([x for x in aa if x > alpha], default=alpha)
        previous = search.best.copy()
        for point in (lo, hi):
            search.evaluate(point, alpha)
        for point in (alo, ahi):
            search.evaluate(d, point)
        score_tolerance = score_rtol * max(search.best["l1"], np.finfo(float).eps)
        if (search.best["D"], search.best["alpha"]) == (d, alpha):
            d_scores = [search.cache[(point, alpha)]["l1"] for point in (lo, hi)]
            a_scores = [search.cache[(d, point)]["l1"] for point in (alo, ahi)]
            return dict(D_low=lo, D_high=hi, alpha_low=alo, alpha_high=ahi,
                        D_flat=bool(max(d_scores) -
                                    search.best["l1"] <= score_tolerance),
                        alpha_flat=bool(
                            max(a_scores) - search.best["l1"] <= score_tolerance),
                        D_at_upper_bound=bool(d_bounds is not None and d == d_bounds[1]))
        improvement = previous["l1"] - search.best["l1"]
        if d_bounds is None and search.best["D"] >= hi and not above and improvement <= score_tolerance:
            # Record a flat upper profile as an unidentified diffusion coefficient.
            return dict(D_low=lo, D_high=search.best["D"],
                        alpha_low=alo, alpha_high=ahi,
                        D_flat=True, alpha_flat=False, upper_flat=True)


def refine_to_convergence(search, reference_d, score_rtol=.005,
                          d_rtol=.25, alpha_atol=.1, d_bounds=None):
    """Refine D and alpha until the score and parameter profiles stabilize.

    Require two small-improvement sweeps and a narrow or locally flat
    neighbourhood for each parameter.
    """
    if not 0 < score_rtol < 1 or d_rtol <= 0 or alpha_atol <= 0:
        raise ValueError("Convergence tolerances must be positive")
    reference_d = max(float(reference_d), np.finfo(float).tiny)
    if d_bounds is not None:
        if (len(d_bounds) != 2 or not np.isfinite(d_bounds).all()
                or not 0 <= d_bounds[0] <= d_bounds[1]):
            raise ValueError("Invalid explicitly restricted diffusion interval")
    history = []
    stable = 0
    round_index = 0
    output_path = search.output / f"convergence_channel{search.channel}.csv"
    while True:
        round_index += 1
        before = search.best["l1"]
        profile = _local_profile(search, reference_d, score_rtol, d_bounds)
        d, alpha = search.best["D"], search.best["alpha"]
        low, high = profile["D_low"], profile["D_high"]
        if not profile["D_flat"]:
            points = []
            if low < d:
                points.append(np.sqrt(low * d) if low > 0 else d / 2)
            if d < high:
                points.append(np.sqrt(d * high) if d > 0 else high / 2)
            for point in points:
                search.evaluate(point, alpha)
        # Update the alpha profile at the current best D.
        profile = _local_profile(search, reference_d, score_rtol, d_bounds)
        d, alpha = search.best["D"], search.best["alpha"]
        if not profile["alpha_flat"]:
            for point in ((profile["alpha_low"] + alpha) / 2,
                          (alpha + profile["alpha_high"]) / 2):
                search.evaluate(d, point)
        profile = _local_profile(search, reference_d, score_rtol, d_bounds)
        relative = (before - search.best["l1"]) / max(before, np.finfo(float).eps)
        stable = stable + 1 if relative <= score_rtol else 0
        d_width = (profile["D_high"] - profile["D_low"]) / max(search.best["D"],
                                                               reference_d if search.best["D"] == 0 else np.finfo(float).tiny)
        a_width = profile["alpha_high"] - profile["alpha_low"]
        d_done = profile["D_flat"] or d_width <= d_rtol
        a_done = profile["alpha_flat"] or a_width <= alpha_atol
        converged = stable >= 2 and d_done and a_done
        record = dict(round=round_index, D=search.best["D"], alpha=search.best["alpha"],
                      clean_full_l1=search.best["l1"], relative_improvement=relative,
                      D_relative_bracket=d_width, alpha_bracket_width=a_width,
                      stable_sweeps=stable, candidates=len(search.cache),
                      **profile, converged=converged)
        history.append(record)
        fields = list(dict.fromkeys(key for row in history for key in row))
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        reason = ("flat_upper_edge_unidentified_D" if profile.get("upper_flat")
                  else "converged_at_search_boundary" if profile.get("D_at_upper_bound")
                  else "converged_neighbourhood")
        status = dict(record, score_rtol=score_rtol, D_rtol=d_rtol, alpha_atol=alpha_atol,
                      stop_reason=reason if converged else "refining")
        (search.output /
         f"convergence_channel{search.channel}.json").write_text(json.dumps(status, indent=2))
        search.progress(phase="refinement_sweep", channel=search.channel, mass=search.mass,
                        convergence=status, best_l1=search.best["l1"])
        print(f"{search.label} refinement {round_index}: D={search.best['D']:.6g}, alpha={search.best['alpha']:.5g}, L1={search.best['l1']:.7f}, improvement={
              relative:.2%}, D bracket={d_width:.1%}, alpha bracket={a_width:.3g}; {'CONVERGED' if converged else 'continuing'}", flush=True)
        if converged:
            return status


def search_channel(search, initial_d, estimated_alpha, radius, explicit_ds=None,
                   explicit_alphas=None, adaptive=True, relative_improvement=1e-3,
                   refine=True, score_rtol=.005, d_rtol=.25, alpha_atol=.1):
    if explicit_ds is None:
        anchor = radius**2 / (2 *
                              search.time) if radius and search.time > search.dt else 0.
        ds = sorted(
            set([0., float(initial_d), *[anchor * f for f in (1 / 16, 1 / 4, 1., 4.)]]))
    else:
        ds = sorted(set(float(d) for d in explicit_ds))
    alphas = sorted(set([0., float(estimated_alpha), .5, 1.]
                    if explicit_alphas is None else explicit_alphas))
    if not ds or not alphas:
        raise ValueError("Candidate sets must not be empty")
    # Evaluate candidates nearest the input-derived D estimate first.
    for d in sorted(ds, key=lambda value: abs(np.log(max(value, 1e-300) / max(initial_d, 1e-300)))):
        for alpha in alphas:
            search.evaluate(d, alpha)
    if adaptive and explicit_ds is None and max(ds) > 0:
        # Expand an improving upper edge geometrically.
        while search.best["D"] == max(ds):
            previous = search.best["l1"]
            upper = max(ds) * 4.
            if not np.isfinite(upper):
                break
            ds.append(upper)
            for alpha in alphas:
                search.evaluate(upper, alpha)
            improvement = previous - search.best["l1"]
            if improvement <= relative_improvement * max(previous, np.finfo(float).eps):
                break
    if refine and explicit_ds is None and explicit_alphas is None and search.time > search.dt:
        reference = radius**2 / (2 * search.time) if radius else max(ds)
        # Restrict refinement to the initial interval when expansion is disabled.
        bounds = None if adaptive else (min(ds), max(ds))
        refine_to_convergence(search, reference, score_rtol, d_rtol, alpha_atol,
                              d_bounds=bounds)
        ds = sorted(set(key[0] for key in search.cache))
        alphas = sorted(set(key[1] for key in search.cache))
    return search.best, ds, alphas


def run_pipeline(mask, target, ids, counts, masses, channels, scale, output,
                 total_time=1., dt=.1, rtol=1e-8, backend="auto", ds=None,
                 alphas=None, adaptive=True, flatten=True, crop=(0, 0, 0, 0),
                 apply_noise=True, noise_seed=42, render=True, refine=True,
                 score_rtol=.005, d_rtol=.25, alpha_atol=.1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "candidates").mkdir()
        started = time.time()
        state = dict(status="running", phase="initializing", started_utc=datetime.now(timezone.utc).isoformat(),
                     grid_shape=list(mask.shape), target_shape=list(target.shape), historical_inputs=False)

        def progress(**values):
            if values.get("phase") == "simulating":
                for key in ("D", "alpha", "gain", "l1", "seconds", "image_file"):
                    state.pop(key, None)
            elif values.get("phase") == "candidate_complete":
                state["last_completed_candidate"] = {key: values[key] for key in
                                                     ("channel", "mass", "D", "alpha", "gain", "l1", "seconds") if key in values}
            state.update(values)
            state["elapsed_seconds"] = time.time() - started
            temporary = output / "progress.json.tmp"
            temporary.write_text(json.dumps(state, indent=2, default=lambda value: value.item(
            ) if isinstance(value, np.generic) else str(value)))
            temporary.replace(output / "progress.json")

        progress()
        np.save(output / "target_msi.npy", target)
        np.save(output / "cell_ids.npy", ids)
        totals = initial_cell_totals(mask, target, scale, len(ids))
        radius, estimates = data_estimates(mask, target, counts, scale)
        np.savez(output / "initial_parameters.npz", cell_total_intensities=totals,
                 initial_D=np.asarray(
                     [e["mu_est"] / total_time if total_time > dt else 0. for e in estimates]),
                 estimated_alpha=np.asarray([e["barrier_alpha_est"]
                                            for e in estimates]),
                 median_radius_px=radius, cell_ids=ids,
                 masses=np.asarray([] if masses is None else masses), scaling_factor=scale)
        forward = CanvasGenerator(mask, backend=backend, rtol=rtol)
        chosen_backend = forward.backend + "_float64"
        progress(backend=chosen_backend, phase="optimization")
        fields = ["channel", "mass", "D", "alpha", "gain", "l1",
                  "seconds", "objective", "solver_rtol", "image_file"]
        with (output / "candidates.csv").open("w", newline="") as handle:
            csv.writer(handle).writerow(fields)
        winners = []
        for c, source_channel in enumerate(channels):
            mass = None if masses is None else float(masses[c])
            label = channel_label(source_channel, mass)
            print(f"Starting {label}, full image, {chosen_backend}", flush=True)
            source = forward.source(render_simulator_source(mask, totals[c]))
            search = CandidateSearch(forward, target[..., c], source, total_time, dt,
                                     scale, True, output, c, mass, progress, label=label)
            initial_d = float(estimates[c]["mu_est"]) / \
                total_time if total_time > dt else 0.
            estimated_alpha = float(estimates[c]["barrier_alpha_est"])
            best, tested_ds, tested_alphas = search_channel(search, initial_d, estimated_alpha, radius,
                                                            ds, alphas, adaptive,
                                                            refine=refine, score_rtol=score_rtol,
                                                            d_rtol=d_rtol, alpha_atol=alpha_atol)
            winners.append(best)
            (output / f"search_channel{c}.json").write_text(json.dumps(dict(
                winner=best, tested_D=tested_ds, tested_alpha=tested_alphas,
                candidate_count=len(search.cache), objective="clean_full_image_symmetric_L1",
                tested_pairs=[{"D": d, "alpha": a} for d, a in search.cache],
                upper_edge_best=bool(best["D"] == max(tested_ds) and best["D"] > 0),
                selection="minimum_of_all_evaluated_full_image_candidates"), indent=2))
            progress(phase="channel_complete", channel=c, best_l1=best["l1"])
            del source, search
        return save_results(output, target, ids, masses, channels, scale, totals, winners,
                            total_time, dt, rtol, chosen_backend, flatten, crop,
                            apply_noise, noise_seed, progress, render, mask=mask)
    except Exception as exc:
        # Record failure after this run has created its output directory.
        path = output / "progress.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(status="failed", phase="error", error=str(exc),
                     error_type=type(exc).__name__)
        temporary = output / "progress.json.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)
        raise


def save_results(output, target, ids, masses, channels, scale, totals, winners,
                 total_time, dt, rtol, chosen_backend, flatten, crop,
                 apply_noise, noise_seed, progress, render, mask=None):
    """Save the reconstruction, parameters and L1 scores before rendering."""
    generated = np.stack(
        [np.load(output / f"best_channel{c}.npy") for c in range(len(channels))], axis=-1)
    if noise_seed is None:
        noise_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    final_totals = totals * np.asarray([row["gain"] for row in winners])[:, None]
    np.save(output / "simulator_msi.npy", generated)
    np.savez(output / "simulator_parameters.npz", cell_total_intensities=final_totals,
             diffusion_coefficients=[row["D"] for row in winners],
             diffusion_barrier_alphas=[row["alpha"] for row in winners],
             diffusion_total_time=total_time, diffusion_dt=dt, leaky_outer_boundary=True,
             scaling_factor=scale, masses=np.asarray([] if masses is None else masses),
             channels=channels, cell_ids=ids,
             crop=np.asarray(crop), flatten_top_percent=flatten, solver_rtol=rtol,
             backend=chosen_backend, objective="clean_full_image_symmetric_L1",
             noise_applied=apply_noise, noise_seed=str(noise_seed))
    with (output / "parameters_by_mass.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["channel", "mass", "D", "alpha", "gain", "clean_full_l1"])
        writer.writeheader()
        for source_channel, row in zip(channels, winners):
            writer.writerow(dict(channel=int(
                source_channel), mass=row["mass"], D=row["D"], alpha=row["alpha"], gain=row["gain"], clean_full_l1=row["l1"]))
    final = generated
    if apply_noise:
        try:
            from .noise_model import apply_noise_model
        except ImportError:
            from noise_model import apply_noise_model
        final = apply_noise_model(generated, target, seed=noise_seed, rank_match=True)
    np.save(output / "generated_final.npy", final)
    with (output / "simulator_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "channel", "l1_error"])
        for c, source_channel in enumerate(channels):
            label = channel_label(source_channel, None if masses is None else masses[c])
            writer.writerow(["simulator_engine", label, clean_l1(
                generated[..., c], target[..., c])])
        if apply_noise:
            for c, source_channel in enumerate(channels):
                label = channel_label(
                    source_channel, None if masses is None else masses[c])
                writer.writerow(["simulator_plus_noise", label,
                                clean_l1(final[..., c], target[..., c])])
    if mask is not None:
        # Save a display-resolution label image for later rendering.
        stride = max(1, int(np.ceil(max(mask.shape) / 1600)))
        np.save(output / "cell_mask_preview.npy", mask[::stride, ::stride])
    progress(status="complete", phase="saved", completed_channels=len(channels))
    if render:
        try:
            if __package__:
                from .render_comparison import render as render_comparison
            else:
                from render_comparison import render as render_comparison
            render_comparison(output)
        except Exception as exc:
            progress(status="complete", phase="rendering_failed", render_error=str(exc))
            print(f"Optimization results are saved; rendering failed: {
                  exc}", flush=True)
    return winners
