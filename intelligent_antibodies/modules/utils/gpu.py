"""
One place to decide how much of the machine a run is allowed to use.

Every entry point (both training scripts, the CLI pipeline, the Streamlit
app) used to configure TensorFlow differently: the training scripts asked for
unbounded `set_memory_growth`, while `main_pipeline.load_models` hardcoded a
5120 MB virtual-device cap that silently did nothing on a smaller card. Now
they all call `configure_compute()` here, which caps the run at a *fraction*
of the hardware -- 50% by default.

What "50%" means per device:

* GPU -- a hard virtual-device memory cap of 50% of the card's total VRAM.
  This is a real ceiling, not a growth hint: TensorFlow cannot allocate past
  it, so half the card stays free for anything else running on the desktop.
* CPU -- intra/inter-op thread pools capped at half the available cores, so
  a run cannot saturate every core. This matters here and now: TensorFlow in
  this environment reports no usable GPU ("Cannot dlopen some GPU
  libraries"), so training actually runs on the CPU.

Override with the `IA_COMPUTE_FRACTION` environment variable (e.g. `0.25`),
or by passing `fraction=` explicitly.
"""
from __future__ import annotations

import os

import tensorflow as tf

DEFAULT_FRACTION = 0.5

# TensorFlow refuses to change either setting after the runtime is
# initialized, and every entry point may call this more than once (the
# Streamlit app re-runs its script on every interaction). Remember that we
# already ran so repeat calls are a no-op instead of a RuntimeError.
_configured = False


def _requested_fraction(fraction: float | None) -> float:
    if fraction is None:
        try:
            fraction = float(os.environ.get("IA_COMPUTE_FRACTION", DEFAULT_FRACTION))
        except ValueError:
            fraction = DEFAULT_FRACTION
    # Below ~1% nothing can run; above 1.0 is not a fraction.
    return min(max(fraction, 0.01), 1.0)


def _total_gpu_memory_mb(gpu) -> int | None:
    """
    Total VRAM of `gpu` in MB, or None if it can't be determined.

    Deliberately asks the *driver*, not TensorFlow. `tf.config.experimental
    .get_memory_info` initializes the device as a side effect, and once that
    has happened both the memory cap and the thread cap below are rejected
    ("Virtual devices cannot be modified after being initialized") -- so
    querying TF to decide the cap is what prevents the cap from applying.
    """
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split()
        return int(out[0])
    except Exception:
        return None


def configure_compute(fraction: float | None = None, verbose: bool = True) -> dict:
    """
    Cap this process at `fraction` of the GPU's memory and the CPU's cores.

    Safe to call from anywhere, including repeatedly: the first call wins and
    later ones return the same summary without touching the runtime.

    Returns
    -------
    dict
        ``{"fraction", "gpu_limit_mb", "cpu_threads", "devices"}`` describing
        what was actually applied -- `gpu_limit_mb` is None when no GPU is
        usable, which is worth printing rather than assuming.
    """
    global _configured, _summary

    if _configured:
        return _summary

    frac = _requested_fraction(fraction)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    gpu_limit_mb: int | None = None

    for gpu in gpus:
        total_mb = _total_gpu_memory_mb(gpu)
        try:
            if total_mb:
                gpu_limit_mb = max(int(total_mb * frac), 512)
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_limit_mb)]
                )
            else:
                # Unknown card size: a growth limit is still better than
                # letting TF pre-allocate the whole device.
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"[compute] Could not cap {gpu.name} ({exc}); continuing with default config.")

    # Thread pools: half the cores, at least one. `os.cpu_count()` is the
    # right number here (not the physical core count) since TF sizes its
    # pools against logical CPUs.
    cpu_threads = max(int((os.cpu_count() or 2) * frac), 1)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(cpu_threads // 2, 1))
    except RuntimeError as exc:
        print(f"[compute] Could not cap CPU threads ({exc}); continuing with default config.")

    _summary = {
        "fraction": frac,
        "gpu_limit_mb": gpu_limit_mb,
        "cpu_threads": cpu_threads,
        "devices": [g.name for g in gpus],
    }
    _configured = True

    if verbose:
        pct = int(frac * 100)
        if gpus:
            where = f"GPU capped at {gpu_limit_mb} MB" if gpu_limit_mb else "GPU memory growth enabled"
        else:
            where = "no usable GPU detected -- running on CPU"
        print(f"[compute] {pct}% budget: {where}; {cpu_threads} intra-op CPU thread(s).")

    return _summary


_summary: dict = {}
