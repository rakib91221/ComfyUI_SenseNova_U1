"""Inference timing profiler.

Records model-load time and per-generation wall time (CUDA-synchronized so
GPU launch overhead doesn't hide inside Python). ``report()`` prints a summary
that also converts per-image time into per-token cost using a fixed image
patch size (the model's generation patchification factor).

Intended for quick, human-readable profiling from CLI scripts under
``examples/``. When ``enabled=False``, every context manager is a no-op and
``report()`` prints nothing, so it can be wired in unconditionally.

Typical usage::

    from sensenova_u1.utils import InferenceProfiler

    prof = InferenceProfiler(enabled=args.profile, device=args.device)
    with prof.time_load():
        engine = SenseNovaU1T2I(model_path)
    with prof.time_generate(width=2048, height=2048, batch=1):
        images = engine.generate(...)
    prof.report()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, List, Tuple

import torch

DEFAULT_IMAGE_PATCH_SIZE = 32


class InferenceProfiler:
    """Minimal wall-clock profiler for model loading + generation.

    Parameters
    ----------
    enabled : bool
        If False, every method is a no-op (zero overhead).
    device : str
        E.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``. Used to decide whether to
        ``torch.cuda.synchronize()`` around timed blocks.
    patch_size : int, optional
        Image-token grid factor used by :meth:`report` to translate wall time
        into ms/token. Defaults to :data:`DEFAULT_IMAGE_PATCH_SIZE`.
    """

    def __init__(
        self,
        enabled: bool,
        device: str = "cuda",
        patch_size: int = DEFAULT_IMAGE_PATCH_SIZE,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.patch_size = patch_size
        self.load_time: float = 0.0
        # records: (width, height, batch, seconds)
        self.gen_records: List[Tuple[int, int, int, float]] = []

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        if self.enabled and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def time_load(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.load_time = time.perf_counter() - t0

    @contextmanager
    def time_generate(self, width: int, height: int, batch: int = 1) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.gen_records.append((width, height, batch, time.perf_counter() - t0))

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print a summary. No-op when ``enabled=False``."""
        if not self.enabled:
            return
        print()
        print("=" * 64)
        print("Profile summary")
        print("=" * 64)
        print(f"  model load          : {self.load_time:8.3f} s")
        if not self.gen_records:
            print("  (no generations were timed)")
            return

        total_images = sum(b for (_w, _h, b, _s) in self.gen_records)
        total_time = sum(s for (_w, _h, _b, s) in self.gen_records)
        avg_per_image = total_time / total_images

        total_tokens = sum((w // self.patch_size) * (h // self.patch_size) * b for (w, h, b, _s) in self.gen_records)
        avg_tokens = total_tokens / total_images
        tokens_per_sec = total_tokens / total_time

        print(
            f"  generations         : {len(self.gen_records)} call(s), "
            f"{total_images} image(s) total, {total_time:.3f} s wall"
        )
        print(f"  avg per image       : {avg_per_image:8.3f} s")
        print(
            f"  image tokens        : patch_size={self.patch_size}, "
            f"avg {avg_tokens:.0f} tok/image ({int(avg_tokens):d})"
        )
        print(f"  throughput          : {tokens_per_sec:8.2f} tok/s")

        if len(self.gen_records) > 1:
            print("  per-call breakdown  :")
            for idx, (w, h, b, s) in enumerate(self.gen_records):
                tokens = (w // self.patch_size) * (h // self.patch_size) * b
                print(f"    [{idx + 1:>3}] {w}x{h} x{b}  {s:7.3f} s  ({tokens:>6d} tok, {tokens / s:8.2f} tok/s)")
        print("=" * 64)
