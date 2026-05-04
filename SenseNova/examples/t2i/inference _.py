from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer

# import sensenova_u1
from src.sensenova_u1 import check_checkpoint_compatibility
from src.sensenova_u1.utils import DEFAULT_IMAGE_PATCH_SIZE, InferenceProfiler

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SEED = 42
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
from contextlib import AbstractContextManager, contextmanager
from layer_streaming import LayerStreamingWrapper
from collections.abc import Iterator
from typing import Callable, TypeVar
import gc
from safetensors.torch import load_file as _load_file
_M = TypeVar("_M", bound=torch.nn.Module)
T = TypeVar("T")

def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
@contextmanager
def _streaming_model(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    wrapped = LayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        prefetch_count=prefetch_count,
    )
    try:
        yield wrapped  # type: ignore[misc]
    finally:
        wrapped.teardown()
        wrapped.to("meta")
        cleanup_memory()
        # Flush the host (pinned) memory cache so that freed pinned pages are
        # returned to the OS.  Without this, sequential streaming models
        # (e.g. text encoder then transformer) exhaust host memory because the
        # CachingHostAllocator keeps freed blocks cached indefinitely.
        torch.cuda.synchronize(device=target_device)
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            print("Host empty cache cleanup failed; ignoring.", exc_info=True)


def set_gguf2meta_model(meta_model,model_state_dict,dtype,device,):
    from diffusers import GGUFQuantizationConfig
    from diffusers.quantizers.gguf import GGUFQuantizer

    g_config = GGUFQuantizationConfig(compute_dtype=dtype or torch.bfloat16)
    hf_quantizer = GGUFQuantizer(quantization_config=g_config)
    hf_quantizer.pre_quantized = True

    hf_quantizer._process_model_before_weight_loading(
        meta_model,
        device_map={"": device} if device else None,
        state_dict=model_state_dict
    )
    from diffusers.models.model_loading_utils import load_model_dict_into_meta
    load_model_dict_into_meta(
        meta_model, 
        model_state_dict, 
        hf_quantizer=hf_quantizer,
        device_map={"": device} if device else None,
        dtype=dtype
    )

    hf_quantizer._process_model_after_weight_loading(meta_model)
    
    del model_state_dict
    gc.collect()
    return meta_model.to(dtype=dtype)


def match_state_dict(meta_model, sd,show_num=10):

    meta_model_keys = set(meta_model.state_dict().keys())   
    state_dict_keys = set(sd.keys())

    # 打印匹配的键的数量
    matching_keys = meta_model_keys.intersection(state_dict_keys)
    print(f"Matching keys count: {len(matching_keys)}")
    
    # 打印不在 meta_model 中但在 state_dict 中的键（多余键）
    extra_keys = state_dict_keys - meta_model_keys
    if extra_keys:
        print(f"Extra keys in state_dict (not in meta_model): {len(extra_keys)}")
        for key in list(extra_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 打印不在 state_dict 中但在 meta_model 中的键（缺失键）
    missing_keys = meta_model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state_dict (not in state_dict): {len(missing_keys)}")
        for key in list(missing_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    
    # 如果需要，也可以打印部分匹配的键
    print(f"Sample matching keys: {list(matching_keys)[:5]}")

def load_gguf_checkpoint(gguf_checkpoint_path):

    from  diffusers.utils  import is_gguf_available, is_torch_available
    if is_gguf_available() and is_torch_available():
        import gguf
        from gguf import GGUFReader
        from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter,dequantize_gguf_tensor
    else:
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    parsed_parameters = {}
 
    for tensor in reader.tensors:
        name = tensor.name
        quant_type = tensor.tensor_type

        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = quant_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        if is_gguf_quant and quant_type not in SUPPORTED_GGUF_QUANT_TYPES:
            _supported_quants_str = "\n".join([str(type) for type in SUPPORTED_GGUF_QUANT_TYPES])
            raise ValueError(
                (
                    f"{name} has a quantization type: {str(quant_type)} which is unsupported."
                    "\n\nCurrently the following quantization types are supported: \n\n"
                    f"{_supported_quants_str}"
                    "\n\nTo request support for this quantization type please open an issue here: https://github.com/huggingface/diffusers"
                )
            )

        weights = torch.from_numpy(tensor.data.copy())
        parsed_parameters[name] = GGUFParameter(weights, quant_type=quant_type) if is_gguf_quant else weights
    
    del reader
    gc.collect()
    return parsed_parameters


SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2720, 1536),
    "9:16": (1536, 2720),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "4:3": (2368, 1760),
    "3:4": (1760, 2368),
    "1:2": (1440, 2880),
    "2:1": (2880, 1440),
    "1:3": (1152, 3456),
    "3:1": (3456, 1152),
}

DEFAULT_WIDTH, DEFAULT_HEIGHT = SUPPORTED_RESOLUTIONS["1:1"]


def _warn_if_unsupported(width: int, height: int) -> None:
    if (width, height) in SUPPORTED_RESOLUTIONS.values():
        return
    buckets = ", ".join(f"{r}->{w}x{h}" for r, (w, h) in SUPPORTED_RESOLUTIONS.items())
    print(
        f"[warn] ({width}x{height}) is outside the trained resolution set; "
        f"quality may degrade. Supported buckets: {buckets}"
    )


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """Invert the (img - mean) / std normalization back to [0, 1]."""
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    """Convert a [B, 3, H, W] float tensor in normalized space to a list of PIL images."""
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


class SenseNovaU1T2I:
    """Thin wrapper around ``AutoModel.from_pretrained``.

    Because ``sensenova_u1`` has already registered the config / model with
    transformers at import time, no ``trust_remote_code=True`` is needed.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        checkpoint: str | None = None,
    ) -> None:
        self.device = device
        self._last_think_text: str = ""
        config = AutoConfig.from_pretrained(model_path)
        #check_checkpoint_compatibility(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        with init_empty_weights():
            self.model=AutoModel.from_config(config)
        if checkpoint is not None:
            if checkpoint.endswith(".gguf"):
                sd=load_gguf_checkpoint(checkpoint)      
                match_state_dict(self.model, sd,show_num=10)
                set_gguf2meta_model(self.model,sd,dtype,torch.device("cpu"),) 
            else:
                self.model = self.model.to_empty(device=torch.device("cpu"))
                sd=_load_file(checkpoint)
                self.model.load_state_dict(sd, strict=False, assign=True)
                self.model = self.model.to(device=torch.device("cpu"),dtype=dtype)
            del sd
            gc.collect()
        else:
            self.model = AutoModel.from_pretrained(model_path, config=config, torch_dtype=dtype).to(device).eval()

    def _model_ctx(
        self,
        streaming_prefetch_count: int | None,
    ) -> AbstractContextManager:
        if streaming_prefetch_count is not None:
            return _streaming_model(
                self.model,
                layers_attr="language_model.model.layers",
                target_device=self.device,
                prefetch_count=streaming_prefetch_count,
            )

        return self.model
    @property
    def last_think_text(self) -> str:
        """Raw decoder output inside ``<think>...</think>`` (T2I think mode only)."""
        return self._last_think_text

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        image_size: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        cfg_scale: float = 4.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        batch_size: int = 1,
        seed: int = 0,
        think_mode: bool = False,
        streaming_prefetch_count=2
    ) -> list[Image.Image]:

        if  streaming_prefetch_count is not None: 
            with self._model_ctx(streaming_prefetch_count) as self.model:
                out = self.model.t2i_generate(
                    self.tokenizer,
                    prompt,
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    seed=seed,
                    think_mode=think_mode,
                )
        else:
            out = self.model.t2i_generate(
                self.tokenizer,
                prompt,
                image_size=image_size,
                cfg_scale=cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
                think_mode=think_mode,
            )
        if think_mode:
            tensor, think_text = out
            self._last_think_text = think_text
        else:
            tensor = out
            self._last_think_text = ""
        return _to_pil(tensor)


def _resolve_size(sample: dict, default_width: int, default_height: int) -> tuple[int, int]:
    """Pick output (W, H) for a sample.

    If the sample JSON provides ``width`` and ``height`` they take precedence.
    Otherwise fall back to the CLI defaults (``--width`` / ``--height``).
    """
    if "width" in sample and "height" in sample:
        return int(sample["width"]), int(sample["height"])
    return default_width, default_height


def _save_images(
    images: Sequence[Image.Image],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(images) == 1:
        images[0].save(out_path)
        print(f"[saved] {out_path}")
        return
    for i, img in enumerate(images):
        p = out_path.with_name(f"{out_path.stem}_{i}{out_path.suffix}")
        img.save(p)
        print(f"[saved] {p}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T2I inference for SenseNova-U1.")
    p.add_argument(
        "--model_path",
        required=True,
        help="HuggingFace Hub id (e.g. sensenova/SenseNova-U1-8B-MoT) or a local path.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Generate from a single prompt.")
    src.add_argument(
        "--jsonl",
        help='JSONL file, one sample per line. Required: {"prompt": ...}. '
        'Optional: {"width": W, "height": H, "seed": S}.',
    )

    p.add_argument("--output", default="output.png", help="Output path when using --prompt.")
    p.add_argument("--output_dir", default="outputs", help="Output directory when using --jsonl.")

    p.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=(
            f"Output image width (default: {DEFAULT_WIDTH}). For --jsonl, this is the "
            "fallback when a sample does not specify its own width/height. "
            f"Trained buckets: {sorted(set(SUPPORTED_RESOLUTIONS.values()))}."
        ),
    )
    p.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Output image height (default: {DEFAULT_HEIGHT}). See --width for supported values.",
    )
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument(
        "--cfg_norm",
        default="none",
        choices=["none", "global", "channel", "cfg_zero_star"],
        help=(
            "Classifier-free guidance rescaling mode. 'none' (default) is classical CFG;"
            "'global'/'channel' rescale the CFG output back to the conditional norm (globally / per-channel);"
            "'cfg_zero_star' is CFG-Zero*-style guidance."
        ),
    )
    p.add_argument("--timestep_shift", type=float, default=3.0)
    p.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("LO", "HI"),
    )
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Random seed for reproducible sampling (default: {DEFAULT_SEED}). "
            "In --jsonl mode, a per-sample `seed` field in the JSONL overrides this."
        ),
    )

    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--attn_backend",
        default="auto",
        choices=["auto", "flash", "sdpa"],
        help=(
            "Attention kernel used by the Qwen3 layers. "
            "'auto' picks flash-attn when it's importable and falls back to SDPA "
            "otherwise. 'flash' hard-requires flash-attn; 'sdpa' forces torch SDPA "
            "even when flash-attn is installed (useful for A/B-ing outputs)."
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Print timing stats: model load time, average per-image generation "
            f"time, and the same time normalized per image token (patch size = "
            f"{DEFAULT_IMAGE_PATCH_SIZE})."
        ),
    )

    p.add_argument(
        "--enhance",
        action="store_true",
        help=(
            "Run the user prompt through an LLM enhancer before T2I inference. "
            "Helpful for short / loose prompts, especially infographic-style "
            "generation. Configure via U1_ENHANCE_{BACKEND,ENDPOINT,API_KEY,MODEL} "
            "env vars; defaults target Gemini 3.1 Pro. "
            "See docs/prompt_enhancement.md for details."
        ),
    )
    p.add_argument(
        "--print_enhance",
        action="store_true",
        help="With --enhance: also print the enhanced prompt for debugging.",
    )
    p.add_argument(
        "--think",
        action="store_true",
        help=(
            "Enable T2I reasoning (think) mode: the model first generates a "
            "<think>...</think> block, then runs image generation."
        ),
    )
    p.add_argument(
        "--think_output",
        type=str,
        default=None,
        help=(
            "When using --prompt with --think: path to save the reasoning text."
            "Default: ``<output_stem>.think.txt`` next to --output."
        ),
    )
    p.add_argument(
        "--print_think",
        action="store_true",
        help="With --think: also print the reasoning block to stdout.",
    )

    p.add_argument(
        "--prefetch_count",type=int, default=2)
     
    p.add_argument(
        "--checkpoint",
        type=str,
        default="None",
        help=(
            "single checkpoint path"
        ),
    )

    return p.parse_args()


def _build_enhancer(args: argparse.Namespace):
    """Instantiate :class:`PromptEnhancer` + a dedicated event loop iff
    ``--enhance`` was passed.

    We keep a single event loop for the whole run so the underlying
    :class:`httpx.AsyncClient` inside the adapter can actually pool
    connections across samples – spawning a fresh ``asyncio.run`` per
    sample would otherwise tear the pool down every time.

    Returns:
        ``(enhancer, loop)`` or ``(None, None)``.
    """
    if not args.enhance:
        return None, None
    import asyncio

    from dotenv import load_dotenv

    from src.sensenova_u1.prompt_enhance import PromptEnhancer

    load_dotenv()
    enhancer = PromptEnhancer.from_env(style="infographic")
    loop = asyncio.new_event_loop()
    return enhancer, loop


def _maybe_enhance(enhancer, loop, prompt: str, *, verbose: bool) -> str:
    """Send ``prompt`` through the enhancer (if configured) and return the result."""
    if enhancer is None:
        return prompt
    enhanced = loop.run_until_complete(enhancer.aenhance(prompt))
    if verbose:
        print(f"[enhance] original : {prompt}")
        print(f"[enhance] enhanced : {enhanced}")
    return enhanced


def main() -> None:
    args = parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    #sensenova_u1.set_attn_backend(args.attn_backend)
    #print(f"[attn] backend={args.attn_backend!r} (effective={sensenova_u1.effective_attn_backend()!r})")

    profiler = InferenceProfiler(enabled=args.profile, device=args.device)
    enhancer, loop = _build_enhancer(args)

    try:
        with profiler.time_load():
            engine = SenseNovaU1T2I(args.model_path, device=args.device, dtype=dtype,checkpoint=args.checkpoint)

        cfg_interval = tuple(args.cfg_interval)

        if args.prompt is not None:
            prompt = _maybe_enhance(enhancer, loop, args.prompt, verbose=args.print_enhance)
            _warn_if_unsupported(args.width, args.height)
            with profiler.time_generate(args.width, args.height, args.batch_size):
                images = engine.generate(
                    prompt,
                    image_size=(args.width, args.height),
                    cfg_scale=args.cfg_scale,
                    cfg_norm=args.cfg_norm,
                    timestep_shift=args.timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=args.num_steps,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    think_mode=args.think,
                    streaming_prefetch_count=args.prefetch_count,
                )
            _save_images(images, Path(args.output))
            if args.think:
                think_path = (
                    Path(args.think_output) if args.think_output else Path(args.output).with_suffix(".think.txt")
                )
                think_path.parent.mkdir(parents=True, exist_ok=True)
                think_path.write_text(engine.last_think_text, encoding="utf-8")
                print(f"[saved] {think_path}")
                if args.print_think:
                    print("--- think ---")
                    print(engine.last_think_text)
                    print("--- end think ---")
            profiler.report()
            return

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        try:
            from tqdm import tqdm
        except ImportError:

            def tqdm(x, **_kw):  # type: ignore[no-redef]
                return x

        for i, sample in enumerate(tqdm(samples, desc="T2I")):
            w, h = _resolve_size(sample, args.width, args.height)
            _warn_if_unsupported(w, h)
            seed_i = int(sample.get("seed", args.seed))
            think_i = bool(sample["think"]) if "think" in sample else args.think
            prompt = _maybe_enhance(enhancer, loop, sample["prompt"], verbose=args.print_enhance)
            with profiler.time_generate(w, h, 1):
                images = engine.generate(
                    prompt,
                    image_size=(w, h),
                    cfg_scale=args.cfg_scale,
                    cfg_norm=args.cfg_norm,
                    timestep_shift=args.timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=args.num_steps,
                    batch_size=1,
                    seed=seed_i,
                    think_mode=think_i,
                )
            tag = sample.get("type")
            stem = f"{i + 1:04d}" + (f"_{tag}" if tag else "") + f"_{w}x{h}.png"
            images[0].save(out_dir / stem)
            if think_i:
                think_stem = stem.replace(".png", ".think.txt")
                (out_dir / think_stem).write_text(engine.last_think_text, encoding="utf-8")
                if args.print_think:
                    print(f"[think] sample {i + 1} -> {think_stem}")

        profiler.report()
    finally:
        if enhancer is not None:
            try:
                loop.run_until_complete(enhancer.aclose())
            finally:
                loop.close()


if __name__ == "__main__":
    main()
