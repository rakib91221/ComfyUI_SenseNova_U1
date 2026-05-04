from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence
from contextlib import nullcontext
from accelerate import init_empty_weights
from diffusers.utils import is_accelerate_available
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer
import gc   
# import sensenova_u1
#from ...src.sensenova_u1 import check_checkpoint_compatibility
from src.sensenova_u1.models.neo_unify.utils import smart_resize
from src.sensenova_u1.utils import (
    DEFAULT_IMAGE_PATCH_SIZE,
    InferenceProfiler,
    save_compare,
)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SEED = 42

# Output H / W must be divisible by this (= patch_size * merge_size).
_IMAGE_GRID_FACTOR = DEFAULT_IMAGE_PATCH_SIZE

# aspect ratio ispreserved, total pixels are normalized to this target
DEFAULT_TARGET_PIXELS = 2048 * 2048
from contextlib import AbstractContextManager, contextmanager
from layer_streaming import LayerStreamingWrapper
from collections.abc import Iterator
from safetensors.torch import load_file as _load_file
from typing import Callable, TypeVar
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

    matching_keys = meta_model_keys.intersection(state_dict_keys)
    print(f"Matching keys count: {len(matching_keys)}")
    

    extra_keys = state_dict_keys - meta_model_keys
    if extra_keys:
        print(f"Extra keys in state_dict (not in meta_model): {len(extra_keys)}")
        for key in list(extra_keys)[:show_num]:  # 只显示前10个
            print(f"  - {key}")
    

    missing_keys = meta_model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state_dict (not in state_dict): {len(missing_keys)}")
        for key in list(missing_keys)[:show_num]:  
            print(f"  - {key}")
    
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> list[Image.Image]:
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def _load_input_image(path: str | Path) -> Image.Image:
    """Load as RGB; flatten RGBA onto white so the generator sees a clean canvas."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def _coerce_image_paths(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _maybe_warn_low_resolution_inputs(
    images: Sequence[Image.Image],
    paths: Sequence[str | Path],
    target_pixels: int,
) -> None:
    """Warn when an input image has fewer total pixels than ``target_pixels``.

    The generator runs at ``target_pixels`` (≈ 2048*2048 by default) regardless
    of input size, so feeding a small image forces implicit up-scaling inside
    the pipeline and usually hurts quality. Pre-resizing the input manually
    while preserving aspect ratio gives noticeably better edits.
    """
    low_res = []
    for path, img in zip(paths, images):
        w, h = img.size
        if w * h < target_pixels:
            low_res.append((path, w, h, w * h))
    if not low_res:
        return

    print(
        f"[editing][warn] {len(low_res)} input image(s) have fewer pixels than "
        f"the target ({target_pixels} ≈ 2048*2048):"
    )
    for path, w, h, px in low_res:
        print(f"  - {path}: {w}x{h} = {px} px")
    print(
        "[editing][warn] For best results, manually pre-resize each input so "
        "that width*height ≈ 2048*2048 (aspect ratio preserved) before running "
        "inference. See examples/editing/resize_inputs.py for a reference script."
    )


def _check_grid_divisible(width: int, height: int) -> None:
    if width % _IMAGE_GRID_FACTOR or height % _IMAGE_GRID_FACTOR:
        raise SystemExit(
            f"[editing] output resolution ({width}x{height}) must be a multiple "
            f"of {_IMAGE_GRID_FACTOR} on both axes (image-token grid factor)."
        )


def _resolve_output_size(
    input_images: Sequence[Image.Image],
    *,
    explicit: tuple[int, int] | None,
    target_pixels: int,
) -> tuple[int, int]:
    """Explicit (W, H) wins; else match the first input's aspect ratio and
    normalize the total pixel count to ``target_pixels``."""
    if explicit is not None:
        width, height = explicit
        _check_grid_divisible(width, height)
        return width, height

    w, h = input_images[0].size
    resized_h, resized_w = smart_resize(
        height=h,
        width=w,
        factor=_IMAGE_GRID_FACTOR,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    return resized_w, resized_h


def _explicit_size_from_sample(sample: dict) -> tuple[int, int] | None:
    if "width" in sample and "height" in sample:
        return int(sample["width"]), int(sample["height"])
    return None



class SenseNovaU1Editing:
    """Thin wrapper calling ``model.it2i_generate`` on top of ``AutoModel``."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        checkpoint: str | None = None,
    ) -> None:
        self.device = device
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
    @torch.inference_mode()
    def edit(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        image_size: tuple[int, int],
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        cfg_norm: str = "none",
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        batch_size: int = 1,
        seed: int = 0,
        streaming_prefetch_count=2
    ) -> list[Image.Image]:
        
        if  streaming_prefetch_count is not None:
            with self._model_ctx(streaming_prefetch_count) as self.model:
                output = self.model.it2i_generate(
                    self.tokenizer,
                    prompt,
                    list(images),
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    img_cfg_scale=img_cfg_scale,
                    cfg_norm=cfg_norm,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    seed=seed,
                )
        else:
             output = self.model.it2i_generate(
                self.tokenizer,
                prompt,
                list(images),
                image_size=image_size,
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                cfg_norm=cfg_norm,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
            )
        return _to_pil(output)


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
    p = argparse.ArgumentParser(description="Image editing (it2i) inference for SenseNova-U1.")
    p.add_argument(
        "--model_path",
        required=True,
        help="HuggingFace Hub id (e.g. sensenova/SenseNova-U1-8B-MoT) or a local path.",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--prompt",
        help="Edit instruction. When the prompt does not include an ``<image>`` "
        "placeholder, the model prepends one per input image automatically. "
        "Requires --image.",
    )
    src.add_argument(
        "--jsonl",
        help='JSONL file, one sample per line. Required: {"prompt": str, '
        '"image": str | list[str]}. Optional: {"width": int, "height": int, '
        '"seed": int, "type": str}. When "width" and "height" are both '
        "present they override --width / --height for that sample.",
    )

    p.add_argument(
        "--image",
        nargs="+",
        metavar="PATH",
        help="One or more input image paths (only used with --prompt).",
    )

    p.add_argument("--output", default="output.png", help="Output path when using --prompt.")
    p.add_argument("--output_dir", default="outputs", help="Output directory when using --jsonl.")

    p.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Explicit output width in pixels. Must be given together with --height, "
            f"and must be a multiple of {_IMAGE_GRID_FACTOR}. "
            "When both --width and --height are omitted the output resolution is "
            "derived from the first input image: aspect ratio preserved, total "
            "pixels normalized to --target_pixels."
        ),
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help=f"Explicit output height in pixels. See --width. Must be a multiple of {_IMAGE_GRID_FACTOR}.",
    )
    p.add_argument(
        "--target_pixels",
        type=int,
        default=DEFAULT_TARGET_PIXELS,
        help=(
            f"Target pixel count for the auto-derived output resolution "
            f"(default: {DEFAULT_TARGET_PIXELS} = 2048*2048). The first input "
            "image's aspect ratio is preserved and H*W is rescaled to match "
            f"this target, which is a multiple of {_IMAGE_GRID_FACTOR}. "
            "Ignored when --width / --height are given."
        ),
    )

    p.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="Text CFG weight. Higher values track the edit instruction more aggressively.",
    )
    p.add_argument(
        "--img_cfg_scale",
        type=float,
        default=1.0,
        help=("Image CFG weight (default: 1.0 = image CFG disabled)."),
    )
    p.add_argument(
        "--cfg_norm",
        default="none",
        choices=["none", "global", "channel"],
        help=(
            "Classifier-free guidance rescaling mode. 'none' (default) is classical CFG; "
            "'global'/'channel' rescale the CFG output back to the conditional norm "
            "(globally / per-channel). Unlike t2i, 'cfg_zero_star' is not supported here."
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
        "--compare",
        action="store_true",
        help=(
            "Also save a side-by-side ``[inputs... | output]`` montage with the "
            "prompt rendered below, written next to the plain output as "
            "``<stem>_compare.png``. Useful for eyeballing edits without an "
            "external image viewer."
        ),

    )

    p.add_argument(
        "--prefetch_count",type=int, default=2)    
    
    p.add_argument(
        "--checkpoint",
        type=str,
        default="None",
        help=(
            "single checkpoint path."
        ),
    )


    args = p.parse_args()
    if args.prompt is not None and not args.image:
        p.error("--prompt requires at least one --image.")
    if args.jsonl is not None and args.image:
        p.error("--image is only valid with --prompt; in --jsonl mode, put 'image' in the JSONL.")
    if (args.width is None) != (args.height is None):
        p.error("--width and --height must be given together (or both omitted).")
    if args.width is not None:
        if args.width % _IMAGE_GRID_FACTOR or args.height % _IMAGE_GRID_FACTOR:
            p.error(
                f"--width / --height must each be a multiple of {_IMAGE_GRID_FACTOR} (got {args.width}x{args.height})."
            )
    return args


def main() -> None:
    args = parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # sensenova_u1.set_attn_backend(args.attn_backend)
    # print(f"[attn] backend={args.attn_backend!r} (effective={sensenova_u1.effective_attn_backend()!r})")

    profiler = InferenceProfiler(enabled=args.profile, device=args.device)

    with profiler.time_load():
        engine = SenseNovaU1Editing(args.model_path, device=args.device, dtype=dtype,checkpoint=args.checkpoint)

    cfg_interval = tuple(args.cfg_interval)
    cli_explicit_size: tuple[int, int] | None = (args.width, args.height) if args.width is not None else None

    if args.prompt is not None:
        images = [_load_input_image(p) for p in args.image]
        _maybe_warn_low_resolution_inputs(images, args.image, args.target_pixels)
        w, h = _resolve_output_size(
            images,
            explicit=cli_explicit_size,
            target_pixels=args.target_pixels,
        )
        # _set_seed(args.seed)
        with profiler.time_generate(w, h, args.batch_size):
            outputs = engine.edit(
                args.prompt,
                images,
                image_size=(w, h),
                cfg_scale=args.cfg_scale,
                img_cfg_scale=args.img_cfg_scale,
                cfg_norm=args.cfg_norm,
                timestep_shift=args.timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=args.num_steps,
                batch_size=args.batch_size,
                seed=args.seed,
                streaming_prefetch_count=args.prefetch_count,
            )
        out_path = Path(args.output)
        _save_images(outputs, out_path)
        if args.compare:
            save_compare(out_path, images, outputs[0], args.prompt)
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

    for i, sample in enumerate(tqdm(samples, desc="Editing")):
        paths = _coerce_image_paths(sample["image"])
        images = [_load_input_image(p) for p in paths]
        _maybe_warn_low_resolution_inputs(images, paths, args.target_pixels)
        w, h = _resolve_output_size(
            images,
            explicit=_explicit_size_from_sample(sample) or cli_explicit_size,
            target_pixels=args.target_pixels,
        )
        # _set_seed(int(sample.get("seed", args.seed)))
        with profiler.time_generate(w, h, 1):
            outputs = engine.edit(
                sample["prompt"],
                images,
                image_size=(w, h),
                cfg_scale=args.cfg_scale,
                img_cfg_scale=args.img_cfg_scale,
                cfg_norm=args.cfg_norm,
                timestep_shift=args.timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=args.num_steps,
                batch_size=1,
                seed=args.seed,
                streaming_prefetch_count=args.prefetch_count,
            )
        tag = sample.get("type")
        stem = f"{i + 1:04d}" + (f"_{tag}" if tag else "") + f"_{w}x{h}.png"
        sample_out = out_dir / stem
        outputs[0].save(sample_out)
        if args.compare:
            save_compare(sample_out, images, outputs[0], sample["prompt"])

    profiler.report()


if __name__ == "__main__":
    main()


