from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
import gc   
# import sensenova_u1

from accelerate import init_empty_weights
from contextlib import AbstractContextManager
from ..utils import _streaming_model,load_gguf_checkpoint, match_state_dict,set_gguf2meta_model
from ...src.sensenova_u1.models.neo_unify.modeling_qwen3 import set_attn_backend
from safetensors.torch import load_file as st_load_file
from ...src.sensenova_u1.models.neo_unify.utils import load_image_native
from ...src.sensenova_u1.models.neo_unify.utils import smart_resize
from ...src.sensenova_u1.utils import (
    DEFAULT_IMAGE_PATCH_SIZE,
    InferenceProfiler,
    save_compare,
)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

DEFAULT_SEED = 42
DEFAULT_SYSTEM_MESSAGE = """You are a multimodal assistant capable of reasoning with both text and images. You support two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> block and place all reasoning inside it. You MUST interleave text with generated images using tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any images naturally alongside the text.\n\nAfter the think block, always provide a concise, user-facing final answer. The answer may include text, images, or both. Match the user's language in both reasoning and the final answer."""

# Output H / W must be divisible by this (= patch_size * merge_size).
_IMAGE_GRID_FACTOR = DEFAULT_IMAGE_PATCH_SIZE

# aspect ratio ispreserved, total pixels are normalized to this target
DEFAULT_TARGET_PIXELS = 2048 * 2048
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

def _to_tensor(x: torch.Tensor) -> torch.Tensor:
   return _denorm(x.float()).permute(0, 2, 3, 1).cpu()

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
        self.dtype = dtype
        #check_checkpoint_compatibility(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.checkpoint = checkpoint
        self.model_path=model_path
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.model = None
    def load_state_dict(self,):
        if self.model is not None:
            return  
        if self.checkpoint is not None:
            with init_empty_weights():
                self.model=AutoModel.from_config(self.config)
            if self.checkpoint.endswith(".gguf"):
                sd=load_gguf_checkpoint(self.checkpoint)      
                #match_state_dict(self.model, sd,show_num=10)
                set_gguf2meta_model(self.model,sd,self.dtype,torch.device("cpu"),) 
            else:
                #self.model = self.model.to_empty(device=torch.device("cpu"))
                sd=st_load_file(self.checkpoint)
                self.model.load_state_dict(sd, strict=False, assign=True)
                self.model = self.model.to(device=torch.device("cpu"),dtype=self.dtype)
                self.model.eval()
            del sd
            gc.collect()
        else:
            self.model = AutoModel.from_pretrained(self.model_path, config=self.config, torch_dtype=self.dtype).to(self.device).eval()


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
        think_mode = False,
        seed: int = 0,
        streaming_prefetch_count=1
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
                    think_mode=think_mode,
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
                think_mode=think_mode,
                seed=seed,
            )
        return output
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
        return _to_tensor(tensor)
    
    @torch.inference_mode()
    def answer(
        self,
        image,
        question: str,
        history: list | None = None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        streaming_prefetch_count: int = 1,
    ) -> tuple[str, list]:
        pixel_values, grid_hw = load_image_native(image)
        pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)
        grid_hw = grid_hw.to(self.device)

        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            generation_config["temperature"] = temperature
            generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
        if repetition_penalty is not None:
            generation_config["repetition_penalty"] = repetition_penalty
        if  streaming_prefetch_count is not None: 
            with self._model_ctx(streaming_prefetch_count) as self.model:
                response, updated_history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=history,
                    return_history=True,
                    grid_hw=grid_hw,
                )
        else:
            response, updated_history = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=history,
                return_history=True,
                grid_hw=grid_hw,
            )
        return response, updated_history


    @torch.inference_mode()
    def interleave_gen(
        self,
        prompt: str,
        input_images: Sequence[Image.Image] = (),
        image_size: tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        think_mode: bool = True,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        seed: int = 0,
        streaming_prefetch_count=1
    ) -> tuple[str, list[Image.Image]]:
        if input_images is None:
            input_images = ()
        if  streaming_prefetch_count is not None:
            with self._model_ctx(streaming_prefetch_count) as self.model:
                text, image_tensors = self.model.interleave_gen(
                    self.tokenizer,
                    prompt,
                    images=list(input_images),
                    image_size=image_size,
                    cfg_scale=cfg_scale,
                    img_cfg_scale=img_cfg_scale,
                    timestep_shift=timestep_shift,
                    cfg_interval=cfg_interval,
                    num_steps=num_steps,
                    system_message=system_message,
                    think_mode=think_mode,
                    seed=seed,
                )
        else:   
            text, image_tensors = self.model.interleave_gen(
                self.tokenizer,
                prompt,
                images=list(input_images),
                image_size=image_size,
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                system_message=system_message,
                think_mode=think_mode,
                seed=seed,
            )
        return text, _to_tensor(image_tensors)


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


def  load_sensenova_model(model_path,device,repo,attn_backend,dtype=torch.bfloat16):
    set_attn_backend(attn_backend)
    engine = SenseNovaU1Editing(repo, device, dtype,model_path)
    engine.load_state_dict()
    return engine

def infer_sensenova_edit(engine,prompt,cfg_scale,cfg_norm,num_steps,batch_size,timestep_shift,img_cfg_scale,cfg_interval,width,height,images,target_pixels,seed,prefetch_count,think_mode=False):
    cfg_interval = tuple(cfg_interval)
    cli_explicit_size: tuple[int, int] | None = (width, height) if width is not None else None

    profiler = InferenceProfiler(enabled=True)
    #images = [_load_input_image(p) for p in args.image]
    #_maybe_warn_low_resolution_inputs(images, args.image, args.target_pixels)
    w, h = _resolve_output_size(
        images,
        explicit=cli_explicit_size,
        target_pixels=target_pixels,
    )
    # _set_seed(args.seed)
    
    with profiler.time_generate(w, h, batch_size):
        output = engine.edit(
            prompt,
            images,
            image_size=(w, h),
            cfg_scale=cfg_scale,
            img_cfg_scale=img_cfg_scale,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
            cfg_interval=cfg_interval,
            num_steps=num_steps,
            batch_size=batch_size,
            think_mode = think_mode,
            seed=seed,
            streaming_prefetch_count=prefetch_count,
        )
        profiler.report()
    if think_mode:
        return _to_tensor(output[0]), output[1]
    return _to_tensor(output), "not think mode"




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


# if __name__ == "__main__":
#     main()


