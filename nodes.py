"""
ComfyUI-DramaBox: Custom nodes for DramaBox expressive TTS with voice cloning.

Source: https://github.com/resemble-ai/DramaBox
Models: https://huggingface.co/ResembleAI/Dramabox

This node clones the DramaBox repository on first use and downloads DramaBox
core weights into ComfyUI/models/DramaBox/.
"""

import gc
import json
import os
import sys
import subprocess
import tempfile
import logging
import urllib.request
import zipfile
from pathlib import Path

import torch
import torchaudio
import folder_paths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NODE_DIR = Path(__file__).parent
DRAMABOX_REPO_DIR = NODE_DIR / "DramaBox"
MODELS_DIR = Path(folder_paths.models_dir) / "DramaBox"

# ---------------------------------------------------------------------------
# Repository bootstrap
# ---------------------------------------------------------------------------
_repo_paths_added = False


def _add_repo_paths():
    """Insert DramaBox source directories into sys.path (idempotent)."""
    global _repo_paths_added
    if _repo_paths_added:
        return
    for subdir in ["ltx2", "src"]:
        p = str(DRAMABOX_REPO_DIR / subdir)
        if p not in sys.path:
            sys.path.insert(0, p)
    _repo_paths_added = True


def _clone_via_git():
    """Clone with git. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "git", "clone", "--depth=1",
                "https://github.com/resemble-ai/DramaBox.git",
                str(DRAMABOX_REPO_DIR),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("[DramaBox] Repository cloned successfully via git.")
            return True
        logger.warning(f"[DramaBox] git clone returned {result.returncode}: {result.stderr}")
    except FileNotFoundError:
        logger.warning("[DramaBox] git not found on PATH, falling back to zipball download.")
    return False


def _clone_via_zipball():
    """Download GitHub zipball as a fallback when git is unavailable."""
    url = "https://github.com/resemble-ai/DramaBox/archive/refs/heads/master.zip"
    tmp_zip = NODE_DIR / "_dramabox_tmp.zip"
    logger.info(f"[DramaBox] Downloading DramaBox zipball from {url} …")
    try:
        urllib.request.urlretrieve(url, str(tmp_zip))
        with zipfile.ZipFile(str(tmp_zip), "r") as zf:
            zf.extractall(str(NODE_DIR))
        extracted = NODE_DIR / "DramaBox-master"
        extracted.rename(DRAMABOX_REPO_DIR)
        logger.info("[DramaBox] Repository extracted successfully.")
        return True
    except Exception as e:
        logger.error(f"[DramaBox] Zipball download failed: {e}")
        return False
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()


def _ensure_repo():
    """Ensure the DramaBox GitHub repository is present and on sys.path."""
    if not DRAMABOX_REPO_DIR.exists():
        logger.info("[DramaBox] DramaBox repository not found – downloading…")
        if not _clone_via_git():
            if not _clone_via_zipball():
                raise RuntimeError(
                    "[DramaBox] Could not download the DramaBox repository.\n"
                    "Please manually clone https://github.com/resemble-ai/DramaBox "
                    f"into {DRAMABOX_REPO_DIR}"
                )
    _add_repo_paths()


# ---------------------------------------------------------------------------
# Model download helpers
# ---------------------------------------------------------------------------

def _download_models(bnb_4bit: bool = True) -> dict:
    """Download all required model weights and return paths dict.

    DramaBox-specific weights (transformer + audio components) go into
    ``ComfyUI/models/DramaBox/``.  The Gemma text encoder is downloaded into
    the global HuggingFace cache (``$HF_HOME`` / ``~/.cache/huggingface/hub``),
    so it is shared across tools and not re-downloaded if already present.

    ``bnb_4bit=True``  → ``unsloth/gemma-3-12b-it-bnb-4bit``  (~8 GB, recommended)
    ``bnb_4bit=False`` → ``google/gemma-3-12b-it``             (~25 GB, full precision)
    """
    _ensure_repo()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dramabox_cache = str(MODELS_DIR)

    from model_downloader import get_model_path   # noqa: PLC0415  (from DramaBox src/)
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    logger.info("[DramaBox] Verifying model weights (will download if missing)…")

    # --- DramaBox-unique weights → ComfyUI/models/DramaBox/ ---
    transformer_path = get_model_path("transformer", dramabox_cache)
    audio_components_path = get_model_path("audio_components", dramabox_cache)
    try:
        get_model_path("silence_latent", dramabox_cache)
    except Exception as exc:
        logger.warning(f"[DramaBox] silence_latent optional download skipped: {exc}")

    # --- Gemma text encoder → global HF cache ---
    hf_token = os.environ.get("HF_TOKEN")
    gemma_repo = "unsloth/gemma-3-12b-it-bnb-4bit" if bnb_4bit else "google/gemma-3-12b-it"
    logger.info(f"[DramaBox] Gemma encoder: checking cache ({gemma_repo})…")
    gemma_path = snapshot_download(repo_id=gemma_repo, token=hf_token)
    logger.info(f"[DramaBox] Gemma ready at: {gemma_path}")

    return {
        "transformer": transformer_path,
        "audio_components": audio_components_path,
        "gemma_root": gemma_path,
    }


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
_DEFAULT_NEG = (
    "worst quality, inconsistent, robotic, distorted, noise, static, muffled, "
    "unclear, unnatural, monotone"
)

# Persistent model cache for keep-warm mode (populated by _generate_offloaded).
_model_cache: dict = {}


def _flush_model_cache() -> None:
    """Free all cached model components and release their VRAM."""
    for key in ("audio_conditioner", "prompt_encoder", "velocity_model", "audio_decoder"):
        m = _model_cache.pop(key, None)
        if m is not None:
            del m
    _model_cache.pop("_key", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[DramaBox] Model cache cleared.")


def _generate_offloaded(
    gemma_root: str,
    bnb_4bit: bool,
    paths: dict,
    prompt: str,
    voice_ref_path,
    cfg_scale: float,
    stg_scale: float,
    duration_multiplier: float,
    seed: int,
    keep_warm: bool = False,
) -> tuple:
    """Memory-efficient generation: each component is loaded, used, then freed
    before the next one loads.  Gemma is loaded with device_map='auto' so
    accelerate distributes its layers across GPU + CPU RAM, preventing the
    ~8 GB weight spike from OOM-ing an 8 GB GPU.

    When ``keep_warm=True`` all four components are cached in ``_model_cache``
    and reused on subsequent calls.  Use this on GPUs with 24+ GB VRAM for
    faster repeated generation; leave off on 8-12 GB cards.

    Stages
    ------
    1. AudioConditioner  — encode voice reference, free (or cache)
    2. PromptEncoder     — encode text (Gemma device_map=auto),
                           move embeddings to CPU RAM, free (or cache)
    3. DramaBox DiT      — load transformer, denoise 30 steps, free (or cache)
    4. AudioDecoder      — decode latent → waveform, free (or cache)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # All DramaBox imports are deferred — the repo must be on sys.path first.
    _ensure_repo()

    from audio_conditioning import AudioConditionByReferenceLatent          # noqa: PLC0415
    from inference_server import auto_rescale_for_cfg, estimate_duration    # noqa: PLC0415
    from ltx_core.batch_split import BatchSplitAdapter                      # noqa: PLC0415
    from ltx_core.components.diffusion_steps import EulerDiffusionStep      # noqa: PLC0415
    from ltx_core.components.guiders import (                               # noqa: PLC0415
        MultiModalGuider, MultiModalGuiderParams,
    )
    from ltx_core.components.noisers import GaussianNoiser                  # noqa: PLC0415
    from ltx_core.components.patchifiers import AudioPatchifier             # noqa: PLC0415
    from ltx_core.components.schedulers import LTX2Scheduler               # noqa: PLC0415
    from ltx_core.loader import SDOps                                       # noqa: PLC0415
    from ltx_core.loader.registry import DummyRegistry                     # noqa: PLC0415
    from ltx_core.loader.single_gpu_model_builder import (                  # noqa: PLC0415
        SingleGPUModelBuilder as Builder,
    )
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio  # noqa: PLC0415
    from ltx_core.model.model_protocol import ModelConfigurator            # noqa: PLC0415
    from ltx_core.model.transformer.attention import AttentionFunction      # noqa: PLC0415
    from ltx_core.model.transformer.model import (                         # noqa: PLC0415
        LTXModel, LTXModelType, X0Model,
    )
    from ltx_core.model.transformer.rope import LTXRopeType               # noqa: PLC0415
    from ltx_core.tools import AudioLatentTools                            # noqa: PLC0415
    from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape    # noqa: PLC0415
    from ltx_pipelines.utils.blocks import (                               # noqa: PLC0415
        AudioConditioner, AudioDecoder, PromptEncoder,
    )
    from ltx_pipelines.utils.denoisers import (                            # noqa: PLC0415
        GuidedDenoiser, SimpleDenoiser,
    )
    from ltx_pipelines.utils.media_io import decode_audio_from_file        # noqa: PLC0415
    from ltx_pipelines.utils.samplers import euler_denoising_loop          # noqa: PLC0415
    from safetensors import safe_open                                       # noqa: PLC0415

    patchifier = AudioPatchifier(patch_size=1)

    # ------------------------------------------------------------------
    # Cache key — flush stale cache if config changed or warm mode is off.
    # ------------------------------------------------------------------
    _cache_key = (gemma_root, paths["audio_components"], paths["transformer"], bnb_4bit)
    if not keep_warm:
        if _model_cache:
            _flush_model_cache()
    elif _model_cache.get("_key") != _cache_key:
        logger.info("[DramaBox] Model config changed — flushing warm cache.")
        _flush_model_cache()

    # ------------------------------------------------------------------
    # Closure applied only when loading a new PromptEncoder.
    # The original uses device_map=str(device) which forces all ~8 GB of
    # the 4-bit Gemma weights onto the GPU at once — fatal on ≤ 8 GB cards.
    # With device_map="auto" accelerate splits layers across GPU + CPU RAM.
    # ------------------------------------------------------------------
    def _low_vram_load_bnb(self_pe, gemma_root_path: str):
        import json as _j
        import os as _o
        import logging as _l
        from transformers import Gemma3ForConditionalGeneration, BitsAndBytesConfig
        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
        from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
        from ltx_core.utils import find_matching_file

        prequantized = False
        cfg_path = _o.path.join(gemma_root_path, "config.json")
        if _o.path.exists(cfg_path):
            try:
                with open(cfg_path) as _f:
                    prequantized = "quantization_config" in _j.load(_f)
            except Exception:
                pass

        from_kwargs: dict = {"device_map": "auto", "torch_dtype": self_pe._dtype}
        if not prequantized:
            _l.info("[DramaBox] Loading Gemma with device_map=auto + bitsandbytes 4-bit…")
            from_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self_pe._dtype,
            )
        else:
            _l.info("[DramaBox] Loading pre-quantized Gemma with device_map=auto…")

        hf_model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_root_path, **from_kwargs
        )
        tokenizer = LTXVGemmaTokenizer(
            str(find_matching_file(gemma_root_path, "tokenizer.model").parent), 1024
        )
        encoder = GemmaTextEncoder(
            model=hf_model, tokenizer=tokenizer, dtype=self_pe._dtype
        )
        if torch.cuda.is_available():
            _l.info(
                f"[DramaBox] Gemma loaded: "
                f"{torch.cuda.memory_allocated() / 1e9:.1f} GB VRAM in use"
            )
        return encoder

    # ------------------------------------------------------------------
    # Duration + target shape
    # ------------------------------------------------------------------
    gen_dur = estimate_duration(prompt, duration_multiplier)
    fps = 25.0
    n_frames = int(round(gen_dur * fps)) + 1
    n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
    pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
    tgt_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    audio_tools = AudioLatentTools(patchifier=patchifier, target_shape=tgt_shape)
    logger.info(f"[DramaBox] Target: {gen_dur:.1f}s audio, {n_frames} latent frames.")

    state = audio_tools.create_initial_state(device, dtype)
    gen = torch.Generator(device=device).manual_seed(seed)
    state = GaussianNoiser(generator=gen)(state, noise_scale=1.0)

    # ------------------------------------------------------------------
    # Stage 1 / 4 — Voice reference (AudioConditioner: load/cache → encode → free/cache)
    # ------------------------------------------------------------------
    if voice_ref_path:
        logger.info("[DramaBox] Stage 1/4: Encoding voice reference…")
        voice = decode_audio_from_file(voice_ref_path, device, 0.0, 10.0)
        if voice is not None:
            w = voice.waveform
            if w.dim() == 2:
                w = (w.repeat(2, 1) if w.shape[0] == 1 else w).unsqueeze(0)
            elif w.dim() == 3 and w.shape[1] == 1:
                w = w.repeat(1, 2, 1)
            target_samples = int(10.0 * voice.sampling_rate)
            if w.shape[-1] < target_samples:
                w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
            w = w[..., :target_samples]
            peak = w.abs().max()
            if peak > 0:
                w = w * (10 ** (-4.0 / 20) / peak)
            voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)

            if keep_warm and "audio_conditioner" in _model_cache:
                logger.info("[DramaBox] Stage 1/4: Using cached AudioConditioner.")
                ac = _model_cache["audio_conditioner"]
            else:
                ac = AudioConditioner(
                    checkpoint_path=paths["audio_components"], dtype=dtype, device=device
                )
                if keep_warm:
                    _model_cache["audio_conditioner"] = ac
                    _model_cache["_key"] = _cache_key

            ref_latent = ac(lambda enc: vae_encode_audio(voice, enc, None))
            if not keep_warm:
                del ac
                gc.collect()
                torch.cuda.empty_cache()

            state = AudioConditionByReferenceLatent(
                latent=ref_latent.to(device, dtype), strength=1.0
            ).apply_to(state, audio_tools)
            logger.info("[DramaBox] Voice reference encoded.")

    # ------------------------------------------------------------------
    # Stage 2 / 4 — Text encoding (PromptEncoder / Gemma: load/cache → encode → free/cache)
    # Embeddings are moved to CPU RAM so VRAM is available for the transformer.
    # ------------------------------------------------------------------
    logger.info(
        "[DramaBox] Stage 2/4: Encoding text with Gemma "
        "(device_map=auto distributes layers across GPU + CPU RAM)…"
    )
    prompts_to_encode = [prompt, _DEFAULT_NEG] if cfg_scale > 1.0 else [prompt]

    if keep_warm and "prompt_encoder" in _model_cache:
        logger.info("[DramaBox] Stage 2/4: Using cached PromptEncoder.")
        pe = _model_cache["prompt_encoder"]
        ctx = pe(prompts_to_encode)
    else:
        _orig_load_bnb = PromptEncoder._load_bnb_4bit_encoder
        PromptEncoder._load_bnb_4bit_encoder = _low_vram_load_bnb
        try:
            pe = PromptEncoder(
                checkpoint_path=paths["audio_components"],
                gemma_root=gemma_root,
                dtype=dtype,
                device=device,
                use_bnb_4bit=bnb_4bit,
                warm=True,       # triggers our patched _load_bnb_4bit_encoder
                audio_only=True,
            )
            ctx = pe(prompts_to_encode)
        finally:
            PromptEncoder._load_bnb_4bit_encoder = _orig_load_bnb
        if keep_warm:
            _model_cache["prompt_encoder"] = pe
            _model_cache["_key"] = _cache_key

    # Pin embeddings to CPU so Gemma's VRAM is freed before the transformer loads.
    a_ctx = ctx[0].audio_encoding.cpu()
    a_ctx_neg = ctx[1].audio_encoding.cpu() if cfg_scale > 1.0 else None
    del ctx
    if not keep_warm:
        del pe
        gc.collect()
        torch.cuda.empty_cache()
    logger.info("[DramaBox] Text encoded.")

    # ------------------------------------------------------------------
    # Stage 3 / 4 — Denoising (DramaBox DiT: load/cache → denoise → free/cache)
    # ------------------------------------------------------------------
    logger.info("[DramaBox] Stage 3/4: Loading transformer and running 30-step denoising…")

    with safe_open(paths["transformer"], framework="pt") as _sf:
        _config = json.loads(_sf.metadata()["config"])

    class _AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
        @classmethod
        def from_config(cls, cfg):
            t = cfg.get("transformer", {})
            cp = None
            if not t.get("caption_proj_before_connector", False):
                from ltx_core.model.transformer.text_projection import (  # noqa: PLC0415
                    create_caption_projection,
                )
                with torch.device("meta"):
                    cp = create_caption_projection(t, audio=True)
            return LTXModel(
                model_type=LTXModelType.AudioOnly,
                audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                audio_in_channels=t.get("audio_in_channels", 128),
                audio_out_channels=t.get("audio_out_channels", 128),
                num_layers=t.get("num_layers", 48),
                audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                norm_eps=t.get("norm_eps", 1e-6),
                attention_type=AttentionFunction(t.get("attention_type", "default")),
                positional_embedding_theta=10000.0,
                audio_positional_embedding_max_pos=[20.0],
                timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                double_precision_rope=(
                    t.get("frequencies_precision", False) == "float64"
                ),
                apply_gated_attention=t.get("apply_gated_attention", False),
                audio_caption_projection=cp,
                cross_attention_adaln=t.get("cross_attention_adaln", False),
            )

    if keep_warm and "velocity_model" in _model_cache:
        logger.info("[DramaBox] Stage 3/4: Using cached transformer.")
        velocity_model = _model_cache["velocity_model"]
    else:
        audio_sd_ops = (
            SDOps("AO")
            .with_matching(prefix="model.diffusion_model.")
            .with_replacement("model.diffusion_model.", "")
        )
        velocity_model = (
            Builder(
                model_path=paths["transformer"],
                model_class_configurator=_AudioOnlyConfigurator,
                model_sd_ops=audio_sd_ops,
                registry=DummyRegistry(),
            )
            .build(device=device, dtype=dtype)
            .to(device)
            .eval()
        )
        if keep_warm:
            _model_cache["velocity_model"] = velocity_model
            _model_cache["_key"] = _cache_key

    # Move embeddings back to GPU for the denoising loop.
    a_ctx = a_ctx.to(device)
    if a_ctx_neg is not None:
        a_ctx_neg = a_ctx_neg.to(device)

    sigmas = LTX2Scheduler().execute(steps=30, latent=state.latent).to(device)
    resc = auto_rescale_for_cfg(cfg_scale)
    if cfg_scale > 1.0:
        denoiser: GuidedDenoiser | SimpleDenoiser = GuidedDenoiser(
            v_context=None,
            a_context=a_ctx,
            video_guider=None,
            audio_guider=MultiModalGuider(
                params=MultiModalGuiderParams(
                    cfg_scale=cfg_scale,
                    stg_scale=stg_scale,
                    stg_blocks=[29],
                    rescale_scale=resc,
                    modality_scale=1.0,
                ),
                negative_context=a_ctx_neg,
            ),
        )
    else:
        denoiser = SimpleDenoiser(v_context=None, a_context=a_ctx)

    x0 = X0Model(velocity_model)
    _, audio_state = euler_denoising_loop(
        sigmas=sigmas,
        video_state=None,
        audio_state=state,
        stepper=EulerDiffusionStep(),
        transformer=BatchSplitAdapter(x0, max_batch_size=1),
        denoiser=denoiser,
    )
    # x0 and denoiser are always per-run; only velocity_model is reusable.
    del x0, denoiser, a_ctx, a_ctx_neg
    if not keep_warm:
        del velocity_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("[DramaBox] Denoising complete.")

    # ------------------------------------------------------------------
    # Post-process latent
    # ------------------------------------------------------------------
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    # Silence-prior fix: LTX-2.3 has a clip-end silence spike at frame 513.
    latent = audio_state.latent
    if latent.shape[2] > 513:
        patched = latent.clone()
        for fi in (512, 513):
            t_lerp = (fi - 511) / 3
            patched[:, :, fi, :] = (
                (1.0 - t_lerp) * latent[:, :, 511, :]
                + t_lerp * latent[:, :, 514, :]
            )
        latent = patched

    # ------------------------------------------------------------------
    # Stage 4 / 4 — Audio decoding (AudioDecoder: load/cache → decode → free/cache)
    # ------------------------------------------------------------------
    logger.info("[DramaBox] Stage 4/4: Decoding latent to waveform…")
    if keep_warm and "audio_decoder" in _model_cache:
        logger.info("[DramaBox] Stage 4/4: Using cached AudioDecoder.")
        ad = _model_cache["audio_decoder"]
    else:
        ad = AudioDecoder(
            checkpoint_path=paths["audio_components"], dtype=dtype, device=device
        )
        if keep_warm:
            _model_cache["audio_decoder"] = ad
            _model_cache["_key"] = _cache_key

    decoded = ad(latent)
    if not keep_warm:
        del ad
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("[DramaBox] Generation complete.")
    return decoded.waveform, decoded.sampling_rate


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class DramaBoxTTS:
    """
    DramaBox expressive TTS with voice cloning.

    Generates rich, dramatic speech from a structured scene prompt.
    An optional voice reference (10+ seconds) clones the speaker timbre.

    Prompt format:
        <speaker description>, "<dialogue>" <action direction> "<more dialogue>"

    Example:
        A woman speaks warmly, "Hello, how are you today?" She laughs,
        "Hahaha, it is so good to see you!"

    Tips:
    - Phonetic sounds go INSIDE quotes: "Hahaha", "Hmm", "Ugh", "Argh"
    - Stage directions go OUTSIDE: She sighs deeply.  He clears his throat.
    - Match the gender/age in your description to the voice reference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            'A woman speaks warmly, "Hello, how are you today?" '
                            'She laughs, "Hahaha, it is so good to see you!"'
                        ),
                        "tooltip": (
                            "Scene prompt. Put dialogue in double quotes, stage "
                            "directions outside them. Phonetic sounds (Hahaha, Hmm) "
                            "go inside quotes; named actions (She sighs.) go outside."
                        ),
                    },
                ),
                "quantization": (
                    ["4bit", "none"],
                    {
                        "default": "4bit",
                        "tooltip": (
                            "4bit: auto-downloads unsloth/gemma-3-12b-it-bnb-4bit (~8 GB, recommended). "
                            "none: auto-downloads google/gemma-3-12b-it (~25 GB, full precision)."
                        ),
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 2.5,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": (
                            "CFG guidance scale. Lower = more natural delivery; "
                            "higher = more text-faithful. DramaBox default: 2.5."
                        ),
                    },
                ),
                "stg_scale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.5,
                        "tooltip": (
                            "Skip-token guidance scale. DramaBox default: 1.5."
                        ),
                    },
                ),
            },
            "optional": {
                "voice_sample": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional voice reference for timbre cloning. "
                            "10+ seconds of clean speech recommended."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed for reproducible generations.",
                    },
                ),
                "duration_multiplier": (
                    "FLOAT",
                    {
                        "default": 1.1,
                        "min": 0.5,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Multiply the auto-estimated speech duration. "
                            "1.1 adds 10 %% breathing room. Increase for slower delivery."
                        ),
                    },
                ),
                "keep_models_warm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Keep all model components (Gemma, DiT transformer, audio codecs) "
                            "loaded in memory between runs. "
                            "Enables instant repeated generation but requires ~18+ GB VRAM. "
                            "Leave off for 8-12 GB cards."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/DramaBox"
    DESCRIPTION = (
        "DramaBox expressive TTS with voice cloning. "
        "Each model component (Gemma, DiT, AudioDecoder) is loaded, used, then freed "
        "before the next loads — safe to run on GPUs with as little as 8 GB VRAM "
        "(generation will be slower than on 24 GB but produces correct output). "
        "Gemma is loaded with device_map=auto so its layers are spread across GPU + CPU RAM."
    )

    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        quantization: str,
        cfg_scale: float,
        stg_scale: float,
        voice_sample=None,
        seed: int = 42,
        duration_multiplier: float = 1.1,
        keep_models_warm: bool = False,
    ):
        if not text or not text.strip():
            raise ValueError("[DramaBox] Text prompt cannot be empty.")

        if quantization not in {"4bit", "none"}:
            raise ValueError(f"[DramaBox] Unsupported quantization mode: {quantization}")

        paths = _download_models(bnb_4bit=(quantization == "4bit"))
        gemma_root = paths["gemma_root"]

        # ---- Write voice reference to a temp WAV if an AUDIO input was given ----
        tmp_wav = None
        voice_ref_path = None
        try:
            if voice_sample is not None:
                waveform = voice_sample["waveform"]
                sr = int(voice_sample["sample_rate"])

                # Normalise to [C, S]
                if waveform.dim() == 3:
                    waveform = waveform[0]        # [B, C, S] -> [C, S]
                elif waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # [S] -> [1, S]

                tmp_wav = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="dramabox_ref_"
                )
                tmp_wav.close()
                voice_ref_path = tmp_wav.name
                torchaudio.save(voice_ref_path, waveform.float().cpu(), sr)
                logger.info(
                    f"[DramaBox] Voice reference saved to temp file: {voice_ref_path} "
                    f"({waveform.shape[-1] / sr:.1f}s)"
                )

            # ---- Run staged offloaded inference ----
            waveform_out, sr_out = _generate_offloaded(
                gemma_root=gemma_root,
                bnb_4bit=(quantization == "4bit"),
                paths=paths,
                prompt=text,
                voice_ref_path=voice_ref_path,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                duration_multiplier=duration_multiplier,
                seed=seed,
                keep_warm=keep_models_warm,
            )

        finally:
            # Clean up temp file regardless of success/failure
            if tmp_wav is not None:
                try:
                    os.unlink(tmp_wav.name)
                except OSError:
                    pass

        # ---- Format output for ComfyUI: {"waveform": [B, C, S], "sample_rate": int} ----
        wav = waveform_out.float().cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)   # -> [1, 1, S]
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)                # -> [1, C, S]
        # dim == 3 is already [B, C, S]; leave as-is

        duration = wav.shape[-1] / sr_out
        logger.info(f"[DramaBox] Generated {duration:.1f}s of audio at {sr_out} Hz.")

        return ({"waveform": wav, "sample_rate": sr_out},)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "DramaBoxTTS": DramaBoxTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxTTS": "DramaBox TTS",
}
