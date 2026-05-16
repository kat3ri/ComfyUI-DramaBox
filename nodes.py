"""
ComfyUI-DramaBox: Custom nodes for DramaBox expressive TTS with voice cloning.

Source: https://github.com/resemble-ai/DramaBox
Models: https://huggingface.co/ResembleAI/Dramabox

This node clones the DramaBox repository on first use and downloads DramaBox
core weights into ComfyUI/models/DramaBox/.
"""

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

def _download_models():
    """Download all required model weights and return paths dict.

    All models are stored under ``ComfyUI/models/DramaBox/``:
      - dramabox-dit-v1.safetensors (transformer)
      - dramabox-audio-components.safetensors (audio components)
    """
    _ensure_repo()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    logger.info("[DramaBox] Verifying model weights (will download if missing)…")
    hf_token = os.environ.get("HF_TOKEN")

    # --- DramaBox-unique weights → ComfyUI/models/DramaBox/ ---
    dramabox_repo = "ResembleAI/Dramabox"
    model_files = {
        "transformer": "dramabox-dit-v1.safetensors",
        "audio_components": "dramabox-audio-components.safetensors",
    }

    paths = {}
    for name, filename in model_files.items():
        local_path = MODELS_DIR / filename
        if local_path.exists():
            logger.info(f"[DramaBox] {filename} found locally.")
        else:
            logger.info(f"[DramaBox] Downloading {filename} from {dramabox_repo}…")
            hf_hub_download(
                repo_id=dramabox_repo,
                filename=filename,
                local_dir=str(MODELS_DIR),
                token=hf_token,
            )
            logger.info(f"[DramaBox] {filename} downloaded.")
        paths[name] = str(local_path)

    return paths


# ---------------------------------------------------------------------------
# TTSServer singleton (lazy, loaded on first generate() call)
# ---------------------------------------------------------------------------
# Folder types searched for text encoder files, in priority order.
_ENCODER_EMPTY_OPTION = "<no text encoder found in models/text_encoders or models/checkpoints>"
_ENCODER_FOLDER_TYPES = ("text_encoders", "checkpoints")



def _list_text_encoder_checkpoints():
    """Return available text encoder filenames from text_encoders and checkpoints folders."""
    names: list[str] = []
    seen: set[str] = set()
    for folder_type in _ENCODER_FOLDER_TYPES:
        try:
            for name in sorted(folder_paths.get_filename_list(folder_type)):
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        except Exception as e:
            logger.warning(f"[DramaBox] Could not list {folder_type}: {e}")
    return names or [_ENCODER_EMPTY_OPTION]


def _resolve_checkpoint_path(checkpoint_name: str) -> Path:
    """Resolve a text encoder filename to an absolute path, searching text_encoders then checkpoints."""
    if not checkpoint_name or checkpoint_name == _ENCODER_EMPTY_OPTION:        raise ValueError(
            "[DramaBox] No text encoder selected. Place the Gemma encoder directory "
            "(tokenizer.model + preprocessor_config.json + model*.safetensors) under "
            "ComfyUI/models/text_encoders/ and select one of its weight files in the node."
        )

    for folder_type in _ENCODER_FOLDER_TYPES:
        resolver = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(resolver):
            try:
                return Path(resolver(folder_type, checkpoint_name))
            except Exception:
                pass

        resolver = getattr(folder_paths, "get_full_path", None)
        if callable(resolver):
            path = resolver(folder_type, checkpoint_name)
            if path:
                return Path(path)

    raise ValueError(
        f"[DramaBox] Could not resolve text encoder '{checkpoint_name}' from "
        "ComfyUI text_encoders or checkpoints folders."
    )


def _resolve_and_validate_gemma_root(checkpoint_name: str) -> str:
    """Resolve the selected .safetensors file to its parent directory and validate it
    as a usable Gemma encoder root for DramaBox.
    DramaBox's PromptEncoder uses ``find_matching_file`` (rglob) to locate:
      - ``tokenizer.model``
      - ``preprocessor_config.json``
      - ``model*.safetensors``
    All three must be present (directly or in a subdirectory) of the resolved root.
    """
    checkpoint_path = _resolve_checkpoint_path(checkpoint_name)
    gemma_root = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent

    if not gemma_root.exists() or not gemma_root.is_dir():
        raise ValueError(
            f"[DramaBox] Selected encoder path does not resolve to a directory: {checkpoint_path}"
        )

    # DramaBox uses rglob internally, so search recursively to match its behaviour.
    has_tokenizer = any(gemma_root.rglob("tokenizer.model"))
    has_preprocessor = any(gemma_root.rglob("preprocessor_config.json"))
    has_weights = any(gemma_root.rglob("model*.safetensors")) or any(gemma_root.rglob("model*.bin"))
    has_weights = any(
        any(gemma_root.glob(pattern))
        for pattern in ("*.safetensors", "*.bin", "*.pt")
    )

    if not (has_tokenizer and has_preprocessor and has_weights):
            raise ValueError(
                "[DramaBox] Selected encoder directory is missing required Gemma files.\n"
                f"Root searched: {gemma_root}\n"
                "Expected (anywhere inside that directory):\n"
                "  • tokenizer.model\n"
                "  • preprocessor_config.json\n"
                "  • model*.safetensors  (or model*.bin)\n"
            "Download with: snapshot_download('unsloth/gemma-3-12b-it-bnb-4bit') "
            "into ComfyUI/models/text_encoders/ and select any of its .safetensors shards."
        )

    return str(gemma_root)


def _get_server(gemma_root: str, bnb_4bit: bool):
    """Return a cached TTSServer keyed by selected text encoder + quantization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (gemma_root, bool(bnb_4bit), device)
    if cache_key in _tts_servers:
        return _tts_servers[cache_key]

    _ensure_repo()
    paths = _download_models()

    from inference_server import TTSServer  # noqa: PLC0415

    logger.info(
        f"[DramaBox] Loading TTSServer on {device} with encoder '{gemma_root}' "
        f"(bnb_4bit={bool(bnb_4bit)}) (first run may take several minutes)…"
    )

    server = TTSServer(
        checkpoint=paths["transformer"],
        full_checkpoint=paths["audio_components"],
        gemma_root=gemma_root,
        device=device,
        dtype="bf16",
        compile_model=False,   # torch.compile can be unstable in some setups
        bnb_4bit=bool(bnb_4bit),
    )
    logger.info("[DramaBox] TTSServer ready.")
    _tts_servers[cache_key] = server
    return server


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
                "text_encoder_checkpoint": (
                    _list_text_encoder_checkpoints(),
                    {
                        "tooltip": (
                            "Gemma text encoder weight file from ComfyUI models/text_encoders/ "
                            "(or models/checkpoints/). Select any .safetensors shard from the "
                            "encoder directory (e.g. unsloth/gemma-3-12b-it-bnb-4bit). "
                            "The directory must also contain tokenizer.model and "
                            "preprocessor_config.json."
                        ),
                    },
                ),
                "quantization": (
                    ["4bit", "none"],
                    {
                        "default": "4bit",
                        "tooltip": (
                            "Text-encoder quantization mode. Use 4bit for bnb quantized encoders; "
                            "use none for full-precision encoder weights."
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
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/DramaBox"
    DESCRIPTION = (
        "DramaBox expressive TTS with voice cloning. "
        "Generates dramatic, expressive speech from a structured scene prompt. "
        "Requires ~24 GB VRAM on first run."
    )

    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        text_encoder_checkpoint: str,
        quantization: str,
        cfg_scale: float,
        stg_scale: float,
        voice_sample=None,
        seed: int = 42,
        duration_multiplier: float = 1.1,
    ):
        if not text or not text.strip():
            raise ValueError("[DramaBox] Text prompt cannot be empty.")

        if quantization not in {"4bit", "none"}:
            raise ValueError(f"[DramaBox] Unsupported quantization mode: {quantization}")

        gemma_root = _resolve_and_validate_gemma_root(text_encoder_checkpoint)
        server = _get_server(gemma_root=gemma_root, bnb_4bit=(quantization == "4bit"))

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

            # ---- Run inference ----
            waveform_out, sr_out = server.generate(
                prompt=text,
                voice_ref=voice_ref_path,
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                duration_multiplier=duration_multiplier,
                seed=seed,
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
