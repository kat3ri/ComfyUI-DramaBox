# ComfyUI-DramaBox

A ComfyUI custom node that wraps [ResembleAI's DramaBox](https://github.com/resemble-ai/DramaBox) — a 3.3B expressive TTS model with voice cloning built on LTX-2.3.

---

## Features

- **Expressive TTS** — dramatic speech with laughs, sighs, pauses, and transitions driven entirely by the prompt
- **Voice cloning** — upload a 10-second reference clip to match any speaker's timbre
- **ComfyUI-native audio** — output plugs directly into **Preview Audio**, **Save Audio**, or any other audio node

---

## Requirements

| | |
|---|---|
| **GPU** | NVIDIA GPU with ~24 GB VRAM |
| **CUDA** | CUDA 12+ recommended |
| **Disk** | ~17 GB for model weights |

---

## Installation

### Via ComfyUI-Manager *(recommended)*

Search for **ComfyUI-DramaBox** in the Manager and click Install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-user/ComfyUI-DramaBox
cd ComfyUI-DramaBox
pip install -r requirements.txt
```

On the **first generation**, the node will automatically:
1. Clone the DramaBox source repo into `custom_nodes/ComfyUI-DramaBox/DramaBox/`
2. Download model weights (~17 GB) into `ComfyUI/models/DramaBox/`

---

## Node: DramaBox TTS

**Category:** `audio/DramaBox`

### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | STRING | — | Scene prompt (see format below) |
| `cfg_scale` | FLOAT | `2.5` | Lower = more natural; higher = more text-faithful |
| `stg_scale` | FLOAT | `1.5` | Skip-token guidance |
| `voice_sample` *(optional)* | AUDIO | — | Voice reference for cloning (10+ s recommended) |
| `seed` *(optional)* | INT | `42` | Seed for reproducibility |
| `duration_multiplier` *(optional)* | FLOAT | `1.1` | Scale the auto-estimated speech duration |

### Output

| Name | Type | Description |
|------|------|-------------|
| `audio` | AUDIO | Generated waveform — pass to **Preview Audio** or **Save Audio** |

---

## Prompt Format

```
<speaker description>, "<dialogue>" <action direction> "<more dialogue>"
```

**Inside quotes** (model produces the sounds):
- Dialogue: `"Hello, how are you?"`
- Phonetic: `"Hahaha"` `"Hehehe"` `"Mmmmm"` `"Ugh"` `"Argh"`

**Outside quotes** (stage directions):
- `She sighs deeply.` · `He gulps nervously.` · `A long pause.`
- `Her voice cracks.` · `He clears his throat.`

**Avoid inside quotes** (model speaks them literally): Ahem, Pfft, Sigh, Gasp, Cough.

### Example

```
A woman speaks warmly, "Hello, how are you today?" She laughs,
"Hahaha, it is so good to see you!"
```

```
A shadowy villain speaks with cold menace, "You have entered my domain, mortal."
He chuckles darkly, "Such arrogance will be your undoing."
His voice rises with fury, "Kneel, or be destroyed where you stand!"
```

---

## Model Weights

Downloaded automatically from HuggingFace on first run:

| File | Size | Purpose |
|------|------|---------|
| `dramabox-dit-v1.safetensors` | 6.6 GB | DiT transformer |
| `dramabox-audio-components.safetensors` | 1.9 GB | Audio VAE + vocoder + embeddings |
| `unsloth/gemma-3-12b-it-bnb-4bit` | ~8 GB | Text encoder (4-bit quantised) |

Weights are cached in `ComfyUI/models/DramaBox/` and reused on subsequent runs.

---

## Notes

- The **TTSServer loads all models once** and keeps them warm. The first generation takes several minutes; subsequent generations are ~2–3 seconds on an H100.
- Output audio is automatically watermarked with [Resemble Perth](https://github.com/resemble-ai/Perth) (imperceptible neural watermark).
- DramaBox is licensed under the [LTX-2 Community License](https://github.com/resemble-ai/DramaBox/blob/master/LICENSE).
