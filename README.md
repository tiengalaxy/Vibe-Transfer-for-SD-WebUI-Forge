# Vibe Transfer for SD WebUI Forge

NovelAI-style vibe transfer for Stable Diffusion WebUI Forge.

## Features

- **IP-Adapter Semantic Vibe** — Extract reference image semantics via CLIP Vision, inject through decoupled cross-attention. Closest to NovelAI Vibe Transfer.
- **Reference Attention + AdaIN** — Latent-space attention replacement + adaptive instance normalization.
- **Hybrid Mode** — Both IP-Adapter and Reference simultaneously.
- **Color/Lighting Only** — Pixel-level Reinhard color transfer, does not alter structure.

## Installation

### From URL (recommended)
1. Open Forge's **Extensions** tab → **Install from URL**
2. Paste: `https://github.com/tiengalaxy/Vibe-Transfer-for-SD-WebUI-Forge`
3. Click **Install**, then **Restart** Forge

### Manual
1. Download ZIP
2. Extract to `your-forge-dir/extensions/`
3. Restart Forge

## Usage

1. Open the **Vibe Transfer** accordion in txt2img/img2img
2. Select transfer mode
3. Upload reference image(s) (up to 4 slots)
4. Adjust parameters:
   - **Reference Strength**: Vibe transfer intensity (0.0–2.0)
   - **Information Extracted**: How much info to extract from reference (0.0–1.0)
   - **Style Fidelity**: Style retention for unconditional region (Reference/Hybrid mode)
5. Generate

## Requirements

- **Stable Diffusion WebUI Forge**
- IP-Adapter models (optional, for IP-Adapter mode) — place in `models/ControlNet/` or `models/ipadapter/`
- CLIP Vision model (auto-loaded from Forge supported preprocessors)

## Notes

- IP-Adapter mode requires compatible IP-Adapter models (`.safetensors` recommended)
- Reference mode works without any additional models
- Color mode is pixel-level only — works on any image
- For best results, use reference images with similar composition to your prompt