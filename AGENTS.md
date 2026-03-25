## Cursor Cloud specific instructions

### Project overview
BAGEL is a multimodal AI foundation model (7B active / 14B total params) by ByteDance for text-to-image generation, image understanding, and image editing. The main UI is a Gradio web app (`app.py`). See `README.md` for full docs.

### Environment
- Python 3.10 virtual environment at `/workspace/.venv` — always activate with `source /workspace/.venv/bin/activate`
- Dependencies: `pip install -r requirements.txt` plus `flash_attn` (installed via pre-built wheel from GitHub, see below)
- `flash_attn` cannot be compiled from source without a full CUDA toolkit. Install via pre-built wheel:
  `pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`

### GPU requirement
- **This project requires an NVIDIA GPU with CUDA** for model inference, training, and evaluation
- Without a GPU, all dependencies install and imports work, but `python app.py` will fail at model weight loading (`load_checkpoint_and_dispatch`)
- The main model weights (`ema.safetensors`, ~29GB) must be downloaded from HuggingFace — see README step 2

### Running the app
- `python app.py` — requires GPU + model weights at `models/BAGEL-7B-MoT/`
- `python app.py --mode 2 --zh` — NF4 quantized mode for 12-32GB VRAM GPUs
- Gradio UI serves on `http://localhost:7860`

### Lint
- No project linter configured. `ruff` is available (installed as gradio dependency): `ruff check --select=E,F --ignore=E501,F401,E402 *.py modeling/ data/ train/ eval/`
- Pre-existing minor lint warnings exist (bare except, ambiguous var names) — do not fix unless asked

### Tests
- No automated test suite in the repository
- `eval/gen/gedit/test_gedit_score.py` is an evaluation benchmark script, not a unit test — requires GPT-4o API and eval datasets

### Model config files
- Small config/tokenizer files are downloaded to `models/BAGEL-7B-MoT/` during setup (~12MB)
- The VAE autoencoder (`ae.safetensors`, ~340MB) is also pre-downloaded
- The main model weights (`ema.safetensors`, ~29GB) are NOT downloaded — too large for CPU-only environments
