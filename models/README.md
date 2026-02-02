# VLA Model Cache Directory

This directory stores the Vision-Language-Action (VLA) model weights downloaded from HuggingFace.

## Model Information

- **Model**: pi0.5 with LoRA fine-tuning
- **HuggingFace ID**: `shehiin/pi05_pick_red_cube_lora`
- **Framework**: LeRobot
- **Architecture**: Transformer-based policy network
- **Size**: ~500MB

## Directory Structure

```
models/
└── vla_cache/
    ├── config.json              # Model configuration
    ├── policy.safetensors       # Model weights (main)
    ├── preprocessor_config.json # Image preprocessing config
    └── tokenizer/               # Language tokenizer
```

## First Run

On first startup, the application will automatically download the VLA model from HuggingFace:

1. Downloads model weights (~500MB)
2. Caches them locally in this directory
3. Subsequent runs load from cache (much faster)

## Manual Download

You can pre-download the model using the HuggingFace CLI:

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Download the model
huggingface-cli download shehiin/pi05_pick_red_cube_lora \
    --local-dir models/vla_cache \
    --local-dir-use-symlinks False
```

## Troubleshooting

### Slow Download
If download is slow, try:
- Using a HuggingFace token (set HF_TOKEN in .env)
- Checking your internet connection
- Using a mirror site

### Out of Space
The model requires ~1GB of disk space (including dependencies).
Clear cache if needed:
```bash
rm -rf models/vla_cache/*
```

### Permission Errors
Ensure the application has write access to this directory:
```bash
chmod -R 755 models/
```

## Fallback Mode

If the VLA model cannot be loaded:
- The system automatically falls back to trajectory-based control
- Pre-recorded action sequences in `trajectories/` are used instead
- Logs will indicate "trajectory_fallback" mode

## Model Updates

To update to a newer version:
1. Clear the cache: `rm -rf models/vla_cache/*`
2. Update model_id in `vla_config.ini`
3. Restart the application

---

**Note**: This directory is auto-generated and managed by the application.
Do not manually edit files unless you know what you're doing.
