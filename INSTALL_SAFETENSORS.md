# 🔧 Fix: Missing Safetensors

## Quick Fix

```bash
pip install safetensors
```

That's it! Then try your test again:

```bash
python scripts/test_visionzip_single.py
```

---

## Why This Happened

The model uses `safetensors` format for weights (safer and faster than pickle). This package is needed to load them properly.

---

## Alternative: Install Accelerate with Offload Support

If you still get offload errors, you might also need:

```bash
pip install safetensors accelerate
```

