# Step 7: unified inference demo. INFERENCE ONLY (no training, no weight updates). Run it after the notebook has finished.
# Each `# %%` block is a cell: VS Code runs it as an Interactive-window cell, and it can be pasted into notebook.ipynb as a "Step 7" section.
# Run as a script:   python step7_inference_demo.py        (from the code/ folder, in the project's venv)

# %% 7.1 Environment: which GPU build, is sm_61 (Pascal / Quadro P1000) supported?
import torch

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
    print("arch list:", torch.cuda.get_arch_list())

# %% 7.2 Load config, write artifacts bundle manifest (reads bundle files and hashes them; loads no model)
from pathlib import Path

from stress_signals import inference as I
from stress_signals.config import load_config
from stress_signals.utils import models_root

CFG = load_config()
ARTIFACTS = models_root(CFG)                                   # <project root>/trained_models
MANIFEST = Path(CFG["_root"]) / CFG["inference"]["bundle_manifest"]
manifest = I.write_bundle_manifest(ARTIFACTS, MANIFEST, cfg=CFG)
print("bundle_id:", manifest["bundle_id"], "->", MANIFEST)
for comp, m in manifest["components"].items():
    print(f"  {comp:8s} {m['version']}  files verified: {m['files_verified']}  manifest sha256: {m['manifest_sha256'][:16]}...")
print("missing optional components:", manifest["missing_optional_components"])

# %% 7.3 Build the pipeline from the manifest (pins the exact versions, re-verifies every file hash)
I.check_bundle_manifest(ARTIFACTS, MANIFEST)
pipe = I.StressSignalPipeline.from_artifacts(ARTIFACTS, device="auto", cfg=CFG, versions=I.versions_from_bundle_manifest(MANIFEST))
print("device:", pipe.device, "| fp16:", pipe.bundle_info["fp16"], "| stressor bundle:", pipe.has_stressor)

# %% 7.4 Tiny usage example: 5 hand-written neutral sentences (no real user text)
import pandas as pd

EXAMPLES = [
    "The train to the city leaves at nine o'clock every morning.",
    "I made a pot of tea and read a chapter of my book.",
    "The library opens at ten and closes at six on weekdays.",
    "We planted tomatoes along the back fence this weekend.",
    "The meeting notes are saved in the shared folder.",
]
df = pipe.predict(EXAMPLES, batch_size=CFG["inference"]["batch_size"])
print(df[["input_status", "stress_prob_calibrated", "stress_flag"]].round(3).to_string())
print(df.filter(like="emotion_group__").round(3).to_string())
print("last run:", pipe.last_run)                            # seconds, device, batch size, peak VRAM (CUDA), status counts

# %% 7.5 Edge cases: every bad input gives NaN scores and a reason, never an exception (synthetic inputs only)
edge = pipe.predict([None, "", "😀😀😀", "https://example.org", "Ça va très bien aujourd'hui, merci beaucoup.", EXAMPLES[0]])
print(edge[["input_status", "stress_prob_calibrated"]].round(3).to_string())

# %% 7.6 Memory and determinism checks
again = pipe.predict(EXAMPLES, batch_size=CFG["inference"]["batch_size"])
print("identical on repeat:", df.equals(again))
if pipe.device == "cuda":
    print(f"peak VRAM {pipe.last_run['peak_vram_gb']} GB (limit {CFG['inference']['max_vram_gb']} GB)")
print("columns:", len(pipe.output_columns()))
