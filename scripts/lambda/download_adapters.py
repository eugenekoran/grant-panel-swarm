"""Download the base model and the pinned AW adapters (MODEL_SERVING.md §2). Runs on the GPU host."""
from huggingface_hub import snapshot_download

ADAPTERS = {
    "aw-sdf-none": ("auditing-agents/qwen_14b_synth_docs_only_animal_welfare",
                    "858bec2505ef37878aef1ed29037d638743edc56"),
    "aw-sdf-sft": ("auditing-agents/qwen_14b_synth_docs_only_then_redteam_high_animal_welfare",
                   "e4d69a34425cf3b11994c887087a744e464520cf"),
    "aw-sdf-kto": ("auditing-agents/qwen_14b_synth_docs_only_then_redteam_kto_animal_welfare",
                   "439d796db057f349dc46301bd80cf82f8398b05d"),
}
for name, (repo, rev) in ADAPTERS.items():
    snapshot_download(repo, revision=rev, local_dir=f"adapters/{name}")
snapshot_download("Qwen/Qwen3-14B")
print("done")
