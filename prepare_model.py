from pathlib import Path
from shutil import copytree
from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

BASE_DIR = Path(__file__).resolve().parent
DESTINATION = BASE_DIR / "models" / "qwen3-vl-2b"

print("Locating the previously downloaded model...")

source = snapshot_download(
    repo_id=MODEL_NAME,
    local_files_only=True
)

print("Copying model files into the project...")

copytree(
    source,
    DESTINATION,
    dirs_exist_ok=True,
    symlinks=False
)

print("Model ready at:", DESTINATION)