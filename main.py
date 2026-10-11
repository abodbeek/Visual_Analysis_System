
import os
from pathlib import Path

# Prevent downloading model files from the internet
os.environ["HF_HUB_OFFLINE"] = "1"

import cv2
import numpy as np
import torch

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration
)
from transformers.video_utils import VideoMetadata


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "qwen3-vl-2b"

VIDEO_PATH = BASE_DIR / "samples" / "test.mp4"

DEVICE = "cpu"

MAX_FRAME_SIZE = 640

torch.set_num_threads(4)


# ==========================================
# 2. CHECK MODEL AND VIDEO FILES
# ==========================================

if not MODEL_PATH.is_dir():
    raise FileNotFoundError(
        f"Model folder not found: {MODEL_PATH}\n"
        "Run prepare_model.py first."
    )

if not (MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(
        f"Model configuration not found: {MODEL_PATH}"
    )

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )


# ==========================================
# 3. READ VIDEO WITH OPENCV
# ==========================================

def prepare_video(video_path):

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError("Cannot open video.")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0 or total_frames < 2:
            raise RuntimeError(
                "Invalid video information."
            )

        duration = total_frames / fps

        # Sample 8 frames with more attention
        # to the final part of the video
        ratios = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.75,
            0.85,
            0.92,
            0.98
        ]

        indexes = sorted(set(
            min(
                total_frames - 1,
                round((total_frames - 1) * ratio)
            )
            for ratio in ratios
        ))

        frames = []
        actual_indexes = []

        for index in indexes:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                index
            )

            success, frame = capture.read()

            if not success:
                continue

            height, width = frame.shape[:2]

            scale = min(
                1.0,
                MAX_FRAME_SIZE / max(height, width)
            )

            new_width = max(
                1,
                round(width * scale)
            )

            new_height = max(
                1,
                round(height * scale)
            )

            frame = cv2.resize(
                frame,
                (new_width, new_height)
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)
            actual_indexes.append(int(index))

        if len(frames) < 2:
            raise RuntimeError(
                "Not enough frames extracted."
            )

        # Qwen3-VL video frames are processed in pairs
        if len(frames) % 2 != 0:

            frames.append(
                frames[-1].copy()
            )

            actual_indexes.append(
                actual_indexes[-1]
            )

        video_array = np.stack(
            frames,
            axis=0
        )

        metadata = VideoMetadata(
            total_num_frames=total_frames,
            fps=float(fps),
            duration=float(duration),
            frames_indices=actual_indexes,
            width=video_array.shape[2],
            height=video_array.shape[1]
        )

        timestamps = [
            round(index / fps, 2)
            for index in actual_indexes
        ]

        return (
            video_array,
            metadata,
            duration,
            timestamps
        )

    finally:
        capture.release()


# ==========================================
# 4. PREPARE VIDEO
# ==========================================

print("Reading video...")

video, metadata, duration, timestamps = (
    prepare_video(VIDEO_PATH)
)

print(f"Duration: {duration:.2f} seconds")
print(f"Frames: {len(video)}")
print(f"Sampled seconds: {timestamps}")


# ==========================================
# 5. LOAD LOCAL VLM
# ==========================================

print("\nLoading local Qwen3-VL model...")
print(f"Model folder: {MODEL_PATH}")

processor = AutoProcessor.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    str(MODEL_PATH),
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    local_files_only=True
)

model = model.to(DEVICE)
model.eval()

print("Local model loaded successfully!")


# ==========================================
# 6. VIDEO ANALYSIS PROMPT
# ==========================================

prompt = """
Analyze this short civilian traffic video.

Write ONLY these four lines in English:

SCENE: Describe the visible environment briefly.

OBJECTS: Mention the main visible object types.

EVENT: Describe the most important visible change.

COLLISION: YES, NO, or UNCERTAIN.
Give one short reason.

STRICT RULES:
- Maximum 70 words total.
- Do not list video frames.
- Do not invent timestamps.
- Do not invent exact object counts.
- Do not guess intentions or speeds.
- Do not determine the cause of an accident.
- Describe a collision only if clearly visible.
- If evidence is insufficient, say UNCERTAIN.
- Do not add extra sections.
"""


# ==========================================
# 7. PREPARE VLM INPUT
# ==========================================

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
]

print("\nPreparing video for VLM...")

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={
        "videos_kwargs": {
            "do_sample_frames": False,
            "video_metadata": [metadata],
            "cap_pixels_per_frame": True
        }
    }
).to(DEVICE)

print("Video ready!")


# ==========================================
# 8. GENERATE REPORT
# ==========================================

print("\nAnalyzing video...")

with torch.inference_mode():

    output = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4
    )

input_length = inputs["input_ids"].shape[1]

new_tokens = output[:, input_length:]

report = processor.batch_decode(
    new_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]


# ==========================================
# 9. DISPLAY REPORT
# ==========================================

print("\n" + "=" * 55)
print("           AEROVISION AI")
print("         VIDEO ANALYSIS REPORT")
print("=" * 55)

print(f"Video duration: {duration:.2f} seconds")
print(f"Frames examined: {len(video)}")

print("\n" + report)

print("\n" + "=" * 55)
print("Analysis completed successfully!")
