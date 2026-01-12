"""
Script to convert the Dexora dataset to a LeRobot/Pi0 compatible format.
Fix: Explicitly resizes images (480x640 -> 256x256) and permutes dimensions (CHW -> HWC).

Usage:
    uv run scripts/convert_dexora_to_openpi.py --data_dir /path/to/Dexora_Real-World_Dataset
"""

import shutil
import json
from pathlib import Path
from typing import Dict

import torch
import tyro
import torchvision.transforms.functional as TF  # Added for resizing

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# --- Configuration ---
REPO_NAME = "dexora/open_vla_merged"

RAW_DATASET_NAMES = [
    "airbot_articulation",
    "airbot_assemble",
    "airbot_dexterous",
    "airbot_pick_and_place",
]

FPS = 20
IMAGE_SIZE = (256, 256)  # Target size
STATE_DIM = 39 
ACTION_DIM = 39

def load_task_map(dataset_dir: Path) -> Dict[int, str]:
    """Loads meta/tasks.jsonl to map task_index to natural language instruction."""
    task_map = {}
    task_file = dataset_dir / "meta/tasks.jsonl"
    
    if not task_file.exists():
        return {}

    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                idx = data.get("task_index")
                desc = data.get("task") or data.get("instruction") or data.get("language")
                if idx is not None and desc:
                    task_map[idx] = desc
            except json.JSONDecodeError:
                continue
    return task_map

def main(data_dir: Path, *, push_to_hub: bool = False):
    data_dir = Path(data_dir)
    output_path = HF_LEROBOT_HOME / REPO_NAME

    # 1. Clean up existing output
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # 2. Define Features (HWC format for compatibility)
    features = {
        "image": {
            "dtype": "image",
            "shape": (*IMAGE_SIZE, 3), # (256, 256, 3)
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (*IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["actions"],
        },
    }

    print(f"Creating LeRobot dataset at: {output_path}")
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        fps=FPS,
        robot_type="airbot_play",
        features=features,
        image_writer_threads=10, 
        image_writer_processes=5,
    )

    # 3. Iterate over sub-datasets
    for sub_name in RAW_DATASET_NAMES:
        sub_dir = data_dir / sub_name
        if not sub_dir.exists():
            print(f"Skipping {sub_name} (not found)")
            continue

        print(f"Processing sub-dataset: {sub_name}")
        task_map = load_task_map(sub_dir)

        try:
            source_ds = LeRobotDataset(str(sub_dir))
        except Exception as e:
            print(f"Error loading {sub_name}: {e}")
            continue
        
        # Iterate over episodes
        for ep_idx in range(source_ds.num_episodes):
            start_frame = source_ds.episode_data_index["from"][ep_idx].item()
            end_frame = source_ds.episode_data_index["to"][ep_idx].item()
            
            frames_added = 0
            for global_frame_idx in range(start_frame, end_frame):
                try:
                    item = source_ds[global_frame_idx]
                except Exception as e:
                    print(f"Error reading frame {global_frame_idx}: {e}")
                    continue

                # --- 1. Main Image ---
                img = item.get("observation.images.front")
                if img is None:
                    img = item.get("observation.images.top")
                
                if img is None:
                    if frames_added == 0: 
                        print(f"Warning: No image in ep {ep_idx}")
                    continue

                # Process Image: Resize (C,H,W) -> Permute (H,W,C)
                # Input img shape is (3, 480, 640)
                img = TF.resize(img, IMAGE_SIZE, antialias=True) # -> (3, 256, 256)
                img = img.permute(1, 2, 0)                       # -> (256, 256, 3)

                # --- 2. Wrist Image ---
                wrist = item.get("observation.images.wrist_left")
                if wrist is None:
                    wrist = item.get("observation.images.wrist_right")
                
                if wrist is not None:
                    # Resize and Permute real wrist image
                    wrist = TF.resize(wrist, IMAGE_SIZE, antialias=True)
                    wrist = wrist.permute(1, 2, 0)
                else:
                    # Create black image matching the processed main image shape
                    wrist = torch.zeros_like(img)

                # --- 3. State & Action & Task ---
                state = item.get("observation.state")
                action = item.get("action")
                
                task_desc = ""
                t_idx = item.get("task_index")
                if t_idx is not None:
                    t_idx_val = t_idx.item() if isinstance(t_idx, torch.Tensor) else t_idx
                    task_desc = task_map.get(t_idx_val, "")
                
                if not task_desc:
                    instr = item.get("language_instruction")
                    if instr is not None:
                        task_desc = str(instr)

                # Write to buffer
                dataset.add_frame({
                    "image": img,
                    "wrist_image": wrist,
                    "state": state,
                    "actions": action,
                    "task": task_desc,
                })
                frames_added += 1

            if frames_added > 0:
                dataset.save_episode()
                if ep_idx % 10 == 0:
                    print(f"  Processed episode {ep_idx}/{source_ds.num_episodes}")

    if push_to_hub:
        print(f"Pushing to Hugging Face Hub: {REPO_NAME}")
        dataset.push_to_hub(
            tags=["dexora", "openpi", "airbot"],
            private=True,
            license="apache-2.0",
        )
    
    print(f"Done! Dataset saved to {output_path}")

if __name__ == "__main__":
    tyro.cli(main)