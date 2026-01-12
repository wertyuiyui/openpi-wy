import shutil
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
import numpy as np
from typing import List
import glob
import json
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader, Subset

# 忽略 DataLoader 的一些多进程警告
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True, precision=4, linewidth=np.inf)

RAW_DATASET_NAMES = [
    "close_open_drawer"
]

def load_task_map(dataset_dir: Path) -> Dict[int, str]:
    """Loads meta/tasks.jsonl to map task_index to natural language instruction."""
    task_map = {}
    task_file = os.path.join(dataset_dir, "meta/tasks.jsonl")
    
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

def main(data_dir: str, REPO_NAME: str):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    print(f"Output path: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="airbot",
        fps=10,
        features={
            "image_front": {
                "dtype": "image",
                "shape": (3,480,640),
                "names": ["height", "width", "channel"],
            },
            "image_right": {
                "dtype": "image",
                "shape": (3,480,640),
                "names": ["height", "width", "channel"],  
            },
            "image_left": {
                "dtype": "image",
                "shape": (3,480,640),
                "names": ["height", "width", "channel"],  
            },
            "state": {
                "dtype": "float32",
                "shape": (39,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (39,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


    # 从文件列表中加载数据集
    for sub_name in RAW_DATASET_NAMES:
        sub_dir = os.path.join(data_dir, sub_name)

        print(f"Processing sub-dataset: {sub_name}")
        task_map = load_task_map(sub_dir)

        try:
            # 必须使用 LeRobotDataset 来正确处理视频解码
            source_ds = LeRobotDataset(str(sub_dir))
        except Exception as e:
            print(f"Error loading {sub_name}: {e}")
            continue
        
        # Iterate over episodes
        for ep_idx in range(source_ds.num_episodes):
            start_frame = source_ds.episode_data_index["from"][ep_idx].item()
            end_frame = source_ds.episode_data_index["to"][ep_idx].item()
            
            # --- 优化点：使用 DataLoader 并行读取 ---
            # 1. 创建仅包含当前 episode 帧的子数据集
            episode_indices = range(start_frame, end_frame)
            subset = Subset(source_ds, episode_indices)

            # 2. 使用 DataLoader 利用多进程 (num_workers) 并行解码视频帧
            # batch_size 可以设大一点，比如 16 或 32，减少 Python 循环开销
            loader = DataLoader(
                subset, 
                batch_size=32, 
                shuffle=False, 
                num_workers=8, # 根据你的 CPU 核心数调整，8-16 比较合适
                collate_fn=None # 默认 collate 会自动把 list of dicts 堆叠成 dict of tensors
            )

            for batch in loader:
                # DataLoader 返回的是 batch 字典，value 是 Tensor (B, ...)
                # 我们需要将其拆回单独的帧
                
                # 获取 batch 大小
                # 随意取一个 key 来查看 batch size
                batch_len = len(batch["index"]) if "index" in batch else len(batch[next(iter(batch))])
                
                # 预先提取需要的数据，避免在循环中重复查询字典
                b_img_front = batch.get("observation.images.front")
                b_img_right = batch.get("observation.images.wrist_right")
                b_img_left = batch.get("observation.images.wrist_left")
                b_state = batch.get("observation.state")
                b_action = batch.get("action")
                b_task_idx = batch.get("task_index")

                for i in range(batch_len):
                    # 从 Batch Tensor 中提取第 i 个数据
                    # 注意：如果 source_ds 返回的是 Tensor，这里拿到的也是 Tensor
                    # add_frame 支持 Tensor 或 Numpy
                    
                    img_front = b_img_front[i] if b_img_front is not None else None
                    img_right = b_img_right[i] if b_img_right is not None else None
                    img_left = b_img_left[i] if b_img_left is not None else None
                    state = b_state[i] if b_state is not None else None
                    action = b_action[i] if b_action is not None else None
                    
                    # Task description
                    task_desc = ""
                    if b_task_idx is not None:
                        t_idx = b_task_idx[i]
                        # Convert tensor scalar to python int if necessary
                        if hasattr(t_idx, "item"):
                            t_idx = t_idx.item()
                        task_desc = task_map.get(t_idx, "")

                    dataset.add_frame({
                        "image_front": img_front,
                        "image_right": img_right,
                        "image_left": img_left,
                        "state": state,    # 直接传 Tensor 即可
                        "actions": action, # 直接传 Tensor 即可
                        "task": task_desc,
                    })

            dataset.save_episode()

if __name__ == "__main__":
    tyro.cli(main)