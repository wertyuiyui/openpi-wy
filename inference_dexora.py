import sys
import io
import time
import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------------------------------------------------------
# 1. 定义关节映射逻辑 (保持不变)
# -----------------------------------------------------------------------------

# 完整的 39 维关节名称 (按顺序)
ALL_JOINT_NAMES = [
    # Left Arm (0-5)
    "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3", "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
    # Right Arm (6-11)
    "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3", "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
    # Left Hand (12-23)
    "left_hand_joint_1", "left_hand_joint_2", "left_hand_joint_3", "left_hand_joint_4", "left_hand_joint_5", "left_hand_joint_6",
    "left_hand_joint_7", "left_hand_joint_8", "left_hand_joint_9", "left_hand_joint_10", "left_hand_joint_11", "left_hand_joint_12",
    # Right Hand (24-35)
    "right_hand_joint_1", "right_hand_joint_2", "right_hand_joint_3", "right_hand_joint_4", "right_hand_joint_5", "right_hand_joint_6",
    "right_hand_joint_7", "right_hand_joint_8", "right_hand_joint_9", "right_hand_joint_10", "right_hand_joint_11", "right_hand_joint_12",
    # Head & Spine (36-38)
    "head_joint_1", "head_joint_2", "spine_joint"
]

# 需要移除的关节名称
REMOVE_LIST = {
    "left_hand_joint_3", "left_hand_joint_5", "left_hand_joint_6", "left_hand_joint_8", "left_hand_joint_10", "left_hand_joint_12",
    "right_hand_joint_3", "right_hand_joint_5", "right_hand_joint_6", "right_hand_joint_8", "right_hand_joint_10", "right_hand_joint_12"
}

def get_keep_indices():
    """生成需要保留的维度索引列表"""
    keep_indices = []
    for idx, name in enumerate(ALL_JOINT_NAMES):
        if name not in REMOVE_LIST:
            keep_indices.append(idx)
    return np.array(keep_indices, dtype=int)

# 获取索引掩码
KEEP_INDICES = get_keep_indices()

# -----------------------------------------------------------------------------
# 2. 主逻辑
# -----------------------------------------------------------------------------

import jax
from openpi.models import model as _model
from openpi.policies import dexora_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

def decode_image(image_data):
    if image_data is None: return None
    if isinstance(image_data, np.ndarray) and image_data.ndim == 0: image_data = image_data.item()
    image_bytes = None
    if isinstance(image_data, bytes): image_bytes = image_data
    elif isinstance(image_data, dict) and 'bytes' in image_data: image_bytes = image_data['bytes']
    if image_bytes:
        try: return np.array(Image.open(io.BytesIO(image_bytes)))
        except: return None
    if isinstance(image_data, np.ndarray): return image_data
    return None

def main():
    # --- 加载配置 ---
    print(">>> Loading Policy...")
    config = _config.get_config("pi05_dexora_test")
    checkpoint_dir = "/qiyuan_research_vepfs_001/weiyi/openpi/checkpoints/pi05_dexora_test/test/29999"
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    print(">>> Policy Loaded.")

    # --- 读取数据 ---
    data_path = "/qiyuan_research_vepfs_001/weiyi/huggingface/lerobot/weiyi/dexora_close_open_drawer/data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(data_path)
    total_frames = len(df)
    
    # 预先提取所有 GT Actions 并过滤 (用于计算 MSE)
    all_gt_actions_raw = np.stack(df['actions'].values) # Shape: (N, 39)
    all_gt_actions_filtered = all_gt_actions_raw[:, KEEP_INDICES] # Shape: (N, 27)
    
    print(f">>> GT Actions Filtered Shape: {all_gt_actions_filtered.shape}")

    scores = []
    step_size = 100
    test_indices = range(0, total_frames, step_size)

    print(f">>> Starting Loop on {len(test_indices)} frames...")

    for i in test_indices:
        row = df.iloc[i]
        
        # 1. 准备输入
        img_front = decode_image(row.get('image_front'))
        img_right = decode_image(row.get('image_right'))
        img_left  = decode_image(row.get('image_left'))
        
        # [关键修正] 输入给模型的 state 必须保留完整的 39 维！
        # 模型内部会根据 config 自己处理需要哪些维度，或者它本身就是基于 39 维输入的。
        state = np.array(row.get('state'), dtype=np.float32)

        example = {
            'image_front': img_front,
            'image_right': img_right,
            'image_left': img_left,
            'state': state, # 传入原始 39 维
            'prompt': "close and open drawer",
        }

        # 2. 推理
        try:
            result = policy.infer(example)
        except Exception as e:
            print(f"Error infer at {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 3. 计算 MSE
        # Pred shape: (Chunk, 27)
        pred_chunk = np.array(result['actions'])
        if pred_chunk.ndim == 3: pred_chunk = pred_chunk[0]
        
        horizon = pred_chunk.shape[0]
        
        # 获取对应的 GT Chunk (27 维)
        end_idx = min(i + horizon, total_frames)
        gt_chunk = all_gt_actions_filtered[i : end_idx]
        
        # 长度对齐
        valid_len = min(len(pred_chunk), len(gt_chunk))
        pred = pred_chunk[:valid_len]
        gt = gt_chunk[:valid_len]

        # 维度对齐检查
        # 此时 pred 应该是 27 维 (模型输出)
        # gt 也是 27 维 (我们手动过滤的)
        if pred.shape[1] != gt.shape[1]:
            print(f"!!! Dimension mismatch: Pred {pred.shape[1]} vs GT {gt.shape[1]}")
            # 如果模型输出不是 27 维，这里可能需要进一步处理
            continue

        # 计算距离
        diff = pred - gt
        dist = np.linalg.norm(diff)
        
        scores.append(dist)
        print(f"Frame {i:04d}: Dist {dist:.4f}")

    if scores:
        avg_dist = np.mean(scores)
        print("\n" + "="*30)
        print(f"Final Average Distance: {avg_dist:.6f}")
        if avg_dist < 1.0:
            print("RESULT: PASS (Dist < 1.0)")
        else:
            print("RESULT: HIGH ERROR (Dist >= 1.0)")
        print("="*30)

if __name__ == "__main__":
    main()