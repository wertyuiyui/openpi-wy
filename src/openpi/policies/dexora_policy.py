import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 定义需要保留的维度索引 (0-based)
# 总共 39 维，去掉左右手的 3, 5, 6, 8, 10, 12 (对应索引见下文注释)，剩余 27 维
KEEP_INDICES = np.array([
    # Left Arm (0-5) - 全部保留
    0, 1, 2, 3, 4, 5,
    # Right Arm (6-11) - 全部保留
    6, 7, 8, 9, 10, 11,
    # Left Hand (12-23)
    12, 13,     # 保留 joint 1, 2
    # 14,       # 去掉 left_hand_joint_3
    15,         # 保留 joint 4
    # 16, 17,   # 去掉 left_hand_joint_5, 6
    18,         # 保留 joint 7
    # 19,       # 去掉 left_hand_joint_8
    20,         # 保留 joint 9
    # 21,       # 去掉 left_hand_joint_10
    22,         # 保留 joint 11
    # 23,       # 去掉 left_hand_joint_12
    # Right Hand (24-35)
    24, 25,     # 保留 joint 1, 2
    # 26,       # 去掉 right_hand_joint_3
    27,         # 保留 joint 4
    # 28, 29,   # 去掉 right_hand_joint_5, 6
    30,         # 保留 joint 7
    # 31,       # 去掉 right_hand_joint_8
    32,         # 保留 joint 9
    # 33,       # 去掉 right_hand_joint_10
    34,         # 保留 joint 11
    # 35,       # 去掉 right_hand_joint_12
    # Head & Spine (36-38) - 全部保留
    36, 37, 38
], dtype=np.int64)


def make_dexora_example() -> dict:
    """Creates a random input example for the Dexora policy."""
    return {
        "state": np.random.rand(39),
        "image_front": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "image_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "image_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DexoraInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    action_dim: int
    
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["image_front"])
        left_wrist_image = _parse_image(data["image_left"])
        right_wrist_image = _parse_image(data["image_right"])
        
        # --- 修改部分开始 ---
        # 提取 state 中的 27 维
        raw_state = np.asarray(data["state"])
        state = raw_state[KEEP_INDICES]
        state = transforms.pad_to_dim(state, self.action_dim)
        # --- 修改部分结束 ---

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # --- 修改部分开始 ---
            # 提取 actions 中的 27 维
            raw_actions = np.asarray(data["actions"])
            # 假设 actions 形状可能是 (..., 39)，在最后一个维度进行筛选
            filtered_actions = raw_actions[..., KEEP_INDICES]
            
            # 如果需要 pad 到模型维度 (通常 Pi0 需要 pad 到比如 32 或 64)
            actions = transforms.pad_to_dim(filtered_actions, self.action_dim)
            inputs["actions"] = actions
            # --- 修改部分结束 ---

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DexoraOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 27 actions (modified from 39)
        # We assume the model was trained on the 27-dim vectors filtered above.
        return {"actions": np.asarray(data["actions"][:, :27])}