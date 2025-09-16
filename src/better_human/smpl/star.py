import pickle
import numpy as np
import viser

import torch
import pypose as pp

from .base import SMPLBase, SMPLOutputs


class STAR(SMPLBase):
    """
    A concrete implementation of the original SMPL model.
    """
    def __init__(self, model_path: str, gender: str = 'neutral', num_betas: int = 10, **kwargs):
        self.num_betas = num_betas
        # `super().__init__` will call `_load_model_data` internally
        super().__init__(model_path=model_path, gender=gender, **kwargs)

    def _load_model_data(self, model_path: str):
        """
        Loads the STAR model data from a .npz file.
        """
        star_data = np.load(model_path, allow_pickle=True)

        # Register model parameters as buffers
        self.register_buffer('vertices_template', torch.tensor(star_data['v_template'], dtype=torch.float32)) # (6890, 3)
        
        # Faces are not used in computation but are essential for visualization
        self.register_buffer('faces', torch.tensor(star_data['f'].astype(np.int64), dtype=torch.long)) # (13776, 3)

        # Shape blend shapes
        shape_blending = torch.tensor(star_data['shapedirs'][:, :, :self.num_betas], dtype=torch.float32)
        self.register_buffer('shape_blending', shape_blending)   # (6890, 3, num_betas)

        # Pose blend shapes
        # Original shape is (6890, 3, 93). Reshape to (6890*3, 93)
        pose_blending = torch.tensor(star_data['posedirs'], dtype=torch.float32).reshape(-1, 93)
        # We need ( (J-1)*9, V*3 ) for matmul, so we transpose
        self.register_buffer('pose_blending', pose_blending) # (6870*3, 93)

        # Joint regressor
        self.register_buffer('joint_regressor', torch.tensor(star_data['J_regressor'], dtype=torch.float32)) # (24, 6890)

        # LBS weights
        self.register_buffer('lbs_weights', torch.tensor(star_data['weights'], dtype=torch.float32)) # (6890, 24)

        # Kinematic tree
        self.parent_tree = star_data['kintree_table']

        # Define number of joints based on loaded data
        self.num_joints = 24

        self.joint_links = [
            [0, 1], [0, 2], [1, 4], [4, 7], [7, 10], [2, 5], [5, 8], [8, 11],
            [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], 
            [9, 13], [13, 16], [16, 18], [18, 20], [20, 22],
            [9, 14], [14, 17], [17, 19], [19, 21], [21, 23]
        ]

    def deform_shape(self, body_pose: pp.LieTensor, neutral_vertices, betas) -> torch.Tensor:
        batch_size = body_pose.shape[0]

        pose_quat = body_pose.tensor()  # (B, 23, 4)
        w_ones = torch.zeros_like(pose_quat, device=self.device, dtype=pose_quat.dtype)  # (B, 23, 4)
        w_ones[..., 3] = 1.0  # Set the last element to 1.0
        pose_quat = pose_quat - w_ones # Normalize the quaternion to have w = 0 at rest

        pose_quat = pose_quat.reshape(-1, 23 * 4)  # (B, 23 * 4)
        pose_feat = torch.cat((pose_quat, betas[:, 1:2]), dim=1)  # (B, 23*4+1 = 93)

        pose_offsets = torch.einsum('ij,bj->bi', self.pose_blending, pose_feat)  # (B, 6890*3)
        pose_offsets = pose_offsets.view(batch_size, -1, 3)  # (B, 6890, 3)

        vertices_blended = neutral_vertices + pose_offsets # (B, 6890, 3)

        return vertices_blended

    @property
    def joint_names(self) -> list[str]:
        # Standard SMPL joint names
        return [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
            'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'left_hand', 'right_hand'
        ]