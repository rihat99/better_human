import pickle
import numpy as np
import viser

import torch
import pypose as pp
# ... (other imports)
from .base import SMPLBase, SMPLOutputs

class SMPL(SMPLBase):
    """
    A concrete implementation of the original SMPL model.
    """
    def __init__(self, model_path: str, gender: str = 'neutral', num_betas: int = 10, **kwargs):
        self.num_betas = num_betas
        # `super().__init__` will call `_load_model_data` internally
        super().__init__(model_path=model_path, gender=gender, **kwargs)

    def _load_model_data(self, model_path: str):
        """
        Loads the SMPL model data from a .pkl file.
        """
        smpl_data = np.load(model_path, allow_pickle=True)

        # Register model parameters as buffers
        self.register_buffer('vertices_template', torch.tensor(smpl_data['vertices_template'], dtype=torch.float32)) # (6890, 3)
        
        # Faces are not used in computation but are essential for visualization
        self.register_buffer('faces', torch.tensor(smpl_data['faces'].astype(np.int64), dtype=torch.long)) # (13776, 3)

        # Shape blend shapes
        shape_blending = torch.tensor(smpl_data['shape_blending'][:, :, :self.num_betas], dtype=torch.float32)
        self.register_buffer('shape_blending', shape_blending)   # (6890, 3, num_betas)

        # Pose blend shapes
        # Original shape is (6890, 3, 207). Reshape to (6890*3, 207)
        pose_blending = torch.tensor(smpl_data['pose_blending'], dtype=torch.float32).reshape(-1, 23*9).T
        # We need ( (J-1)*9, V*3 ) for matmul, so we transpose
        self.register_buffer('pose_blending', pose_blending) # (207, 6890*3)

        # Joint regressor
        self.register_buffer('joint_regressor', torch.tensor(smpl_data['joint_regressor'], dtype=torch.float32)) # (24, 6890)

        # LBS weights
        self.register_buffer('lbs_weights', torch.tensor(smpl_data['weights'], dtype=torch.float32)) # (6890, 24)

        # Kinematic tree
        self.parent_tree = smpl_data['kintree_table']

        # Define number of joints based on loaded data
        self.num_joints = 24

        self.joint_links = [
            [0, 1], [0, 2], [1, 4], [4, 7], [7, 10], [2, 5], [5, 8], [8, 11],
            [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], 
            [9, 13], [13, 16], [16, 18], [18, 20], [20, 22],
            [9, 14], [14, 17], [17, 19], [19, 21], [21, 23]
        ]


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