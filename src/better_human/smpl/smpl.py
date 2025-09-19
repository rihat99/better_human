import pickle
import numpy as np
import json
from importlib import resources

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
        self.gender = gender

        # `super().__init__` will call `_load_model_data` internally
        super().__init__(model_path=model_path, **kwargs)

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

        # load config as class attributes
        with resources.files('better_human.smpl.config').joinpath('smpl.json').open('r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)