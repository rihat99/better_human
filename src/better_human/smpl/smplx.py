import pickle
import numpy as np
import json
from importlib import resources

import torch
import pypose as pp
# ... (other imports)
from .base import SMPLBase, SMPLOutputs


class SMPLX(SMPLBase):
    """
    A concrete implementation of the SMPL-X model, which extends SMPL
    with hand and face pose parameters.
    """
    def __init__(
            self, 
            model_path: str,
            gender: str = 'neutral',
            num_betas: int = 10,
            use_pca: bool = True,
            num_pca_components: int = 6,
            flat_hand_mean: bool = False,
            **kwargs
    ):
        
        self.num_betas = num_betas
        self.gender = gender
        self.use_pca = use_pca
        self.num_pca_components = num_pca_components
        if num_pca_components == 45:
            self.use_pca = False  # Override to use full pose if 45 components are requested
        self.flat_hand_mean = flat_hand_mean

        # `super().__init__` will call `_load_model_data` internally
        super().__init__(model_path)



    def _load_model_data(self, model_path: str):
        smplx_data = np.load(model_path, allow_pickle=True)

        # Register model parameters as buffers
        self.register_buffer('vertices_template', torch.tensor(smplx_data['v_template'], dtype=torch.float32)) # (10475, 3)

        # Faces are not used in computation but are essential for visualization
        self.register_buffer('faces', torch.tensor(smplx_data['f'].astype(np.int64), dtype=torch.long)) # (21078, 3)

        # Shape blend shapes
        shape_blending = torch.tensor(smplx_data['shapedirs'][:, :, :self.num_betas], dtype=torch.float32)
        self.register_buffer('shape_blending', shape_blending)   # (10475, 3, num_betas)

        # Pose blend shapes
        # Original shape is (10475, 3, 486). Reshape to (10475*3, 486)
        pose_blending = torch.tensor(smplx_data['posedirs'], dtype=torch.float32).reshape(-1, 486).T
        # We need ( (J-1)*9, V*3 ) for matmul, so we transpose
        self.register_buffer('pose_blending', pose_blending) # (486, 10475*3)

        # Joint regressor
        self.register_buffer('joint_regressor', torch.tensor(smplx_data['J_regressor'], dtype=torch.float32)) # (55, 10475)

        # LBS weights
        self.register_buffer('lbs_weights', torch.tensor(smplx_data['weights'], dtype=torch.float32)) # (10475, 55)

        # Kinematic tree
        self.parent_tree = smplx_data['kintree_table']

        # Hands related parameters
        if self.flat_hand_mean:
            self.register_buffer('hand_mean_left', torch.zeros((15, 3), dtype=torch.float32)) # (15, 3)
            self.register_buffer('hand_mean_right', torch.zeros((15, 3), dtype=torch.float32)) # (15, 3)
        else:
            self.register_buffer('hand_mean_left', torch.tensor(smplx_data['hands_meanl'].reshape(15, 3), dtype=torch.float32)) # (15, 3)
            self.register_buffer('hand_mean_right', torch.tensor(smplx_data['hands_meanr'].reshape(15, 3), dtype=torch.float32)) # (15, 3)

        self.register_buffer('hand_components_left', torch.tensor(smplx_data['hands_componentsl'], dtype=torch.float32)) # (45, 45)
        self.register_buffer('hand_components_right', torch.tensor(smplx_data['hands_componentsr'], dtype=torch.float32)) # (45, 45)


        # Face related parameters
        # TODO: Implement face deformation handling


        # load config as class attributes
        with resources.files('better_human.smpl.config').joinpath('smplx.json').open('r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)


    def forward(self,
        betas: torch.Tensor,
        body_pose: pp.LieTensor,
        left_hand_pose: pp.LieTensor,
        right_hand_pose: pp.LieTensor,
        global_transform: pp.LieTensor,
        # face_pose: pp.LieTensor,  # TODO: Add face pose handling
    ) -> SMPLOutputs:
        
        batch_size = betas.shape[0]
        
        if self.use_pca:
            left_hand_pose = torch.einsum('bc, cj -> bj', left_hand_pose, self.hand_components_left[:self.num_pca_components, :]) # (B, 45)
            right_hand_pose = torch.einsum('bc, cj -> bj', right_hand_pose, self.hand_components_right[:self.num_pca_components, :]) # (B, 45)

            left_hand_pose = pp.so3(left_hand_pose.reshape(-1, 15, 3)).Exp() # (B, 15, 4)
            right_hand_pose = pp.so3(right_hand_pose.reshape(-1, 15, 3)).Exp() # (B, 15, 4)
        else:
            left_hand_pose = pp.SO3(left_hand_pose) # (B, 15, 4)
            right_hand_pose = pp.SO3(right_hand_pose) # (B, 15, 4)

        left_hand_pose = (left_hand_pose.Log() + self.hand_mean_left).Exp()
        right_hand_pose = (right_hand_pose.Log() + self.hand_mean_right).Exp()

        jaw_pose = pp.identity_SO3(batch_size, 1, device=self.device) # (B, 1, 4)
        left_eye_pose = pp.identity_SO3(batch_size, 1, device=self.device) # (B, 1, 4)
        right_eye_pose = pp.identity_SO3(batch_size, 1, device=self.device) # (B, 1, 4)

        full_pose = torch.cat([
            body_pose,
            jaw_pose,
            left_eye_pose,
            right_eye_pose,
            left_hand_pose,
            right_hand_pose,
        ], dim=1) # (B, 54, 4)

        return super().forward(betas=betas, body_pose=full_pose, global_transform=global_transform)
        
