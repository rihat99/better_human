import pickle
import numpy as np
import viser

import torch
import pypose as pp
# ... (other imports)
from .base import SMPLBase, SMPLOutputs


class SMPLH(SMPLBase):
    """
    A concrete implementation of the SMPL-H model, which extends SMPL
    with hand pose parameters.
    """
    def __init__(
            self, 
            model_path: str, 
            gender: str = 'neutral', 
            num_betas: int = 16,
            use_pca: bool = True,
            num_pca_components: int = 6,
            flat_hand_mean: bool = False,
            **kwargs):
        
        self.num_betas = num_betas
        self.use_pca = use_pca
        self.num_pca_components = num_pca_components
        if num_pca_components == 45:
            self.use_pca = False  # Override to use full pose if 45 components are requested
        self.flat_hand_mean = flat_hand_mean
        # `super().__init__` will call `_load_model_data` internally
        super().__init__(model_path=model_path, gender=gender, **kwargs)

        self.num_joints = 52  # SMPL-H has 52 joints (24 body + 15 left hand + 15 right hand - 2 overlap)

    def _load_model_data(self, model_path: str):
        smplh_data = pickle.load(open(model_path, 'rb'), encoding='latin1')

        # Register model parameters as buffers
        self.register_buffer('vertices_template', torch.tensor(smplh_data['v_template'], dtype=torch.float32)) # (6890, 3)

        # Faces are not used in computation but are essential for visualization
        self.register_buffer('faces', torch.tensor(smplh_data['f'].astype(np.int64), dtype=torch.long)) # (13776, 3)

        # Shape blend shapes
        shape_blending = torch.tensor(smplh_data['shapedirs'][:, :, :self.num_betas], dtype=torch.float32)
        self.register_buffer('shape_blending', shape_blending)   # (6890, 3, num_betas)

        # Pose blend shapes
        # Original shape is (6890, 3, 459). Reshape to (6890*3, 459)
        pose_blending = torch.tensor(smplh_data['posedirs'], dtype=torch.float32).reshape(-1, 459).T
        # We need ( (J-1)*9, V*3 ) for matmul, so we transpose
        self.register_buffer('pose_blending', pose_blending) # (459, 6890*3)

        # Joint regressor
        self.register_buffer('joint_regressor', torch.tensor(smplh_data['J_regressor'], dtype=torch.float32)) # (52, 6890)

        # LBS weights
        self.register_buffer('lbs_weights', torch.tensor(smplh_data['weights'], dtype=torch.float32)) # (6890, 52)

        # Kinematic tree
        self.parent_tree = smplh_data['kintree_table']

        # Hands related parameters
        if self.flat_hand_mean:
            self.register_buffer('hand_mean_left', torch.zeros((15, 3), dtype=torch.float32)) # (15, 3)
            self.register_buffer('hand_mean_right', torch.zeros((15, 3), dtype=torch.float32)) # (15, 3)
        else:
            self.register_buffer('hand_mean_left', torch.tensor(smplh_data['hands_meanl'].reshape(15, 3), dtype=torch.float32)) # (15, 3)
            self.register_buffer('hand_mean_right', torch.tensor(smplh_data['hands_meanr'].reshape(15, 3), dtype=torch.float32)) # (15, 3)

        self.register_buffer('hand_components_left', torch.tensor(smplh_data['hands_componentsl'], dtype=torch.float32)) # (45, 45)
        self.register_buffer('hand_components_right', torch.tensor(smplh_data['hands_componentsr'], dtype=torch.float32)) # (45, 45)


    def forward(
        self,
        betas: torch.Tensor,
        body_pose: pp.LieTensor,
        global_transform: pp.LieTensor,
        left_hand_pose: pp.LieTensor,
        right_hand_pose: pp.LieTensor,
    ):
        
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

        full_pose = torch.cat([body_pose, left_hand_pose, right_hand_pose], dim=1) # (B, 51, 4)

        return super().forward(betas=betas, body_pose=full_pose, global_transform=global_transform)



        
        
