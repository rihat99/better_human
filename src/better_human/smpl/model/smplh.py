import pickle
import numpy as np
import json
from importlib import resources


import torch
import pypose as pp
# ... (other imports)
from ..base import SMPLBase, SMPLOutputs


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
        self.gender = gender
        self.use_pca = use_pca
        self.num_pca_components = num_pca_components
        if num_pca_components == 45:
            self.use_pca = False  # Override to use full pose if 45 components are requested
        self.flat_hand_mean = flat_hand_mean

        # `super().__init__` will call `_load_model_data` internally
        super().__init__(**kwargs)

        smplh_data = pickle.load(open(model_path, 'rb'), encoding='latin1')
        self._load_model_specific_data(smplh_data)
        self._load_model_base_data(smplh_data, model_config='smplh.json')

    def _load_model_specific_data(self, model_data: dict):
        
        # Hands related parameters
        if self.flat_hand_mean:
            self.register_buffer('hand_mean_left', torch.zeros((15, 3), dtype=torch.float32)) # (15, 3)
            self.register_buffer('hand_mean_right', torch.zeros((15, 3), dtype=torch.float32)) # (15, 3)
        else:
            self.register_buffer('hand_mean_left', torch.tensor(model_data['hands_meanl'].reshape(15, 3), dtype=torch.float32)) # (15, 3)
            self.register_buffer('hand_mean_right', torch.tensor(model_data['hands_meanr'].reshape(15, 3), dtype=torch.float32)) # (15, 3)

        self.register_buffer('hand_components_left', torch.tensor(model_data['hands_componentsl'], dtype=torch.float32)) # (45, 45)
        self.register_buffer('hand_components_right', torch.tensor(model_data['hands_componentsr'], dtype=torch.float32)) # (45, 45)


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



        
        
