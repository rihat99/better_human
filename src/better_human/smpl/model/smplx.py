import pickle
import numpy as np
import json
from importlib import resources

import torch
import pypose as pp
# ... (other imports)
from ..base import SMPLBase, SMPLOutputs


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

        super().__init__(**kwargs)

        smplx_data = np.load(model_path, allow_pickle=True)
        self._load_model_specific_data(smplx_data)
        self._load_model_base_data(smplx_data, model_config='smplx.json')


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


        # Face related parameters
        # TODO: Implement face deformation handling

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

        jaw_pose = pp.identity_SO3(batch_size, 1, device=self.vertices_template.device) # (B, 1, 4)
        left_eye_pose = pp.identity_SO3(batch_size, 1, device=self.vertices_template.device) # (B, 1, 4)
        right_eye_pose = pp.identity_SO3(batch_size, 1, device=self.vertices_template.device) # (B, 1, 4)

        full_pose = torch.cat([
            body_pose,
            jaw_pose,
            left_eye_pose,
            right_eye_pose,
            left_hand_pose,
            right_hand_pose,
        ], dim=1) # (B, 54, 4)

        return super().forward(betas=betas, body_pose=full_pose, global_transform=global_transform)
        
