import pickle
import numpy as np
import json
from importlib import resources

import torch
import pypose as pp

from ..base import SMPLBase, SMPLOutputs


class MANO(SMPLBase):
    """
    A concrete implementation of the MANO hand model.
    """
    def __init__(
            self, 
            model_path: str, 
            hand_side: str = 'right', 
            use_pca: bool = True,
            num_pca_components: int = 6,
            flat_hand_mean: bool = False,
            **kwargs):
        
        self.num_betas = 10
        self.use_pca = use_pca
        self.num_pca_components = num_pca_components
        if num_pca_components == 45:
            self.use_pca = False  # Override to use full pose if 45 components are requested
        self.flat_hand_mean = flat_hand_mean

        self.hand_side = hand_side  # 'right' or 'left'
        # `super().__init__` will call `_load_model_data` internally
        super().__init__(**kwargs)

        mano_data = np.load(model_path, allow_pickle=True)
        self._load_model_specific_data(mano_data)
        self._load_model_base_data(mano_data, model_config='mano.json')


    def _load_model_specific_data(self, model_data):
        """
        Loads the MANO model data from a .pkl file.
        """
        
        # Register hand related parameters
        self.register_buffer('hands_components', torch.tensor(model_data['hands_components'], dtype=torch.float32)) # (45, 45)
        if self.flat_hand_mean:
            self.register_buffer('hands_mean', torch.zeros((15, 3), dtype=torch.float32)) # (15, 3)
        else:
            self.register_buffer('hands_mean', torch.tensor(model_data['hands_mean'].reshape(15, 3), dtype=torch.float32)) # (15, 3)
        # self.register_buffer('hands_coeffs', torch.tensor(model_data['hands_coeffs'], dtype=torch.float32)) # (1554, 45)

        # load config as class attributes
        with resources.files('better_human.smpl.config').joinpath('mano.json').open('r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)

    
    def forward(self, hand_pose: torch.Tensor, global_transform: pp.LieTensor, betas: torch.Tensor = None, **kwargs) -> SMPLOutputs:

        if self.use_pca:
            hand_pose = torch.einsum('bc, cj -> bj', hand_pose, self.hands_components[:self.num_pca_components, :])  # (B, 45)
            hand_pose = pp.so3(hand_pose.reshape(-1, 15, 3)).Exp()  # (B, 15, 4)
        else:
            hand_pose = pp.SO3(hand_pose)  # (B, 15, 4)

        hand_pose = (hand_pose.Log() + self.hands_mean).Exp()

        return super().forward(betas=betas, body_pose=hand_pose, global_transform=global_transform, **kwargs)

