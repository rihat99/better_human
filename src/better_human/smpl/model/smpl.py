import pickle
import numpy as np
import json
from importlib import resources

import torch
import pypose as pp
# ... (other imports)
from ..base import SMPLBase, SMPLOutputs

class SMPL(SMPLBase):
    """
    A concrete implementation of the original SMPL model.
    """
    def __init__(self, model_path: str, gender: str = 'neutral', num_betas: int = 10, **kwargs):
        self.num_betas = num_betas
        self.gender = gender

        super().__init__(**kwargs)

        smpl_data = np.load(model_path, allow_pickle=True)

        self._load_model_specific_data(smpl_data)
        self._load_model_base_data(smpl_data, model_config='smpl.json')

    def _load_model_specific_data(self, model_data: dict):
        """
        Loads the SMPL model data from a .pkl file.
        """

        pass

    def classic_input(
        self,
        body_pose: torch.Tensor,
        translation: torch.Tensor,
        global_orient: torch.Tensor,
    ):

        shape = translation.shape[:-1]

        q = torch.zeros((*shape, self.nq), device=self.vertices_template.device)

        q[..., :3] = translation
        q[..., 3:7] = pp.so3(global_orient).Exp().tensor()
        q[..., 7:] = pp.so3(body_pose.reshape(*shape, self.num_joints-1, 3)).Exp().tensor().reshape(*shape, (self.num_joints-1)*4)

        return q