import numpy as np


def load_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    for key in data:
        print(f"{key}: {data[key]}")


load_npz("models/smpl/SMPL_NEUTRAL.npz")