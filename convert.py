import numpy as np
import argparse
import pickle


def process_smpl(file_path):
    data = pickle.load(open(file_path, 'rb'), encoding='latin1')

    new_data = {}
    new_data["faces"] = np.array(data['f'])
    new_data["vertice_template"] = np.array(data['v_template'])
    new_data["shape_blending"] = np.array(data['shapedirs'])
    new_data["pose_blending"] = np.array(data['posedirs'])
    new_data["joint_template"] = np.array(data['J'])
    new_data["joint_regressor"] = np.array(data['J_regressor'].toarray())
    new_data["kintree_table"] = np.array(data['kintree_table'])
    new_data["weights"] = np.array(data['weights'])
    new_data["blending_skinning_type"] = str(data["bs_type"])
    new_data["blending_skinning_style"] = str(data["bs_style"])

    return new_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Type of the model')
    parser.add_argument('--file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--save', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()

    if str(args.model).lower() == 'smpl':
        new_data = process_smpl(args.file)
        print("Processed SMPL")
        np.savez(args.save, **new_data)
        print(f"Saved to {args.save}")
    else:
        raise ValueError(f"Model {args.model} not supported")

if __name__ == '__main__':
    main()
