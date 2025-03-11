import os
import time
import numpy as np
from rdkit import Chem
from tap import Tap

from chemprop.atomic_matrices import mol2matrix
from chemprop.features import save_features
from chemprop.utils import makedirs

class Args(Tap):
    mol_dir: str  # Path to directory containing .mol files
    save_dir: str  # Path to save feature matrices
    adjacency: bool = False  # Generate adjacency matrix
    coulomb: bool = False  # Generate coulomb matrix
    distance: bool = False  # Generate distance matrix

def load_mol(mol_path):
    """Load an individual .mol file and return RDKit Mol object."""
    mol = Chem.MolFromMolFile(mol_path, removeHs=False)
    return mol

def process_molecules(args: Args):
    """
    Reads .mol files from a directory, computes feature matrices, and saves them in .npz format.
    """
    makedirs(args.save_dir, isfile=True)

    mol_files = sorted([os.path.join(args.mol_dir, f) for f in os.listdir(args.mol_dir) if f.endswith(".mol")])
    print("Number of molecules:", len(mol_files))

    if not (args.coulomb or args.distance or args.adjacency):
        print("Please specify at least one feature type: (adjacency, distance, coulomb)")
        return

    t = time.time()

    matrices = {"coulomb": [], "distance": [], "adjacency": []}
    max_size = 0  # Track the largest molecule

    # 读取mol文件
    for mol_path in mol_files:
        mol = load_mol(mol_path)
        if mol is None:
            print(f"Warning: Could not read {mol_path}, skipping.")
            continue

        # 更新最大分子大小
        num_atoms = mol.GetNumAtoms()
        max_size = max(max_size, num_atoms)

        # 计算特征矩阵
        graph = mol2matrix([mol], args)

        if args.coulomb:
            matrices["coulomb"].append(graph.get_coulomb()[1])
        if args.distance:
            matrices["distance"].append(graph.get_distance()[1])
        if args.adjacency:
            matrices["adjacency"].append(graph.get_adjacency()[1])

    # padding
    for key in ["coulomb", "distance", "adjacency"]:
        if key in matrices and matrices[key]:
            padded_matrices = [
                np.pad(matrix, ((0, max_size - matrix.shape[0]), (0, max_size - matrix.shape[1])), mode='constant')
                for matrix in matrices[key]
            ]
            save_features(os.path.join(args.save_dir, f"{key}.npz"), padded_matrices)
            print(f"{key.capitalize()} matrices saved.")

    print("Processing time:", time.time() - t)

if __name__ == '__main__':
    process_molecules(Args().parse_args())
