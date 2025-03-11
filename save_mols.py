import os
from tap import Tap
from chemprop.data import get_smiles
from chemprop.utils import makedirs
from chemprop.polymer import make_polymer_mol
from rdkit import Chem

class Args(Tap):
    data_path: str  # Path to data CSV
    smiles_column: str = None  # Name of the column containing SMILES strings
    save_dir: str  # Path to save .mol files

def save_mols_to_folder(mols, folder_path):
    """
    Saves each mol object as an individual .mol file in the specified folder.
    
    :param mols: List of RDKit Mol objects
    :param folder_path: Path to the directory where .mol files will be saved
    """
    os.makedirs(folder_path, exist_ok=True)

    for i, mol in enumerate(mols):
        mol_file = os.path.join(folder_path, f"mol_{i}.mol")
        with Chem.SDWriter(mol_file) as writer:
            writer.write(mol)

    print(f"Saved {len(mols)} molecules to {folder_path}")

def save_mol_files(args: Args):
    """
    Converts SMILES strings from a dataset into RDKit Mol objects and saves them as individual .mol files.
    """
    makedirs(args.save_dir)

    smiles = get_smiles(path=args.data_path, smiles_columns=args.smiles_column, flatten=True)
    print("Number of molecules:", len(smiles))

    # 生成 mol 对象
    mols = [make_polymer_mol(smi, keep_h=True, add_h=True) for smi in smiles]

    # 保存 mol 对象到单独的 .mol 文件
    mol_folder = os.path.join(args.save_dir, "mol_files")
    save_mols_to_folder(mols, mol_folder)

if __name__ == '__main__':
    save_mol_files(Args().parse_args())
