from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Iterator

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

import pickle
from typing import List
import os

class EnumMapping(StrEnum):
    @classmethod
    def get(cls, name: str | EnumMapping) -> EnumMapping:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise KeyError(
                f"Unsupported {cls.__name__} member! got: '{name}'. expected one of: {cls.keys()}"
            )

    @classmethod
    def keys(cls) -> Iterator[str]:
        return (e.name for e in cls)

    @classmethod
    def values(cls) -> Iterator[str]:
        return (e.value for e in cls)

    @classmethod
    def items(cls) -> Iterator[tuple[str, str]]:
        return zip(cls.keys(), cls.values())


def make_mol(smi: str, keep_h: bool, add_h: bool) -> Chem.Mol:
    """build an RDKit molecule from a SMILES string.

    Parameters
    ----------
    smi : str
        a SMILES string.
    keep_h : bool
        whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
    add_h : bool
        whether to add hydrogens to the molecule

    Returns
    -------
    Chem.Mol
        the RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
        )
    else:
        mol = Chem.MolFromSmiles(smi)

    if mol is None:
        raise RuntimeError(f"SMILES {smi} is invalid! (RDKit returned None)")

    if add_h:
        mol = Chem.AddHs(mol)

    return mol

def make_polymer_mol(smiles: str, keep_h: bool, add_h: bool, coord=True, version=2, ez='E', chiral='S') -> Chem.Mol:
    '''
    Generate 3D coordinates of a molecule from SMILES string using RDKit.
    Parameters
    ----------
    smiles : str
        SMILES string of a molecule.
    keep_h : bool
        Whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    add_h : bool
        Whether to add hydrogens to the molecule.
    coord : bool
        Whether to generate 3D coordinates.
    version : int
        Version of ETKDG algorithm. 2 or 3.
    ez : str
        Configuration of unspecified double bonds. 'E' or 'Z'.
    chiral : str
        Configuration of unspecified chirality. 'S' or 'R'.
    Returns
    '''
    n_conn = smiles.count('[*]') + smiles.count('*') + smiles.count('[3H]')
    smi = smiles.replace('[*]', '[3H]')
    smi = smi.replace('*', '[3H]')

    if version == 3:
        etkdg = AllChem.ETKDGv3()
    elif version == 2:
        etkdg = AllChem.ETKDGv2()
    else:
        etkdg = AllChem.ETKDG()
    etkdg.enforceChirality = True
    etkdg.useRandomCoords = False
    etkdg.maxAttempts = 100

    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
    except Exception as e:
        print('Cannot transform to RDKit Mol object from %s' % smiles)
        return None

    ### cis/trans and chirality control
    Chem.AssignStereochemistry(mol)

    # Get polymer backbone
    backbone_atoms = []
    backbone_bonds = []
    backbone_dih = []
    if n_conn == 2:
        link_idx = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "H" and atom.GetIsotope() == 3:
                link_idx.append(atom.GetIdx())
        backbone_atoms = Chem.GetShortestPath(mol, link_idx[0], link_idx[1])

        for i in range(len(backbone_atoms) - 1):
            bond = mol.GetBondBetweenAtoms(backbone_atoms[i], backbone_atoms[i + 1])
            backbone_bonds.append(bond.GetIdx())
            if bond.GetBondTypeAsDouble() == 2 and str(bond.GetStereo()) == 'STEREONONE' and not bond.IsInRing():
                backbone_dih.append(
                    (backbone_atoms[i - 1], backbone_atoms[i], backbone_atoms[i + 1], backbone_atoms[i + 2]))

    # List of unspecified double bonds (except for bonds in polymer backbone and a ring structure)
    db_list = []
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() == 2 and str(bond.GetStereo()) == 'STEREONONE' and not bond.IsInRing():
            if n_conn == 2 and bond.GetIdx() in backbone_bonds:
                continue
            else:
                db_list.append(bond.GetIdx())

    # Enumerate stereo isomers
    opts = Chem.EnumerateStereoisomers.StereoEnumerationOptions(unique=True, tryEmbedding=True)
    isomers = tuple(Chem.EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts))

    if len(isomers) > 1:
        print('Warning: '+'%i candidates of stereoisomers were generated.' % len(isomers))
        chiral_num_max = 0

        for isomer in isomers:
            ez_flag = False
            chiral_flag = 0

            Chem.AssignStereochemistry(isomer)

            # Contorol unspecified double bonds (except for bonds in polymer backbone and a ring structure)
            ez_list = []
            for idx in db_list:
                bond = isomer.GetBondWithIdx(idx)
                if str(bond.GetStereo()) == 'STEREOANY' or str(bond.GetStereo()) == 'STEREONONE':
                    continue
                elif ez == 'E' and (str(bond.GetStereo()) == 'STEREOE' or str(bond.GetStereo()) == 'STEREOTRANS'):
                    ez_list.append(True)
                elif ez == 'Z' and (str(bond.GetStereo()) == 'STEREOZ' or str(bond.GetStereo()) == 'STEREOCIS'):
                    ez_list.append(True)
                else:
                    ez_list.append(False)

            if len(ez_list) > 0:
                ez_flag = np.all(np.array(ez_list))
            else:
                ez_flag = True

            # Contorol unspecified chirality
            chiral_list = np.array(Chem.FindMolChiralCenters(isomer))
            if len(chiral_list) > 0:
                chiral_centers = chiral_list[:, 0]

                chirality = chiral_list[:, 1]
                chiral_num = np.count_nonzero(chirality == chiral)
                if chiral_num == len(chiral_list):
                    chiral_num_max = chiral_num
                    chiral_flag = 2
                elif chiral_num > chiral_num_max:
                    chiral_num_max = chiral_num
                    chiral_flag = 1
            else:
                chiral_flag = 2

            if ez_flag and chiral_flag:
                mol = isomer
                if chiral_flag == 2:
                    break

    # Generate 3D coordinates
    if coord:
        try:
            enbed_res = AllChem.EmbedMolecule(mol, etkdg)
        except Exception as e:
            print('Cannot generate 3D coordinate of %s' % smiles)
            return None
        if enbed_res == -1:
            etkdg.useRandomCoords = True
            enbed_res = AllChem.EmbedMolecule(mol, etkdg)
            if enbed_res == -1:
                print('Cannot generate 3D coordinate of %s' % smiles)
                return None

    # Dihedral angles of unspecified double bonds in a polymer backbone are modified to 180 degree.
    if len(backbone_dih) > 0:
        for dih_idx in backbone_dih:
            Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), dih_idx[0], dih_idx[1], dih_idx[2], dih_idx[3],
                                                180.0)

            for na in mol.GetAtomWithIdx(dih_idx[2]).GetNeighbors():
                na_idx = na.GetIdx()
                if na_idx != dih_idx[1] and na_idx != dih_idx[3]:
                    break
            Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), dih_idx[0], dih_idx[1], dih_idx[2], na_idx, 0.0)

    mol = Chem.RemoveAllHs(mol)
    return Chem.AddHs(mol) if add_h else mol

def pretty_shape(shape: Iterable[int]) -> str:
    """Make a pretty string from an input shape

    Example
    --------
    >>> X = np.random.rand(10, 4)
    >>> X.shape
    (10, 4)
    >>> pretty_shape(X.shape)
    '10 x 4'
    """
    return " x ".join(map(str, shape))


def save_mols_to_folder(mols, folder_path):
    """Saves each RDKit mol object as a .mol file in the specified folder."""
    os.makedirs(folder_path, exist_ok=True)
    for i, mol in enumerate(mols):
        file_path = os.path.join(folder_path, f"molecule_{i}.mol")
        Chem.MolToMolFile(mol, file_path)

def load_mols_from_folder(folder_path):
    """Loads RDKit mol objects from .mol files in the specified folder."""
    mol_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".mol")])
    mols = [Chem.MolFromMolFile(os.path.join(folder_path, f)) for f in mol_files]
    return [mol for mol in mols if mol is not None]  # Filter out failed loads
