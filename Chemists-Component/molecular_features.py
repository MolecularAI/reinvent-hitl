import numpy as np
from rdkit.Chem import DataStructs,Descriptors
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools

#morgan fingerprints
def morganX(mol, bits=1024, radius=3):
    morgan = np.zeros((1, bits))
    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    DataStructs.ConvertToNumpyArray(fp, morgan)
    return morgan

#MACCS
def maccsX(mol):
    maccs = np.zeros((1,167))
    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, maccs)
    return maccs

#Avalon fps
def avalonX(mol, avbits=512):
    avalon = np.zeros((1, avbits))
    fp = pyAvalonTools.GetAvalonFP(mol)
    DataStructs.ConvertToNumpyArray(fp, avalon)
    return avalon

# physchem descriptors
def descriptorsX(m):
    descr = [Descriptors.ExactMolWt(m),
             Descriptors.MolLogP(m),
             Descriptors.TPSA(m),
             Descriptors.NumHAcceptors(m),
             Descriptors.NumHDonors(m),
             Descriptors.NumRotatableBonds(m),
             Descriptors.NumHeteroatoms(m),
             Descriptors.NumAromaticRings(m),
             Descriptors.FractionCSP3(m)]
    return np.array(descr)