from rdkit import Chem
sdfs = Chem.SDMolSupplier('../outputs/hello_charge/3cl-min_new_e4.sdf')
for i, mol in enumerate(sdfs):
    # get the summation of predicted charge in a molecule #计算分子中预测的电荷总和。这个过程通常涉及以下几个步骤：你这完全就是用rdkit的方法获取电荷，用途何在
    print(float(mol.GetProp('charge')))
    for atom in mol.GetAtoms():
        # get the predicted charge of each atom
        print(float(atom.GetProp('molFileAlias')))
