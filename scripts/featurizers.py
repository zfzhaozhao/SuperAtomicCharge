# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Node and edge featurization for molecular graphs.
# pylint: disable= no-member, arguments-differ, invalid-name
#提取特征的代码
import itertools        #itertools 是 Python 的一个内置模块，提供了用于操作可迭代对象的函数。这个模块包含了许多有用的函数
                        #比如生成排列、组合、笛卡尔积等。它可以帮助你高效地处理迭代器和生成器。
import os.path as osp   #os 是 Python 的一个标准库，提供了与操作系统交互的功能。os.path 是 os 模块的一个子模块，专门用于处理文件和目录的路径。

from collections import defaultdict
                                   #collections 是 Python 的一个内置模块，提供了许多有用的集合类（如 Counter, deque, OrderedDict 等）。
                                   #defaultdict 是 collections 模块中的一个类，扩展了内置的字典（dict）类型。其主要特点是：
                                   #当你访问一个不存在的键时，defaultdict 会自动为该键创建一个默认值，而不是抛出 KeyError。
                                   #你可以在创建 defaultdict 时指定一个工厂函数，这个函数会返回默认值。例如，如果使用 defaultdict(int)，那么访问不存在的键时会返回 0，因为 int() 的返回值是 0。
from functools import partial
                                 #functools 是 Python 的一个内置模块，提供了一些高阶函数，用于操作或返回其他函数。
                                 #partial 是 functools 模块中的一个函数，它允许你固定函数的一些参数，从而生成一个新的函数。这个新函数可以在调用时忽略那些被固定的参数，简化函数调用。
import numpy as np
import torch
import dgl.backend as F  #backend 是 DGL 中的一个模块，通常用于提供与底层张量操作相关的功能，例如创建张量、进行数学运算等。
                         #它的设计目的是为用户提供一个与后端深度学习框架（如 PyTorch 或 TensorFlow）无关的接口。

try:
    from rdkit import Chem, RDConfig                    #Chem 是 rdkit 中的一个模块，包含处理分子的各种功能，比如分子的读取、写入、转换等。
    from rdkit.Chem import AllChem, ChemicalFeatures     #RDConfig 是 rdkit 提供的一个模块，用于获取 RDKit 安装路径等配置信息
except ImportError:                                        #AllChem 包含了额外的化学计算功能，比如构建分子、生成三维结构等
    pass                                                  #ChemicalFeatures 用于从分子中提取化学特征。

#_all__ 是 Python 中一个特殊的变量，主要用于模块的导入控制。当你在一个模块中定义 __all__ 变量时，它通常是一个包含字符串的列表，列出了该模块希望公开的属性和方法。

__all__ = ['one_hot_encoding',
           'atom_type_one_hot',
           'atomic_number_one_hot',
           'atomic_number',
           'atom_degree_one_hot',
           'atom_degree',
           'atom_total_degree_one_hot',
           'atom_total_degree',
           'atom_explicit_valence_one_hot',
           'atom_explicit_valence',
           'atom_implicit_valence_one_hot',
           'atom_implicit_valence',
           'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot',
           'atom_total_num_H',
           'atom_formal_charge_one_hot',
           'atom_formal_charge',
           'atom_num_radical_electrons_one_hot',
           'atom_num_radical_electrons',
           'atom_is_aromatic_one_hot',
           'atom_is_aromatic',
           'atom_is_in_ring_one_hot',
           'atom_is_in_ring',
           'atom_chiral_tag_one_hot',
           'atom_chirality_type_one_hot',
           'atom_mass',
           'atom_is_chiral_center',
           'ConcatFeaturizer',
           'BaseAtomFeaturizer',
           'CanonicalAtomFeaturizer',
           'WeaveAtomFeaturizer',
           'PretrainAtomFeaturizer',
           'AttentiveFPAtomFeaturizer',
           'PAGTNAtomFeaturizer',
           'bond_type_one_hot',
           'bond_is_conjugated_one_hot',
           'bond_is_conjugated',
           'bond_is_in_ring_one_hot',
           'bond_is_in_ring',
           'bond_stereo_one_hot',
           'bond_direction_one_hot',
           'BaseBondFeaturizer',
           'CanonicalBondFeaturizer',
           'WeaveEdgeFeaturizer',
           'PretrainBondFeaturizer',
           'AttentiveFPBondFeaturizer',
           'PAGTNEdgeFeaturizer']

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.

    Parameters
    ----------
    x
        Value to encode.  x是需要转为独热编码的具体数值
    allowable_set : list   容纳某个特征的所有数值，用于给独热编码确定每个值的位置，这也表名，不同的特征独热编码的长度不一样哦 
        The elements of the allowable_set should be of the
        same type as x.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element.用于控制输入不再allowable中是是否用设置为额外的最后一位元素

    Returns
    -------
    list
        List of boolean values where at most one value is True.  （返回的是布尔数的列表，只有其对应值的位置为True）
        The list is of length ``len(allowable_set)`` if ``encode_unknown=False``
        and ``len(allowable_set) + 1`` otherwise.

    Examples
    --------
    >>> from dgllife.utils import one_hot_encoding
    >>> one_hot_encoding('C', ['C', 'O'])
    [True, False]
    >>> one_hot_encoding('S', ['C', 'O'])
    [False, False]
    >>> one_hot_encoding('S', ['C', 'O'], encode_unknown=True)
    [False, False, True]
    """
  #这两个if  用于处理 encode_unkonwn 是True的情况，先给allowable_set 的最后一位添加None ,如果x 不属于里面的值，就让x=None 作为最后一位处理
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))  #返回的是bool索引的列表
#lambda s: x == s: 这是一个匿名函数（即 lambda 函数），它接受一个参数 s，并返回一个布尔值（True 或 False）。
#这个函数的作用是判断变量 x 是否等于 s。
#map(...): map 函数会对可迭代对象中的每个元素应用指定的函数（在这里是 lambda 函数），并返回一个迭代器。
#也就是说，它会遍历 allowable_set 中的每个元素 s，并将 lambda 函数应用于每一个 s。
#################################################################
# Atom featurization  原子特征（是c H O S P 这种元素判断）
#################################################################

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Atom types to consider. Default: ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``,
        ``Cl``, ``Br``, ``Mg``, ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``,
        ``K``, ``Tl``, ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
        ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``, ``Cr``,
        ``Pt``, ``Hg``, ``Pb``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atomic_number
    atomic_number_one_hot
    """
  #allowable_set 是None的话，使用内置set (就是默认的）不然就自己给set
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)  #获取原子类型
#atom.GetSymbol() 方法是用于获取特定原子的化学符号
#独热编码是43维哦（False） Ture 44维
def atomic_number_one_hot(atom, allowable_set=None, encode_unknown=False):  #用与编码元素的周期号，有了元素符号，还要这个周期号吗？？ 感觉有点多余
    """One hot encoding for the atomic number of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atomic numbers to consider. Default: ``1`` - ``100``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atomic_number
    atom_type_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(1, 101))
    return one_hot_encoding(atom.GetAtomicNum(), allowable_set, encode_unknown) #atom.GetAtomicNum() 获取原子序号
#false  100维 true 101 维 atom.GetAtomicNum() 方法用于获取特定原子的原子序数
def atomic_number(atom):
    """Get the atomic number for an atom.  
#获取原子序数
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
       List containing one int only.

    See Also
    --------
    atomic_number_one_hot
    atom_type_one_hot
    """
    return [atom.GetAtomicNum()]

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.
#请注意，结果会因氢原子（Hs）是否在图中显式建模而有所不同。
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom degrees to consider. Default: ``0`` - ``10``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_total_degree
    atom_total_degree_one_hot
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)
#atom.GetDegree() GetDegree(): 这是一个方法，用于返回与该原子直接相连的其他原子的数量。这个数量被称为原子的度，它反映了原子在分子中的连接情况。
#维度 False 11维 True 12维
def atom_degree(atom):
    """Get the degree of an atom.

    Note that the result will be different depending on whether the Hs are
    explicitly modeled in the graph.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_degree_one_hot
    atom_total_degree
    atom_total_degree_one_hot
    """
    return [atom.GetDegree()]

def atom_total_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom including Hs.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list
        Total degrees to consider. Default: ``0`` - ``5``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    See Also
    --------
    one_hot_encoding
    atom_degree
    atom_degree_one_hot
    atom_total_degree
    """
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)  
#GetDegree()：只计算显性连接的原子（如碳、氧等）。
#GetTotalDegree()：考虑了与原子相连的所有原子，包括隐式氢。
#维度 False 6 True 7
def atom_total_degree(atom):
    """The degree of an atom including Hs.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_degree_one_hot
    atom_degree
    atom_degree_one_hot
    """
    return [atom.GetTotalDegree()]

def atom_explicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the explicit valence of an aotm.
#显式价态 (Explicit Valence): 显式价态指的是在分子结构中直接表示的与其他原子的连接。
#它考虑了原子周围的所有化学键，并且通常只计算那些在分子模型中明确呈现的键。
#这个和度的差异是什么？？？？
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom explicit valences to consider. Default: ``1`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_explicit_valence
    """
    if allowable_set is None:
        allowable_set = list(range(1, 7))
    return one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)
#atom.GetExplicitValence() 方法用于获取一个原子的显式价电子数。价电子是原子最外层的电子，它们参与化学键的形成。显式价电子数是指通过化学键直接与其他原子相连的价电子的数量。

def atom_explicit_valence(atom):
    """Get the explicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_explicit_valence_one_hot
    """
    return [atom.GetExplicitValence()]

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valence of an atom.
#隐式价态 (Implicit Valence): 有时，分子中可能存在隐式的氢原子或其他原子，这些原子没有在分子图中显示，但它们仍然影响原子的总价态
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Atom implicit valences to consider. Default: ``0`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    atom_implicit_valence
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)
#atom.GetImplicitValence() 是 RDKit 中用于获取原子的隐式价电子数的方法。隐式价电子是指原子在分子中并没有通过化学键显式连接的那些价电子，通常是指孤对电子或未成对电子。

def atom_implicit_valence(atom):
    """Get the implicit valence of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Reurns
    ------
    list
        List containing one int only.

    See Also
    --------
    atom_implicit_valence_one_hot
    """
    return [atom.GetImplicitValence()]

# pylint: disable=I1101
def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.
#化（Hybridization）**是化学中的一个概念，用于描述原子中电子轨道的混合，以形成新的、等效的杂化轨道。
#这一概念通常用于解释分子的几何结构和化学键的性质。
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,           #SP：线性杂化           乙炔（C₂H₂          
                         Chem.rdchem.HybridizationType.SP2,          #SP2：平面三角形杂化    乙烯（C₂H₄）               
                         Chem.rdchem.HybridizationType.SP3,          #SP3：四面体杂化        甲烷（CH₄）
                         Chem.rdchem.HybridizationType.SP3D,         #SP3D：五面体杂化       磷酸根离子（PO₄³⁻）
                         Chem.rdchem.HybridizationType.SP3D2]        #SP3D2：八面体杂化      六氟化硫（SF₆）
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)
   


def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Total number of Hs to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_total_num_H
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

def atom_total_num_H(atom):
    """Get the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_total_num_H_one_hot
    """
    return [atom.GetTotalNumHs()]

def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the formal charge of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Formal charges to consider. Default: ``-2`` - ``2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_formal_charge
    """
    if allowable_set is None:
        allowable_set = list(range(-2, 3))
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)
#atom.GetFormalCharge() 是 RDKit 中用于获取原子的形式电荷（formal charge）的方法。形式电荷是指一个原子在分子中相对于其基态时所带的电荷，通常用于描述分子中的电荷分布。
def atom_formal_charge(atom):
    """Get formal charge for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_formal_charge_one_hot
    """
    return [atom.GetFormalCharge()]

def atom_partial_charge(atom):
    """Get Gasteiger partial charge for an atom.

    For using this function, you must have called ``AllChem.ComputeGasteigerCharges(mol)``
    to compute Gasteiger charges.

    Occasionally, we can get nan or infinity Gasteiger charges, in which case we will set
    the result to be 0.
#`AllChem.ComputeGasteigerCharges(mol)` 来计算分子的 Gasteiger 电荷。Gasteiger 电荷是一种用于描述分子中原子电荷分布的量，常用于药物设计和分子模拟。
#以下是一些关键点： 1. **计算 Gasteiger 电荷**：在调用需要 Gasteiger 电荷的函数之前，必须先计算这些电荷。可以通过 RDKit 库中的 `AllChem.ComputeGasteigerCharges(mol)` 方法来完成。
#2. **处理异常值**：在计算过程中，有时可能会得到 NaN（不是一个数字）或无穷大（infinity）的电荷值。如果出现这种情况，系统会将这些值设置为 0，以避免在后续计算中造成问题。
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one float only.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return [float(gasteiger_charge)]
#atom.GetProp('_GasteigerCharge') 用于从原子对象中提取 Gasteiger 电荷。这是一种计算化学电荷的方法，通常用于量化分子中原子的电荷分布。

def atom_num_radical_electrons_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the number of radical electrons of an atom.
#Radical electrons” 通常指的是与自由基（radicals）相关的电子。在化学中，自由基是指具有一个或多个未配对电子的分子或原子
#总之，radical electrons 是指自由基中的未配对电子，这些电子使得自由基具有高度的反应性和重要的化学性质。
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Number of radical electrons to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_num_radical_electrons
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetNumRadicalElectrons(), allowable_set, encode_unknown)
#atom.GetNumRadicalElectrons()用于获取原子的自由基电子数的方法
def atom_num_radical_electrons(atom):
    """Get the number of radical electrons for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.

    See Also
    --------
    atom_num_radical_electrons_one_hot
    """
    return [atom.GetNumRadicalElectrons()]

def atom_is_aromatic_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is aromatic.
    #判断原子是否是芳香性原子

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_is_aromatic
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.GetIsAromatic(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    atom_is_aromatic_one_hot
    """
    return [atom.GetIsAromatic()]

def atom_is_in_ring_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is in ring.
#判断原子是个在环上
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    atom_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.IsInRing(), allowable_set, encode_unknown)

def atom_is_in_ring(atom):
    """Get whether the atom is in ring.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    atom_is_in_ring_one_hot
    """
    return [atom.IsInRing()]

def atom_chiral_tag_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chiral tag of an atom.
#“Chiral tag” 指的是在化学中用于标识和分离手性分子的一种标记或标签。手性（chirality）是指分子或原子由于其结构的不同而存在两种不可重叠的立体异构体，通常被称为对映体（enantiomers）
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.ChiralType
        Chiral tags to consider. Default: ``rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_OTHER``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    one_hot_encoding
    atom_chirality_type_one_hot
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,                           #未指定的手性
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,                        #四面体手性，顺时针
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,                       #四面体手性，逆时针
                         Chem.rdchem.ChiralType.CHI_OTHER]                                 #其他手性类型
    return one_hot_encoding(atom.GetChiralTag(), allowable_set, encode_unknown)

def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chirality type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Chirality types to consider. Default: ``R``, ``S``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    one_hot_encoding
    atom_chiral_tag_one_hot  #处理原子的 CIP（Cahn-Ingold-Prelog）手性编码
    """
    if not atom.HasProp('_CIPCode'): #检查原子是否具有 _CIPCode 属性
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)
#atom.GetProp('_CIPCode') 方法用于获取原子的 CIP 手性编码

def atom_mass(atom, coef=0.01):
    """Get the mass of an atom and scale it.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    coef : float
        The mass will be multiplied by ``coef``.#类似矫正因子的作用

    Returns
    -------
    list
        List containing one float only.
    """
    return [atom.GetMass() * coef]

def atom_is_chiral_center(atom):
    """Get whether the atom is chiral center  #原子是否是手性中心

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [atom.HasProp('_ChiralityPossible')]
    #HasProp 是 RDKit 中的方法，用于检查一个原子是否拥有特定的属性。
    #在这里，'_ChiralityPossible' 是一个特定的属性名，表示原子是否可能是手性中心  #所以输出是啥？？？

class ConcatFeaturizer(object):
    """
    Concatenate the evaluation results of multiple functions as a single feature.
    #将多个函数的评估结果连接为单一特征。

    Parameters
    ----------
    func_list : list #（这个列表参数 ，里面就是提取原子特征的各种函数)
        List of functions for computing molecular descriptors from objects of a same
        particular data type, e.g. ``rdkit.Chem.rdchem.Atom``. Each function is of signature
        ``func(data_type) -> list of float or bool or int``. The resulting order of
        the features will follow that of the functions in the list.
  #用于从相同特定数据类型的对象（例如 ``rdkit.Chem.rdchem.Atom``）计算分子描述符的函数列表。
  #每个函数的签名为 ``func(data_type) -> list of float or bool or int``。生成的特征顺序将遵循列表中函数的顺序。
  """

    Examples
    --------

    Setup for demo.

    >>> from dgllife.utils import ConcatFeaturizer
    >>> from rdkit import Chem
    >>> smi = 'CCO'
    >>> mol = Chem.MolFromSmiles(smi)

  #  Concatenate multiple atom descriptors as a single node feature.#（就是将多个特征作为一个维度)

    >>> from dgllife.utils import atom_degree, atomic_number, BaseAtomFeaturizer
    >>> # Construct a featurizer for featurizing one atom a time
    >>> atom_concat_featurizer = ConcatFeaturizer([atom_degree, atomic_number])
    >>> # Construct a featurizer for featurizing all atoms in a molecule
    >>> mol_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})
    >>> mol_atom_featurizer(mol)
    {'h': tensor([[1., 6.],#(列表示的是原子特征，行表示的是各个原子）
                  [2., 6.],
                  [1., 8.]])}

 #   Conctenate multiple bond descriptors as a single edge feature.

    >>> from dgllife.utils import bond_type_one_hot, bond_is_in_ring, BaseBondFeaturizer
    >>> # Construct a featurizer for featurizing one bond a time
    >>> bond_concat_featurizer = ConcatFeaturizer([bond_type_one_hot, bond_is_in_ring])
    >>> # Construct a featurizer for featurizing all bonds in a molecule
    >>> mol_bond_featurizer = BaseBondFeaturizer({'h': bond_concat_featurizer})
    >>> mol_bond_featurizer(mol)
    {'h': tensor([[1., 0., 0., 0., 0.],#（这里的列是特征，但是是one-hot的形式哦）
                  [1., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0.]])} """

                  
    def __init__(self, func_list):
        self.func_list = func_list

    def __call__(self, x):  #（x是要特征化的数据，就是小分子吧）
        """Featurize the input data.

        Parameters
        ----------
        x :
            Data to featurize.

        Returns
        -------
        list
            List of feature values, which can be of type bool, float or int.
        """
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers. # 用于特征提取的类                         

    Loop over all atoms in a molecule and featurize them with the ``featurizer_funcs``.

    **We assume the resulting DGLGraph will not contain any virtual nodes and a node i in the
    graph corresponds to exactly atom i in the molecule.**

    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Atom) -> list or 1D numpy array``.
    feat_sizes : dict  #就是每个特征的实际维度
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.

    Examples
    --------

    >>> from dgllife.utils import BaseAtomFeaturizer, atom_mass, atom_degree_one_hot
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = BaseAtomFeaturizer({'mass': atom_mass, 'degree': atom_degree_one_hot})
    >>> atom_featurizer(mol)
    {'mass': tensor([[0.1201],
                     [0.1201],
                     [0.1600]]),
     'degree': tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size for atom mass
    >>> print(atom_featurizer.feat_size('mass'))
    1
    >>> # Get feature size for atom degree
    >>> print(atom_featurizer.feat_size('degree'))
    11

    See Also
    --------
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """


    
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))
#Chem.MolFromSmiles('C') 创建了一个表示单个碳原子的分子对象。
#.GetAtomWithIdx(0) 获取该分子中索引为 0 的原子（即这个单一的碳原子）。
        return self._feat_sizes[feat_name]

    def __call__(self, mol):
    #，__call__ 方法通常用于特征提取器或处理器类。当你调用这个实例并传入一个分子对象时
    #__call__ 方法会被执行，通常会返回某种计算结果，比如特征列表、描述符等
        """Featurize all atoms in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_atoms = mol.GetNumAtoms()  #总原子数
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)   #（将每个feat_name 下的每个原子的特征按行堆起来）
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))  
#F.zerocopy_from_numpy(...)：
#这是 DGL 中的一个函数，通常位于 dgl.backend 模块。这个函数的作用是创建一个与 NumPy 数组共享内存的 DGL 张量，而不是复制数据。
#这意味着它不会占用额外的内存，提高了效率，特别是在处理大型数据集时。
        return processed_features

class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
    """A default featurizer for atoms.

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 10``.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include ``0 - 6``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import CanonicalAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                      1., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
                      0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
                      0., 0.]])}

    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    74

    See Also
    --------
    BaseAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        super(CanonicalAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atom_type_one_hot,
                 atom_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_total_num_H_one_hot]
            )})

class WeaveAtomFeaturizer(object):
    """Atom featurizer in Weave.

    The atom featurization performed in `Molecular Graph Convolutions: Moving Beyond Fingerprints
    <https://arxiv.org/abs/1603.00856>`__, which considers:

    * atom types
    * chirality
    * formal charge
    * partial charge
    * aromatic atom
    * hybridization
    * hydrogen bond donor
    * hydrogen bond acceptor
    * the number of rings the atom belongs to for ring size between 3 and 8

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.
    atom_types : list of str or None
        Atom types to consider for one-hot encoding. If None, we will use a default
        choice of ``'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'``.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice of ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``.
    hybridization_types : list of Chem.rdchem.HybridizationType or None
        Atom hybridization types to consider for one-hot encoding. If None, we will use a
        default choice of ``Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3``.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import WeaveAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = WeaveAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0418,  0.0000,
                       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000],
                     [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0402,  0.0000,
                       0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000],
                     [ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.3967,  0.0000,
                       0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  0.0000,  0.0000,  0.0000,
                       0.0000,  0.0000,  0.0000]])}
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size())
    27

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h', atom_types=None, chiral_types=None,
                 hybridization_types=None):
        super(WeaveAtomFeaturizer, self).__init__()

        self._atom_data_field = atom_data_field

        if atom_types is None:
            atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        self._atom_types = atom_types

        if chiral_types is None:
            chiral_types = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
        self._chiral_types = chiral_types

        if hybridization_types is None:
            hybridization_types = [Chem.rdchem.HybridizationType.SP,
                                   Chem.rdchem.HybridizationType.SP2,
                                   Chem.rdchem.HybridizationType.SP3]
        self._hybridization_types = hybridization_types

        self._featurizer = ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=atom_types, encode_unknown=True),
            partial(atom_chiral_tag_one_hot, allowable_set=chiral_types),
            atom_formal_charge, atom_partial_charge, atom_is_aromatic,
            partial(atom_hybridization_one_hot, allowable_set=hybridization_types)
        ])

        fdef_name = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self._mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._atom_data_field]

        return feats.shape[-1]

#self(mol)：这部分调用了当前对象的 __call__ 方法（如果该对象实现了 __call__）。它将 mol 作为参数传入，通常用于提取该分子的特征。返回的结果可能是一个字典或列表，包含了不同的特征。
#[self._atom_data_field]：这部分从提取的特征中获取与 _atom_data_field 相关的特征数据。_atom_data_field 通常是一个字符串，表示特定的特征键（例如，原子的属性或特征），用于索引提取的特征。
    def get_donor_acceptor_info(self, mol_feats):
        """Bookkeep whether an atom is donor/acceptor for hydrogen bonds.

        Parameters
        ----------
        mol_feats : tuple of rdkit.Chem.rdMolChemicalFeatures.MolChemicalFeature
            Features for molecules.

        Returns
        -------
        is_donor : dict
            Mapping atom ids to binary values indicating whether atoms
            are donors for hydrogen bonds
        is_acceptor : dict
            Mapping atom ids to binary values indicating whether atoms
            are acceptors for hydrogen bonds
        """
        is_donor = defaultdict(bool)
        is_acceptor = defaultdict(bool)
        # Get hydrogen bond donor/acceptor information
        for feats in mol_feats:
            if feats.GetFamily() == 'Donor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_donor[u] = True
            elif feats.GetFamily() == 'Acceptor':
                nodes = feats.GetAtomIds()
                for u in nodes:
                    is_acceptor[u] = True

        return is_donor, is_acceptor

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping atom_data_field as specified in the input argument to the atom
            features, which is a float32 tensor of shape (N, M), N is the number of
            atoms and M is the feature size.
        """
        atom_features = []

        AllChem.ComputeGasteigerCharges(mol)
        num_atoms = mol.GetNumAtoms()

        # Get information for donor and acceptor
        mol_feats = self._mol_featurizer.GetFeaturesForMol(mol)
        is_donor, is_acceptor = self.get_donor_acceptor_info(mol_feats)
#调用分子特征提取器（self._mol_featurizer）的方法 GetFeaturesForMol(mol)，用于从给定的分子对象 mol 中提取特征。提取的特征将存储在 mol_feats 变量中
#这行代码调用当前类的 get_donor_acceptor_info 方法，并将 mol_feats 作为参数传入。这个方法的作用是分析分子的特征，识别出哪些原子是氢键供体（donor）和哪些是氢键受体（acceptor）。
        # Get a symmetrized smallest set of smallest rings
        # Following the practice from Chainer Chemistry (https://github.com/chainer/
        # chainer-chemistry/blob/da2507b38f903a8ee333e487d422ba6dcec49b05/chainer_chemistry/
        # dataset/preprocessors/weavenet_preprocessor.py)
        sssr = Chem.GetSymmSSSR(mol)
#GetSymmSSSR(mol) 函数会返回一个列表，其中包含分子中所有的环结构，并确保这些环是对称的。它找到的环是分子中所有环的最小集合。
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            # Features that can be computed directly from RDKit atom instances, which is a list
            feats = self._featurizer(atom)
            # Donor/acceptor indicator
            feats.append(float(is_donor[i]))
            feats.append(float(is_acceptor[i]))
            # Count the number of rings the atom belongs to for ring size between 3 and 8
            count = [0 for _ in range(3, 9)]
        #这是一个列表推导式（list comprehension）。
#range(3, 9) 生成一个迭代器，产生的值依次是 3, 4, 5, 6, 7, 8。
#对于每个从 range 生成的值（即 _），都返回 0，最终构成一个列表。
            for ring in sssr:
                ring_size = len(ring)
                if i in ring and 3 <= ring_size <= 8:
                    count[ring_size - 3] += 1
            feats.extend(count)
            atom_features.append(feats)
        atom_features = np.stack(atom_features)

        return {self._atom_data_field: F.zerocopy_from_numpy(atom_features.astype(np.float32))}
#i in ring：检查原子 i 是否在当前环中。
#3 <= ring_size <= 8：确保环的大小在 3 到 8 之间。
#count[ring_size - 3] += 1：根据环的大小更新计数器。如果 ring_size 为 3，则更新 count[0]；如果为 4，则更新 count[1]；以此类推。

class PretrainAtomFeaturizer(object):
    """AtomFeaturizer in Strategies for Pre-training Graph Neural Networks.

    The atom featurization performed in `Strategies for Pre-training Graph Neural Networks
    <https://arxiv.org/abs/1905.12265>`__, which considers:

    * atomic number
    * chirality

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atomic_number_types : list of int or None
        Atomic number types to consider for one-hot encoding. If None, we will use a default
        choice of 1-118.
    chiral_types : list of Chem.rdchem.ChiralType or None
        Atom chirality to consider for one-hot encoding. If None, we will use a default
        choice, including ``Chem.rdchem.ChiralType.CHI_UNSPECIFIED``,
        ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW``,
        ``Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``, ``Chem.rdchem.ChiralType.CHI_OTHER``.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import PretrainAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = PretrainAtomFeaturizer()
    >>> atom_featurizer(mol)
    {'atomic_number': tensor([5, 5, 7]), 'chirality_type': tensor([0, 0, 0])}

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atomic_number_types=None, chiral_types=None):
        if atomic_number_types is None:
            atomic_number_types = list(range(1, 119))
        self._atomic_number_types = atomic_number_types

        if chiral_types is None:
            chiral_types = [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ]
        self._chiral_types = chiral_types

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping 'atomic_number' and 'chirality_type' to separately an int64 tensor
            of shape (N, 1), N is the number of atoms
        """
        atom_features = []
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features.append([
                self._atomic_number_types.index(atom.GetAtomicNum()),
                self._chiral_types.index(atom.GetChiralTag())
            ])
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.int64))

        return {
            'atomic_number': atom_features[:, 0],
            'chirality_type': atom_features[:, 1]
        }

class AttentiveFPAtomFeaturizer(BaseAtomFeaturizer):
    """The atom featurizer used in AttentiveFP

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
#AttentiveFP 是一种基于图注意力机制的分子表示方法，旨在改进药物发现过程中的分子表示能力。
#它的关键思想是利用图神经网络（GNN）和注意力机制来捕捉分子中原子之间的关系，从而更有效地建模分子的特性。
    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``B``, ``C``, ``N``, ``O``, ``F``, ``Si``, ``P``, ``S``, ``Cl``, ``As``,
      ``Se``, ``Br``, ``Te``, ``I``, ``At``, and ``other``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 5``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``, and ``other``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.
    * **Whether the atom is chiral center**
    * **One hot encoding of the atom chirality type**. The supported possibilities include
      ``R``, and ``S``.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import AttentiveFPAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                      0., 0., 0.],
                     [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                      0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                      0., 0., 0.]])}

    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    39

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    WeaveAtomFeaturizer
    PretrainAtomFeaturizer
    PAGTNAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        super(AttentiveFPAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                    'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], encode_unknown=True),
                 partial(atom_degree_one_hot, allowable_set=list(range(6))),
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 partial(atom_hybridization_one_hot, encode_unknown=True),
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_is_chiral_center,
                 atom_chirality_type_one_hot]
            )})

class PAGTNAtomFeaturizer(BaseAtomFeaturizer):
    """The atom featurizer used in PAGTN

    PAGTN is introduced in
    `Path-Augmented Graph Transformer Network. <https://arxiv.org/abs/1905.12712>`__
#AGTN（Path-Augmented Graph Transformer Network）是一种新型的图神经网络架构，旨在改进图数据的表示和处理，尤其在图上进行任务时表现出色
    The atom features include:

    * **One hot encoding of the atom type**.
    * **One hot encoding of formal charge of the atom**.
    * **One hot encoding of the atom degree**
    * **One hot encoding of explicit valence of an atom**. The supported possibilities
      include ``0 - 6``.
    * **One hot encoding of implicit valence of an atom**. The supported possibilities
      include ``0 - 5``.
    * **Whether the atom is aromatic**.

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    Examples
    --------

    >>> from rdkit import Chem
    >>> from dgllife.utils import PAGTNAtomFeaturizer

    >>> mol = Chem.MolFromSmiles('C')
    >>> atom_featurizer = PAGTNAtomFeaturizer(atom_data_field='feat')
    >>> atom_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 0., 0.]])}
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size())
    94

    See Also
    --------
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer
    PretrainAtomFeaturizer
    WeaveAtomFeaturizer
    AttentiveFPAtomFeaturizer
    """
    def __init__(self, atom_data_field='h'):
        SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                   'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                   'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                   'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                   'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re',
                   'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm',
                   'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', '*', 'UNK']

        super(PAGTNAtomFeaturizer, self).__init__(
            featurizer_funcs={
                atom_data_field: ConcatFeaturizer([partial(atom_type_one_hot,
                                                           allowable_set=SYMBOLS,
                                                           encode_unknown=False),
                                                   atom_formal_charge_one_hot,
                                                   atom_degree_one_hot,
                                                   partial(atom_explicit_valence_one_hot,
                                                           allowable_set=list(range(7)),
                                                           encode_unknown=False),
                                                   partial(atom_implicit_valence_one_hot,
                                                           allowable_set=list(range(6)),
                                                           encode_unknown=False),
                                                   atom_is_aromatic])})

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondType
        Bond types to consider. Default: ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

def bond_is_conjugated_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the bond is conjugated.#键是否共轭

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    bond_is_conjugated
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.GetIsConjugated(), allowable_set, encode_unknown)

def bond_is_conjugated(bond):
    """Get whether the bond is conjugated.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    bond_is_conjugated_one_hot
    """
    return [bond.GetIsConjugated()]

def bond_is_in_ring_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    bond_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.IsInRing(), allowable_set, encode_unknown)

def bond_is_in_ring(bond):
    """Get whether the bond is in a ring of any size.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.

    Returns
    -------
    list
        List containing one bool only.

    See Also
    --------
    bond_is_in_ring_one_hot
    """
    return [bond.IsInRing()]

def bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the stereo configuration of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of rdkit.Chem.rdchem.BondStereo
        Stereo configurations to consider. Default: ``rdkit.Chem.rdchem.BondStereo.STEREONONE``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOANY``, ``rdkit.Chem.rdchem.BondStereo.STEREOZ``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOE``, ``rdkit.Chem.rdchem.BondStereo.STEREOCIS``,
        ``rdkit.Chem.rdchem.BondStereo.STEREOTRANS``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondStereo.STEREONONE,           #无立体化学
                         Chem.rdchem.BondStereo.STEREOANY,            #任何立体化学
                         Chem.rdchem.BondStereo.STEREOZ,              #Z型立体化学
                         Chem.rdchem.BondStereo.STEREOE,              #E型立体化学
                         Chem.rdchem.BondStereo.STEREOCIS,            #顺式立体化学
                         Chem.rdchem.BondStereo.STEREOTRANS]          #反式立体化学
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)

def bond_direction_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the direction of a bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        RDKit bond instance.
    allowable_set : list of Chem.rdchem.BondDir
        Bond directions to consider. Default: ``Chem.rdchem.BondDir.NONE``,
        ``Chem.rdchem.BondDir.ENDUPRIGHT``, ``Chem.rdchem.BondDir.ENDDOWNRIGHT``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.

    See Also
    --------
    one_hot_encoding
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondDir.NONE,              #没有特定方向
                         Chem.rdchem.BondDir.ENDUPRIGHT,        #键的方向向上右侧
                         Chem.rdchem.BondDir.ENDDOWNRIGHT]      #键的方向向下右侧
    return one_hot_encoding(bond.GetBondDir(), allowable_set, encode_unknown)

class BaseBondFeaturizer(object):
    """An abstract class for bond featurizers.
    Loop over all bonds in a molecule and featurize them with the ``featurizer_funcs``.
    We assume the constructed ``DGLGraph`` is a bi-directed graph where the **i** th bond in the
    molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the **(2i)**-th and **(2i+1)**-th edges
    in the DGLGraph.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    featurizer_funcs : dict
        Mapping feature name to the featurization function.
        Each function is of signature ``func(rdkit.Chem.rdchem.Bond) -> list or 1D numpy array``.
    feat_sizes : dict
        Mapping feature name to the size of the corresponding feature. If None, they will be
        computed when needed. Default: None.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops in each bond feature.
        The features of the self loops will be zero except for the additional columns.

    Examples
    --------

    >>> from dgllife.utils import BaseBondFeaturizer, bond_type_one_hot, bond_is_in_ring
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = BaseBondFeaturizer({'type': bond_type_one_hot, 'ring': bond_is_in_ring})
    >>> bond_featurizer(mol)
    {'type': tensor([[1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.]]),
     'ring': tensor([[0.], [0.], [0.], [0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('type')
    4
    >>> bond_featurizer.feat_size('ring')
    1

    # Featurization with self loops to add

    >>> bond_featurizer = BaseBondFeaturizer(
    ...                       {'type': bond_type_one_hot, 'ring': bond_is_in_ring},
    ...                       self_loop=True)
    >>> bond_featurizer(mol)
    {'type': tensor([[1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 1.]]),
     'ring': tensor([[0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 0.],
                     [0., 1.],
                     [0., 1.],
                     [0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('type')
    5
    >>> bond_featurizer.feat_size('ring')
    2

    See Also
    --------
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, featurizer_funcs, feat_sizes=None, self_loop=False):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes
        self._self_loop = self_loop

    def feat_size(self, feat_name=None):
        """Get the feature size for ``feat_name``.

        When there is only one feature, users do not need to provide ``feat_name``.

        Parameters
        ----------
        feat_name : str
            Feature for query.

        Returns
        -------
        int
            Feature size for the feature with name ``feat_name``. Default to None.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        mol = Chem.MolFromSmiles('CCO')
        feats = self(mol)

        return feats[feat_name].shape[1]

    def __call__(self, mol):
        """Featurize all bonds in a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            For each function in self.featurizer_funcs with the key ``k``, store the computed
            feature under the key ``k``. Each feature is a tensor of dtype float32 and shape
            (N, M), where N is the number of atoms in the molecule.
        """
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])
#feat：一个新的特征，可能是某种数据结构或值。
#feat.copy()：feat 的一个拷贝。使用 copy() 方法通常是为了确保不会对原始特征的修改影响到拷贝后的特征
        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        if self._self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
    #torch.cat 是 PyTorch 中的一个函数，用于将多个张量沿指定维度连接在一起。
    #[feats, torch.zeros(feats.shape[0], 1)]：

#这里创建了一个列表，包含两个张量：
#feats：原始的张量。
#torch.zeros(feats.shape[0], 1)：生成一个全为零的新张量，其形状为 (feats.shape[0], 1)。这表示它有与 feats 相同的行数，但只有一列。
#dim=1 表示要在第二个维度（列的方向）进行连接。这意味着新生成的零列将被添加到 feats 的右侧。
                self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
                self_loop_feats[:, -1] = 1
                feats = torch.cat([feats, self_loop_feats], dim=0)
                processed_features[feat_name] = feats

        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles('CO')
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.zeros(num_atoms, feats.shape[1])
                feats[:, -1] = 1
                processed_features[feat_name] = feats

        return processed_features

class CanonicalBondFeaturizer(BaseBondFeaturizer):
    """A default featurizer for bonds.

    The bond features include:
    * **One hot encoding of the bond type**. The supported bond types include
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``,
      ``STEREOCIS``, ``STEREOTRANS``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops. The feature of the
        self loops will be zero except for the additional column.

    Examples
    --------

    >>> from dgllife.utils import CanonicalBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    12

    # Featurization with self loops to add
    >>> bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat', self_loop=True)
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    13

    See Also
    --------
    BaseBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, bond_data_field='e', self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 bond_stereo_one_hot]
            )}, self_loop=self_loop)

# pylint: disable=E1102
class WeaveEdgeFeaturizer(object):
    """Edge featurizer in Weave.

    The edge featurization is introduced in `Molecular Graph Convolutions:
    Moving Beyond Fingerprints <https://arxiv.org/abs/1603.00856>`__.

    This featurization is performed for a complete graph of atoms with self loops added,
    which considers:

    * Number of bonds between each pairs of atoms
    * One-hot encoding of bond type if a bond exists between a pair of atoms
    * Whether a pair of atoms belongs to a same ring

    Parameters
    ----------
    edge_data_field : str
        Name for storing edge features in DGLGraphs, default to ``'e'``.
    max_distance : int
        Maximum number of bonds to consider between each pair of atoms.
        Default to 7.
    bond_types : list of Chem.rdchem.BondType or None
        Bond types to consider for one hot encoding. If None, we consider by
        default single, double, triple and aromatic bonds.

    Examples
    --------

    >>> from dgllife.utils import WeaveEdgeFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CO')
    >>> edge_featurizer = WeaveEdgeFeaturizer(edge_data_field='feat')
    >>> edge_featurizer(mol)
    {'feat': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> edge_featurizer.feat_size()
    12

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, edge_data_field='e', max_distance=7, bond_types=None):
        super(WeaveEdgeFeaturizer, self).__init__()

        self._edge_data_field = edge_data_field
        self._max_distance = max_distance
        if bond_types is None:
            bond_types = [Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC]
        self._bond_types = bond_types

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self._edge_data_field]

        return feats.shape[-1]

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping self._edge_data_field to a float32 tensor of shape (N, M), where
            N is the number of atom pairs and M is the feature size.
        """
        # Part 1 based on number of bonds between each pair of atoms
        distance_matrix = torch.from_numpy(Chem.GetDistanceMatrix(mol))
        # Change shape from (V, V, 1) to (V^2, 1)
        distance_matrix = distance_matrix.float().reshape(-1, 1)
        # Elementwise compare if distance is bigger than 0, 1, ..., max_distance - 1
        distance_indicators = (distance_matrix >
                               torch.arange(0, self._max_distance).float()).float()

        # Part 2 for one hot encoding of bond type.
        num_atoms = mol.GetNumAtoms()
        bond_indicators = torch.zeros(num_atoms, num_atoms, len(self._bond_types))
        for bond in mol.GetBonds():
            bond_type_encoding = torch.tensor(
                bond_type_one_hot(bond, allowable_set=self._bond_types)).float()
            begin_atom_idx, end_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_indicators[begin_atom_idx, end_atom_idx] = bond_type_encoding
            bond_indicators[end_atom_idx, begin_atom_idx] = bond_type_encoding
        # Reshape from (V, V, num_bond_types) to (V^2, num_bond_types)
        bond_indicators = bond_indicators.reshape(-1, len(self._bond_types))

        # Part 3 for whether a pair of atoms belongs to a same ring.
        sssr = Chem.GetSymmSSSR(mol)
        ring_mate_indicators = torch.zeros(num_atoms, num_atoms, 1)
        for ring in sssr:
            ring = list(ring)
            num_atoms_in_ring = len(ring)
            for i in range(num_atoms_in_ring):
                ring_mate_indicators[ring[i], torch.tensor(ring)] = 1
        ring_mate_indicators = ring_mate_indicators.reshape(-1, 1)

        return {self._edge_data_field: torch.cat([distance_indicators,
                                                  bond_indicators,
                                                  ring_mate_indicators], dim=1)}

class PretrainBondFeaturizer(object):
    """BondFeaturizer in Strategies for Pre-training Graph Neural Networks.

    The bond featurization performed in `Strategies for Pre-training Graph Neural Networks
    <https://arxiv.org/abs/1905.12265>`__, which considers:

    * bond type
    * bond direction

    Parameters
    ----------
    bond_types : list of Chem.rdchem.BondType or None
        Bond types to consider. Default to ``Chem.rdchem.BondType.SINGLE``,
        ``Chem.rdchem.BondType.DOUBLE``, ``Chem.rdchem.BondType.TRIPLE``,
        ``Chem.rdchem.BondType.AROMATIC``.
    bond_direction_types : list of Chem.rdchem.BondDir or None
        Bond directions to consider. Default to ``Chem.rdchem.BondDir.NONE``,
        ``Chem.rdchem.BondDir.ENDUPRIGHT``, ``Chem.rdchem.BondDir.ENDDOWNRIGHT``.
    self_loop : bool
        Whether self loops will be added. Default to True.

    Examples
    --------

    >>> from dgllife.utils import PretrainBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CO')
    >>> bond_featurizer = PretrainBondFeaturizer()
    >>> bond_featurizer(mol)
    {'bond_type': tensor([0, 0, 4, 4]),
     'bond_direction_type': tensor([0, 0, 0, 0])}
    """
    def __init__(self, bond_types=None, bond_direction_types=None, self_loop=True):
        if bond_types is None:
            bond_types = [
                Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
            ]
        self._bond_types = bond_types

        if bond_direction_types is None:
            bond_direction_types = [
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT
            ]
        self._bond_direction_types = bond_direction_types
        self._self_loop = self_loop

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping 'bond_type' and 'bond_direction_type' separately to an int64
            tensor of shape (N, 1), where N is the number of edges.
        """
        edge_features = []
        num_bonds = mol.GetNumBonds()
        if num_bonds == 0:
            assert self._self_loop, \
                'The molecule has 0 bonds and we should set self._self_loop to True.'

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_feats = [
                self._bond_types.index(bond.GetBondType()),
                self._bond_direction_types.index(bond.GetBondDir())
            ]
            edge_features.extend([bond_feats, bond_feats.copy()])

        if self._self_loop:
            self_loop_features = torch.zeros((mol.GetNumAtoms(), 2), dtype=torch.int64)
            self_loop_features[:, 0] = len(self._bond_types)

        if num_bonds == 0:
            edge_features = self_loop_features
        else:
            edge_features = np.stack(edge_features)
            edge_features = F.zerocopy_from_numpy(edge_features.astype(np.int64))
            if self._self_loop:
                edge_features = torch.cat([edge_features, self_loop_features], dim=0)

        return {'bond_type': edge_features[:, 0], 'bond_direction_type': edge_features[:, 1]}

class AttentiveFPBondFeaturizer(BaseBondFeaturizer):
    """The bond featurizer used in AttentiveFP

    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    The bond features include:
    * **One hot encoding of the bond type**. The supported bond types include
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    self_loop : bool
        Whether self loops will be added. Default to False. If True, it will use an additional
        column of binary values to indicate the identity of self loops. The feature of the
        self loops will be zero except for the additional column.

    Examples
    --------

    >>> from dgllife.utils import AttentiveFPBondFeaturizer
    >>> from rdkit import Chem

    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='feat')
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    10

    >>> # Featurization with self loops to add
    >>> bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='feat', self_loop=True)
    >>> bond_featurizer(mol)
    {'feat': tensor([[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size('feat')
    11

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    PAGTNEdgeFeaturizer
    """
    def __init__(self, bond_data_field='e', self_loop=False):
        super(AttentiveFPBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated,
                 bond_is_in_ring,
                 partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                             Chem.rdchem.BondStereo.STEREOANY,
                                                             Chem.rdchem.BondStereo.STEREOZ,
                                                             Chem.rdchem.BondStereo.STEREOE])]
            )}, self_loop=self_loop)

class PAGTNEdgeFeaturizer(object):
    """The edge featurizer used in PAGTN

    PAGTN is introduced in
    `Path-Augmented Graph Transformer Network. <https://arxiv.org/abs/1905.12712>`__

    We build a complete graph and the edge features include:
    * **Shortest path between two nodes in terms of bonds**. To encode the path,
        we encode each bond on the path and concatenate their encodings. The encoding
        of a bond contains information about the bond type, whether the bond is
        conjugated and whether the bond is in a ring.
    * **One hot encoding of type of rings based on size and aromaticity**.
    * **One hot encoding of the distance between the nodes**.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_complete_graph` with
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.
    max_length : int
        Maximum distance up to which shortest paths must be considered.
        Paths shorter than max_length will be padded and longer will be
        truncated, default to ``5``.

    Examples
    --------

    >>> from dgllife.utils import PAGTNEdgeFeaturizer
    >>> from rdkit import Chem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> bond_featurizer = PAGTNEdgeFeaturizer(max_length=1)
    >>> bond_featurizer(mol)
    {'e': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}
    >>> # Get feature size
    >>> bond_featurizer.feat_size()
    14

    See Also
    --------
    BaseBondFeaturizer
    CanonicalBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    """
    def __init__(self, bond_data_field='e', max_length=5):
        self.bond_data_field = bond_data_field
        # Any two given nodes can belong to the same ring and here only
        # ring sizes of 5 and 6 are used. True & False indicate if it's aromatic or not.
        self.RING_TYPES = [(5, False), (5, True), (6, False), (6, True)]
        self.ordered_pair = lambda a, b: (a, b) if a < b else (b, a)
        self.bond_featurizer = ConcatFeaturizer([bond_type_one_hot,
                                                 bond_is_conjugated,
                                                 bond_is_in_ring])
        self.max_length = max_length

    def feat_size(self):
        """Get the feature size.

        Returns
        -------
        int
            Feature size.
        """
        mol = Chem.MolFromSmiles('C')
        feats = self(mol)[self.bond_data_field]

        return feats.shape[-1]

    def bond_features(self, mol, path_atoms, ring_info):
        """Computes the edge features for a given pair of nodes.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.
        path_atoms: tuple
            Shortest path between the given pair of nodes.
        ring_info: list
            Different rings that contain the pair of atoms
        """
        features = []
        path_bonds = []
        path_length = len(path_atoms)
        for path_idx in range(path_length - 1):
            bond = mol.GetBondBetweenAtoms(path_atoms[path_idx], path_atoms[path_idx + 1])
            if bond is None:
                import warnings
                warnings.warn('Valid idx of bonds must be passed')
            path_bonds.append(bond)

        for path_idx in range(self.max_length):
            if path_idx < len(path_bonds):
                features.append(self.bond_featurizer(path_bonds[path_idx]))
            else:
                features.append([0, 0, 0, 0, 0, 0])

        if path_length + 1 > self.max_length:
            path_length = self.max_length + 1
        position_feature = np.zeros(self.max_length + 2)
        position_feature[path_length] = 1
        features.append(position_feature)
        if ring_info:
            rfeat = [one_hot_encoding(r, allowable_set=self.RING_TYPES) for r in ring_info]
            rfeat = [True] + np.any(rfeat, axis=0).tolist()
            features.append(rfeat)
        else:
            # This will return a boolean vector with all entries False
            features.append([False] + one_hot_encoding(ring_info, allowable_set=self.RING_TYPES))
        return np.concatenate(features, axis=0)

    def __call__(self, mol):
        """Featurizes the input molecule.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Mapping self._edge_data_field to a float32 tensor of shape (N, M), where
            N is the number of atom pairs and M is the feature size depending on max_length.
        """

        n_atoms = mol.GetNumAtoms()
        # To get the shortest paths between two nodes.
        paths_dict = {
            (i, j): Chem.rdmolops.GetShortestPath(mol, i, j)
            for i in range(n_atoms)
            for j in range(n_atoms)
            if i != j
            }
        # To get info if two nodes belong to the same ring.
        rings_dict = {}
        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        for ring in ssr:
            ring_sz = len(ring)
            is_aromatic = True
            for atom_idx in ring:
                if not mol.GetAtoms()[atom_idx].GetIsAromatic():
                    is_aromatic = False
                    break
            for ring_idx, atom_idx in enumerate(ring):
                for other_idx in ring[ring_idx:]:
                    atom_pair = self.ordered_pair(atom_idx, other_idx)
                    if atom_pair not in rings_dict:
                        rings_dict[atom_pair] = [(ring_sz, is_aromatic)]
                    else:
                        if (ring_sz, is_aromatic) not in rings_dict[atom_pair]:
                            rings_dict[atom_pair].append((ring_sz, is_aromatic))
        # Featurizer
        feats = []
        for i in range(n_atoms):
            for j in range(n_atoms):

                if (i, j) not in paths_dict:
                    feats.append(np.zeros(7*self.max_length + 7))
                    continue
                ring_info = rings_dict.get(self.ordered_pair(i, j), [])
                feats.append(self.bond_features(mol, paths_dict[(i, j)], ring_info))

        return {self.bond_data_field: torch.tensor(feats).float()}
