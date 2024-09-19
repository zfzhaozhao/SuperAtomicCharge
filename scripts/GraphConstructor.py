from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
import dgl
from scipy.spatial import distance_matrix
import numpy as np
import torch
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_implicit_valence_one_hot, atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    atom_chiral_tag_one_hot, one_hot_encoding, bond_is_conjugated, atom_formal_charge, atom_num_radical_electrons, bond_is_in_ring, bond_stereo_one_hot
import pickle
import copy
import sys
import os
from dgl.data.utils import save_graphs, load_graphs
import pandas as pd
from torch.utils.data import Dataset
from dgl.data.chem import mol_to_bigraph
from dgl.data.chem import BaseBondFeaturizer
from functools import partial
import warnings
from dgl.data.utils import save_graphs, load_graphs
import multiprocessing as mp
warnings.filterwarnings('ignore')
from torchani import SpeciesConverter, AEVComputer

converter = SpeciesConverter(['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H'])
#SpeciesConverter 提供了一个 convert 方法，将化学符号转换为相应的整数索引。如果符号不在列表中，通常返回一个默认值（例如 -1）表示未找到。


def chirality(atom):  # the chirality information defined in the AttentiveFP  ## AttentiveFP 中定义的手性信息
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]  #[1, 0, True]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')] # [False, False, True]

#one_hot_encoding(value,catagues(目录列表））
#这段代码尝试从 atom 对象获取 _CIPCode 属性。这是一个用于表示原子手性的信息，通常是根据 Cahn-Ingold-Prelog（CIP）规则给出的手性标识符。
#可能的返回值包括 'R' 和 'S'，分别代表顺时针（Right）和逆时针（Sleft）手性。
#这个方法检查原子是否可能具有手性属性。返回值为布尔值 (True 或 False)。

class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot, allowable_set=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'H', 'Si'], encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})
#atom_type_one_hot: 将原子的类型进行独热编码（如 'C', 'N' 等），allowable_set 指定了可能的原子类型，encode_unknown=True 表示对未知类型进行编码。
#atom_degree_one_hot: 将原子的度数进行独热编码。度数是原子与其他原子的连接数。
#atom_formal_charge: 提取原子的正式电荷。
#atom_num_radical_electrons: 提取原子的自由电子数。
#atom_hybridization_one_hot: 将原子的杂化状态进行独热编码，encode_unknown=True 表示对未知杂化状态进行编码。
#atom_is_aromatic: 提取原子是否是芳香性的特征。
#atom_total_num_H_one_hot: 将原子结合的氢原子总数进行独热编码。
#chirality: 提取手性相关的特征。

class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                                                                               Chem.rdchem.BondStereo.STEREOANY,
                                                                                                               Chem.rdchem.BondStereo.STEREOZ,
                                                                                                               Chem.rdchem.BondStereo.STEREOE], encode_unknown=True)])})

#bond_type_one_hot: 将键的类型进行独热编码。键的类型可能包括单键、双键、三键等。
#bond_is_conjugated: 提取键是否为共轭键的特征。共轭键是指在芳香性体系中相邻的双键。
#bond_is_in_ring: 提取键是否在环中的特征。环键是指连接形成环状结构的键。
#partial(bond_stereo_one_hot, allowable_set=[Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE], encode_unknown=True)
#partial 用于创建一个新的函数，固定了 bond_stereo_one_hot 函数的部分参数。
#bond_stereo_one_hot 用于对键的立体化学进行独热编码。allowable_set 参数指定了所有可能的立体化学状态，包括：
#Chem.rdchem.BondStereo.STEREONONE：无立体化学。
#Chem.rdchem.BondStereo.STEREOANY：任何立体化学。
#Chem.rdchem.BondStereo.STEREOZ：Z-立体化学。
#Chem.rdchem.BondStereo.STEREOE：E-立体化学。
#encode_unknown=True 表示对未知的立体化学状态进行编码。



#one_hot_encoder 是一个 partial 对象，它已经固定了 atom_type_one_hot 函数的两个参数，
#所以当你调用 one_hot_encoder 时，只需要提供一个原子类型，而无需重复指定 allowable_set 和 encode_unknown
#partial 函数来自于 Python 的 functools 模块，它允许你冻结一个函数的部分参数，生成一个新的可调用对象（partial object）。这个新的对象在调用时具有更少的参数需要传递。
def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))  # 计算向量 ab 和 ac 之间的余弦角
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge #“计算每个有向边的三维信息”
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            #g.ndata 是 DGL 图对象用于存储和访问节点特征的属性。（是个字典）
            #这个角度的计算，是固定了前两个点，这样的角度计算，值是那两个点和其它所有点这件的角度关系，这？？？？
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles)*0.01, np.sum(Angles)*0.01, np.mean(Angles)*0.01, np.max(Areas), np.sum(Areas), np.mean(Areas),
                np.max(Distances)*0.1, np.sum(Distances)*0.1, np.mean(Distances)*0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
#最后的放回结果也不是所有的角度信息，而是做了整合运算，所以这个函数的用意是？？？？

AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()


def graph_from_mol(m, add_self_loop=False, add_3D=False):
    """
    :param m: molecule 
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return: 
    complex: graphs contain m1
    """
    # small molecule
    # new_order = rdmolfiles.CanonicalRankAtoms(m)
    # mol = rdmolops.RenumberAtoms(m, new_order)
    mol = m
    # construct graph
    g = dgl.DGLGraph()  # small molecule

    # add node
    num_atoms = mol.GetNumAtoms()  # number of ligand atoms  # 不应该是rdkit的mol对象才可以吗
    #总结来说，num_atoms = mol.GetNumAtoms() 这行代码的作用是获取 mol 分子对象中的原子总数，并将这个数量存储在变量 num_atoms
    g.add_nodes(num_atoms)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)
    #自环是指从一个节点到它自己的边
    #g.nodes() 方法返回一个包含所有节点的列表或集合，具体取决于使用的图处理库。

    # add edges, ligand molecule
    num_bonds = mol.GetNumBonds()  #num_bonds = mol.GetNumBonds() 这行代码的作用是计算并存储分子中所有化学键的数量。
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i) #方法返回的是一个代表化学键的对象或信息，通常包括该键连接的原子对以及键的类型（例如单键、双键等）。
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = np.concatenate([src, dst])
    dst_ls = np.concatenate([dst, src])
    g.add_edges(src_ls, dst_ls) #无向图咯

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = AtomFeaturizer(mol)['h']
   
    #ndata 是图 g 的节点特征数据字典。通过 ndata['h']，我们可以访问或设置图中节点的特征数据。'h' 是特征的键名，通常代表节点特征矩阵
    # 'charge'
    charges = [float(mol.GetAtomWithIdx(i).GetProp('molFileAlias')) for i in range(num_atoms)]
    
    #mol.GetAtomWithIdx(i)方法返回的是一个代表原子的对象，这个对象包含了关于该原子的各种信息，如原子的类型、性质以及其他相关属性。
    #GetProp('molFileAlias')：从原子对象中获取一个名为 'molFileAlias' 的属性值。这里假设 'molFileAlias' 存储了原子的电荷信息
   
    g.ndata['charge'] = torch.tensor(charges, dtype=torch.float).unsqueeze(dim=1)
    
#.unsqueeze(dim=1) 在张量的第二维（索引为 1 的维度）上添加一个维度。这个操作通常用于调整张量的形状，使其符合模型输入的要求。
#在这种情况下，如果原始张量是形状 (num_atoms,)，unsqueeze(dim=1) 将其变为形状 (num_atoms, 1)。这样做的目的是将电荷信息作为一个特征列，每个原子有一个特征值。


    # assign edge features
    # 'e', edge features
    efeats = BondFeaturizer(mol)['e']  # 重复的边存在！
    g.edata['e'] = torch.cat([efeats[::2], efeats[::2]])
#通过将边特征 efeats 中每隔一个的特征与自身连接，确保了处理重复边的情况。这使得图 g 中的边特征 'e' 与实际的边数量和重复情况一致。
    #这里的 ::2 表示步长为 2 的切片，从而只选择每隔一个的元素
    # 'd', distance
    dis_matrix_L = distance_matrix(mol.GetConformers()[0].GetPositions(), mol.GetConformers()[0].GetPositions())
    g_d = torch.tensor(dis_matrix_L[src_ls, dst_ls], dtype=torch.float).view(-1, 1)
#GetConformers() 方法返回该分子的所有构象（可能有多种空间排列），返回的列表中的第一个构象用 [0] 访问。
#GetPositions() 方法返回该构象中所有原子的三维坐标（位置），通常返回一个形状为 (num_atoms, 3) 的数组或张量。
#输出 dis_matrix_L 是一个形状为 (num_atoms, num_atoms) 的矩阵，其中的每个元素 dis_matrix_L[i][j] 是原子 i 和原子 j 之间的距离。
    #'e', total features for edges
    # g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)

    if add_3D:
        g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)
        g.ndata['pos'] = mol.GetConformers()[0].GetPositions()
        #dim=-1 指定了在最后一个维度上进行连接。在这个上下文中，dim=-1 表示特征维度（列方向）。因此，torch.cat 会将 g_d 张量沿着特征维度追加到 g.edata['e'] 张量的右边。

        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist() #tolist() 是 PyTorch 张量的一个方法，用于将张量转换为 Python 列表。这一步将 src_nodes 和 dst_nodes 从张量格式转换为列表格式
        neighbors_ls = []
        
# g.find_edges() 是一个方法，用于获取图中边的源节点和目标节点。
#range(g.number_of_edges()) 生成一个包含图中所有边索引的范围对象。g.number_of_edges() 返回图中边的总数。因此，range(g.number_of_edges()) 生成了从 0 到 num_edges-1 的整数序列，代表所有边的索引。
#g.find_edges() 方法会根据这些边的索引返回对应的源节点（src_nodes）和目标节点（dst_nodes）。返回的通常是两个张量（或列表），分别包含每条边的源节点和目标节点的索引。
        
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g.predecessors(src_node).tolist()

            #使用 g.predecessors(src_node) 获取图中 src_node 的所有前驱节点（即所有指向 src_node 的节点）。这些前驱节点是一个张量，调用 tolist() 方法将其转换为 Python 列表。
            neighbors.remove(dst_nodes[i])
            #从前驱节点列表中移除目标节点 dst_nodes[i]，因为我们只对除了当前边的目标节点外的所有邻居感兴趣。
            tmp.extend(neighbors) #将前驱节点（即邻居节点）添加到 tmp 列表中。tmp 现在包含源节点、目标节点以及所有其他邻居节点。
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        #使用 map 函数和 partial 函数（来自 functools 模块）对 neighbors_ls 列表中的每个元素应用 D3_info_cal 函数
        #。partial(D3_info_cal, g=g) 固定了参数 g，只传递 neighbors_ls 的元素给 D3_info_cal 函数。
        #D3_info_cal 函数。 是角度的计算，这就解释了前面为什么要固定前两个原子，很合理
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
    return g


def graph_from_mol_for_prediction(m):
    """
    :param m: molecule #为啥构建图数据的函数有多个？？？  这和前面是否添加3D信息得代码一样啊，存在意义是？
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return:
    complex: graphs contain m1
    """
    # small molecule
    # new_order = rdmolfiles.CanonicalRankAtoms(m)
    # mol = rdmolops.RenumberAtoms(m, new_order)
    mol = m
    # construct graph
    g = dgl.DGLGraph()  # small molecule

    # add nodes
    num_atoms = mol.GetNumAtoms()  # number of ligand atoms
    g.add_nodes(num_atoms)

    # add edges, ligand molecule
    num_bonds = mol.GetNumBonds()  #边总数
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = np.concatenate([src, dst])
    dst_ls = np.concatenate([dst, src])
    g.add_edges(src_ls, dst_ls)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = AtomFeaturizer(mol)['h']
    # 'charge'  #这里错了吧（和前面的不一致，而且你索引用来当电荷值？
    charges = [float(0) for i in range(num_atoms)]
    g.ndata['charge'] = torch.tensor(charges, dtype=torch.float).unsqueeze(dim=1)  

    # assign edge features
    # 'e', edge features
    efeats = BondFeaturizer(mol)['e']  # 重复的边存在！
    g.edata['e'] = torch.cat([efeats[::2], efeats[::2]])

    # 'd', distance
    dis_matrix_L = distance_matrix(mol.GetConformers()[0].GetPositions(), mol.GetConformers()[0].GetPositions())
    #计算原子之间的距离：我们想要计算分子中所有原子对之间的距离。将原子坐标矩阵传递给 distance_matrix() 的两个参数实际上是相同的，因为我们计算的是同一组原子之间的距离。
#对称矩阵：距离矩阵是对称的，即距离矩阵中的元素 dis_matrix[i][j] 和 dis_matrix[j][i] 是相等的。因此，传递相同的坐标矩阵两次是合理的。
    g_d = torch.tensor(dis_matrix_L[src_ls, dst_ls], dtype=torch.float).view(-1, 1)

    #'e', total features for edges
    # g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)

    g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)
    g.ndata['pos'] = mol.GetConformers()[0].GetPositions()

    # calculate the 3D info for g
    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)
    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
    return g


def graph_from_mol_new(data_dir, key, cache_path, path_marker):
    # small molecule
    # new_order = rdmolfiles.CanonicalRankAtoms(m)
    # mol = rdmolops.RenumberAtoms(m, new_order)
    #CanonicalRankAtoms(m)：对分子的原子进行标准化排序，生成一个新的原子顺序。
#RenumberAtoms(m, new_order)：根据新的原子顺序对分子进行重新编号，生成一个新的分子对象。
    add_self_loop = False
    mol = Chem.MolFromMolFile(data_dir, removeHs=False)
    #mol = Chem.MolFromMolFile(data_dir, removeHs=False) 代码行用于从 .mol 文件中读取分子数据，并创建一个 RDKit 的 Mol 对象。
    #removeHs=False 参数确保氢原子在读取过程中被保留。这个过程通常用于分子结构的处理、分析或计算。
    # construct graph
    g = dgl.DGLGraph()  # small molecule

    # add nodes
    num_atoms = mol.GetNumAtoms()  # number of ligand atoms
    g.add_nodes(num_atoms)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    # add edges, ligand molecule
    num_bonds = mol.GetNumBonds()
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = np.concatenate([src, dst])
    dst_ls = np.concatenate([dst, src])
    g.add_edges(src_ls, dst_ls)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = AtomFeaturizer(mol)['h']
    # 'charge'
    charges = [float(mol.GetAtomWithIdx(i).GetProp('molFileAlias')) for i in range(num_atoms)]
    g.ndata['charge'] = torch.tensor(charges, dtype=torch.float).unsqueeze(dim=1)

    # assign edge features
    # 'e', edge features
    efeats = BondFeaturizer(mol)['e']  # 重复的边存在！
    g.edata['e'] = torch.cat([efeats[::2], efeats[::2]])

    # 'd', distance
    dis_matrix_L = distance_matrix(mol.GetConformers()[0].GetPositions(), mol.GetConformers()[0].GetPositions())
    g_d = torch.tensor(dis_matrix_L[src_ls, dst_ls], dtype=torch.float).view(-1, 1)

    #'e', total features for edges
    # g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)

    g.edata['e'] = torch.cat([g.edata['e'], g_d], dim=-1)
    g.ndata['pos'] = mol.GetConformers()[0].GetPositions()

    # calculate the 3D info for g
    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)
    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)

    save_graphs(cache_path + path_marker + key, [g])


class GraphDataset(object):
    def __init__(self, data_dirs, cache_file_path, add_3D):
        self.data_dirs = data_dirs
        self.retained_dirs = []
        self.cache_file_path = cache_file_path
        self.add_3D = add_3D
        self._pre_process()

    def _pre_process(self):
        # for i, data_dir in enumerate(self.data_dirs):
        #     m = Chem.MolFromMolFile(data_dir, removeHs=False)
        #     atom_num = m.GetNumAtoms()
        #     if (atom_num >= 0) and (atom_num <= 65):
        #         self.retained_dirs.append(data_dir)
        if os.path.exists(self.cache_file_path):
            print('Loading previously saved dgl graphs...')
            with open(self.cache_file_path, 'rb') as f:
                self.graphs = pickle.load(f)
                #'rb': 这是打开文件的模式。'rb' 表示以二进制读取模式打开文件。这个模式是用于读取二进制数据，适用于 pickle 序列化的数据，因为 pickle 通常会生成二进制文件
        else:
            print('Generate complex graph...')
            self.graphs = []
            for i, data_dir in enumerate(self.data_dirs):
                m = Chem.MolFromMolFile(data_dir, removeHs=False)
                atom_num = m.GetNumAtoms()
                if (atom_num >= 0) and (atom_num <= 65):
                    print('Processing complex {:d}/{:d}'.format(i + 1, len(self.data_dirs)))
                    g = graph_from_mol(m, add_3D=self.add_3D)
                    self.graphs.append(g)
            with open(self.cache_file_path, 'wb') as f:
                pickle.dump(self.graphs, f)
                #'wb': 文件打开模式。'wb' 表示以二进制写入模式打开文件。这意味着文件将以二进制格式写入数据，这对于序列化数据（如 pickle 生成的数据）是合适的

    def __getitem__(self, indx):
        return self.graphs[indx]

    def __len__(self):
        # return len(self.data_dirs)
        return len(self.graphs)


class GraphDatasetNew(object):
    """
    created in 20210706
    """
    def __init__(self, data_dirs, data_keys, cache_bin_file, tmp_cache_path, path_marker='/', num_process=8):
        self.data_dirs = data_dirs
        self.data_keys = data_keys
        self.cache_bin_file = cache_bin_file
        self.num_process = num_process
        self.tmp_cache_path = tmp_cache_path
        self.path_marker = path_marker
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.cache_bin_file):
            print('Loading previously saved dgl graphs...')
            self.graphs = load_graphs(self.cache_bin_file)[0]
#load_graphs(self.cache_bin_file): 从指定的文件中加载图数据。该函数通常返回两个值：
#graphs: 包含图对象的列表或元组。
#labels: 可能的标签或其他相关信息。
        else:
            print('Generate dgl graph...')
            if not os.path.exists(self.tmp_cache_path):
                cmdline = 'mkdir -p %s' % self.tmp_cache_path
                os.system(cmdline)
# cmdline:这个变量存储了一个完整的命令行字符串，像这样：'mkdir -p /path/to/directory'。其中 /path/to/directory 是 self.tmp_cache_path 的值
#os.system 是一个函数，用于在操作系统的 shell 中执行命令。它会运行 cmdline 变量中包含的命令。
# os.system(cmdline) 将执行 mkdir -p /path/to/directory 命令，创建指定的目录及其所有父目录（如果不存在的话）。
            pool = mp.Pool(self.num_process) #Pool: multiprocessing.Pool 是 multiprocessing 模块中的一个类，用于创建一个进程池。进程池允许你并行地执行多个任务，
            #每个任务由一个独立的进程来处理。进程池可以显著提高程序在处理大量独立任务时的效率。
            #使用进程池可以充分利用多核 CPU 来加速任务的处理。
            
            self.graphs = pool.starmap(partial(graph_from_mol_new, cache_path=self.tmp_cache_path, path_marker=self.path_marker),
                                       zip(self.data_dirs, self.data_keys))
            #使用 partial 创建一个新的函数，这个函数与 graph_from_mol_new 相同，但 cache_path 和 path_marker 参数已经被预设为 self.tmp_cache_path 和 self.path_marker。
            #新的函数不再需要这些参数，因为它们已经固定了。
            #将 self.data_dirs 和 self.data_keys 打包成一个元组的迭代器。假设 self.data_dirs 是一个目录列表，self.data_keys 是与这些目录相关的键值列表。
            #zip 会生成一个包含 (data_dir, data_key) 元组的迭代器。
            
            pool.close() # # 关闭进程池，停止接受新的任务
            pool.join() # 等待进程池中的所有任务完成
            self.graphs = []
            # load the saved individual graphs
            for key in self.data_keys: #可能是文件名？？
                self.graphs.append(load_graphs(self.tmp_cache_path + self.path_marker + key)[0][0])
            save_graphs(self.cache_bin_file, self.graphs)
            cmdline = 'rm -rf %s' % self.tmp_cache_path
            os.system(cmdline)

    def __getitem__(self, indx):
        return self.graphs[indx], self.data_keys[indx]

    def __len__(self):
        return len(self.graphs)


def collate_fn(data_batch):
    graphs = data_batch
    bg = dgl.batch(graphs)
    return bg


def collate_fn_new(data_batch):
    graphs, keys = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    return bg, keys
#这个keys到底是什么
