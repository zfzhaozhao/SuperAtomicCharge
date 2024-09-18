"""
some codes in this script was based on
https:https://github.com/awslabs/dgl-lifesci
"""

import torch.nn as nn
from dgllife.model.gnn import GAT  #图注意力网络
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax


class AttentiveGRU1(nn.Module):  #处理边的
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size) #与完整的 nn.GRU 不同，nn.GRUCell 只实现了 GRU 的单个时间步计算，而不处理整个序列。它适合于需要对每个时间步单独进行处理的情况

    def forward(self, g, edge_logits, edge_feats, node_feats): #这个 edge_logits 是什么？？？
        g = g.local_var()
        #.local_var():这个方法用于创建一个图的局部副本。具体来说，它返回一个新的图对象，这个图对象的结构和数据与原始图 g 相同，但它是一个局部副本，通常用于在计算过程中进行临时操作。
        #使用 local_var() 可以确保对图的修改不会影响到原始图 g，这对于图计算中的中间步骤特别有用。
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
#update_all 是 DGL 图对象中的一个方法，用于在图的所有节点上执行消息传递操作。在这个方法中，通常会指定一个消息传递函数（message_func）和一个聚合函数（reduce_func）
#fn.copy_edge('e', 'm'):这是一个消息传递函数（message_func），它定义了在图中传递消息的方式。fn.copy_edge 是一个 DGL 提供的内置函数，用于将边上的特征从源节点复制到消息中。
#'e' 是边特征的键，表示要复制的边特征。
#'m' 是消息的键，表示消息的内容将被存储到 'm' 中。
#这个函数的作用是将每条边的特征 'e' 复制到消息 'm' 中，消息 'm' 将在接下来的步骤中被用于聚合。

#fn.sum('m', 'c'):这是一个聚合函数（reduce_func），它定义了如何将消息聚合到目标节点上。fn.sum 是 DGL 提供的内置函数，用于对传递到目标节点的所有消息进行求和操作。
#'m' 是消息的键，表示要进行聚合的消息。
#'c' 是聚合结果的键，表示将聚合结果存储在 'c' 中。
#这个函数的作用是将从所有邻居节点传递过来的消息 'm' 进行求和，并将结果存储在目标节点的特征 'c' 中。
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))
#得到的是每个边的更新特征，这些特征经过 GRU 单元处理并通过 ReLU 激活函数进行非线性变换。

class AttentiveGRU2(nn.Module): #处理节点的
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits) #g.edata: 这是一个字典，用于存储图中边的数据特征。
        g.ndata['hv'] = self.project_node(node_feats) #g.ndata: 这是一个字典，用于存储图中节点的数据特征。

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        #fn.src_mul_edge: 这是 DGL 中的一个消息函数，用于计算每条边的消息。在这个例子中，它执行源节点特征和边特征的乘法。
#参数解释:'hv': 这是源节点的特征名称。'a': 这是边特征的名称。'm': 这是计算出的消息的存储键名。
#对于每条边，fn.src_mul_edge('hv', 'a', 'm') 会将源节点的 'hv' 特征与边的 'a' 特征逐元素相乘，得到消息 'm'。
#  fn.sum: 这是 DGL 中的一个聚合函数，用于对边发来的消息进行聚合。在这个例子中，它对消息进行求和。
#参数解释:'m': 这是从消息函数中传递过来的消息名称。'c': 这是聚合结果的存储键名。
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout): 
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size, 
                                           graph_feat_size, dropout)  ##node_feat_size, edge_feat_size, edge_hidden_size

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}
    #{'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)} 是在 DGL（Deep Graph Library）中的消息传递函数中，
    #定义了如何构造消息。它的作用是将源节点的特征和边特征结合起来，形成一个新的消息。
    #dim=1 指定了沿着特征维度进行拼接。假设源节点特征的维度是 feature_dim，边特征的维度是 edge_feature_dim，拼接后的消息的维度将是 feature_dim + edge_feature_dim
    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        #apply_edges 是 DGL 提供的一个方法，用于在图的边上应用某个操作。
        #它接受一个函数作为参数，这个函数会被应用到每条边上。

        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))


class ModifiedChargeModelNNV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedChargeModelNNV2, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats


class ModifiedChargeModelV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedChargeModelV2, self).__init__()

        self.gnn = ModifiedChargeModelNNV2(node_feat_size=node_feat_size,
                                           edge_feat_size=edge_feat_size,
                                           num_layers=num_layers,
                                           graph_feat_size=graph_feat_size,
                                           dropout=dropout)
        self.predict = nn.Sequential(
            nn.Linear(graph_feat_size, graph_feat_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return self.predict(sum_node_feats)


class ModifiedChargeModelV2New(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0., n_tasks=1):
        super(ModifiedChargeModelV2New, self).__init__()

        self.gnn = ModifiedChargeModelNNV2(node_feat_size=node_feat_size,
                                           edge_feat_size=edge_feat_size,
                                           num_layers=num_layers,
                                           graph_feat_size=graph_feat_size,
                                           dropout=dropout)
        self.predict = nn.Sequential(
            nn.Linear(graph_feat_size, graph_feat_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return self.predict(sum_node_feats)


class ModifiedChargeModelNNV3(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedChargeModelNNV3, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )
        self.sum_predictions = 0
        self.num_layers = num_layers

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_predictions = self.sum_predictions + self.predict(node_feats)
        return self.sum_predictions / (self.num_layers - 1)


class ModifiedChargeModelV3(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedChargeModelV3, self).__init__()

        self.gnn = ModifiedChargeModelNNV3(node_feat_size=node_feat_size,
                                            edge_feat_size=edge_feat_size,
                                            num_layers=num_layers,
                                            graph_feat_size=graph_feat_size,
                                            dropout=dropout)

    def forward(self, g, node_feats, edge_feats):
        predictions = self.gnn(g, node_feats, edge_feats)
        return predictions


class ModifiedGATPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None):
        super(ModifiedGATPredictor, self).__init__()

        self.gnn = GAT(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.predict = nn.Sequential(nn.Linear(gnn_out_feats, 1))

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        return self.predict(node_feats)


# class ModifiedChargeModel(nn.Module):
#     def __init__(self,
#                  node_feat_size,
#                  edge_feat_size,
#                  num_layers=2,
#                  graph_feat_size=200,
#                  dropout=0.):
#         super(ModifiedChargeModel, self).__init__()
#
#         self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
#                                   edge_feat_size=edge_feat_size,
#                                   num_layers=num_layers,
#                                   graph_feat_size=graph_feat_size,
#                                   dropout=dropout)
#         self.predict = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(graph_feat_size, 1)
#         )
#
#     def forward(self, g, node_feats, edge_feats):
#         node_feats = self.gnn(g, node_feats, edge_feats)
#         return self.predict(node_feats)


# incorporate both the node and edge features using Multilayer Perception
class AttentiveMLP1(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveMLP1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        # self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)
        self.MPL = nn.Sequential(
            nn.Linear(edge_hidden_size + node_feat_size, node_feat_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(node_feat_size, node_feat_size),
            nn.Dropout(dropout)
        )

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        # return F.relu(self.gru(context, node_feats))
        return F.relu(self.MPL(torch.cat([context, node_feats], dim=1)))


class AttentiveMLP2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveMLP2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        # self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)
        self.MPL = nn.Sequential(
            nn.Linear(edge_hidden_size + node_feat_size, node_feat_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(node_feat_size, node_feat_size),
            nn.Dropout(dropout)
        )

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        # return F.relu(self.gru(context, node_feats))
        return F.relu(self.MPL(torch.cat([context, node_feats], dim=1)))


class GetMLPContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetMLPContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        # self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
        #                                    graph_feat_size, dropout)
        self.attentive_mlp = AttentiveMLP1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_mlp(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNMLPLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNMLPLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        # self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.attentive_mlp = AttentiveMLP2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        # return self.bn_layer(self.attentive_gru(g, logits, node_feats))
        return self.bn_layer(self.attentive_mlp(g, logits, node_feats))


class GNNMLP(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(GNNMLP, self).__init__()

        self.init_context = GetMLPContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNMLPLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats


class GNNMLPPredictor(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(GNNMLPPredictor, self).__init__()

        self.gnn = GNNMLP(node_feat_size=node_feat_size,
                          edge_feat_size=edge_feat_size,
                          num_layers=num_layers,
                          graph_feat_size=graph_feat_size,
                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Linear(graph_feat_size, graph_feat_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return self.predict(sum_node_feats)
