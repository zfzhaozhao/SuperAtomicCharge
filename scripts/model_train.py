import rdkit
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from GraphConstructor import *
import time
from MyUtils import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
from scipy.stats import pearsonr

torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.benchmark = True 用于优化卷积操作，通过选择最适合当前硬件和输入数据的算法来提高训练和推理速度。
#该设置适用于输入数据尺寸和网络结构在整个训练过程中不变的情况。
#对于动态输入尺寸或需要快速测试不同网络结构的场景，可能需要将其设置为 False 以避免性能不稳定。
import warnings

warnings.filterwarnings('ignore')
import torch
from dgl.data import split_dataset
from MyModel import ModifiedChargeModelV2
from itertools import accumulate
from dgl.data import Subset
import os
import argparse


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
        
#通过在后台线程中异步加载数据，BackgroundGenerator 能够减少数据加载的等待时间。这样，主线程可以在后台线程准备数据时进行其他计算，提升数据处理效率。
#BackgroundGenerator 是一个用于在后台线程中异步生成数据的生成器。它的主要目的是减少数据加载过程中的瓶颈，以提高数据处理效率。通常情况下，当主线程在处理一个批次的数据时，后台线程可以提前准备下一个批次的数据，从而缩短数据加载的时间。
#DataLoaderX 继承自 PyTorch 的 DataLoader 类，这意味着它继承了所有 DataLoader 的功能，如数据批次的生成、数据预处理等。
#通过重写 __iter__ 方法，DataLoaderX 替代了 DataLoader 的迭代器。
#super().__iter__() 调用父类 DataLoader 的 __iter__ 方法，返回一个标准的迭代器。
#BackgroundGenerator(super().__iter__()) 包装了这个标准迭代器，使得数据加载过程在后台线程中进行，从而实现数据的异步加载。

def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    # training model for one epoch 一次迭代的函数
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bg, keys = batch #bg是图数据，keys是名字？？？？
        bg = bg.to(device)
        Ys = bg.ndata.pop('charge') #pop 是一个 Python 字典的方法，它用于删除指定的键，并返回与该键相关联的值。在 ndata 中，它用于删除指定的节点特征，并返回删除的特征数据。
        #Ys是标签，我之前没注意道他是怎么把这个标签值弄上去的
        feats = bg.ndata.pop('h')  
        efeats = bg.edata.pop('e')
        outputs = model(bg, feats, efeats)
        
        #将图 bg（其结构信息仍然保留）以及从节点特征和边特征中提取的数据传递给模型进行计算。
#model 是一个图神经网络模型，通常会处理图的结构信息、节点特征和边特征，进行图卷积或其他操作以生成输出。
#在训练过程中，特征可能会从图对象中提取出来，然后进行某些转换或归一化操作。在这些操作之后，原始特征可能不再需要，或已被替代。
#删除特征可以避免冗余数据，确保在模型训练过程中只使用需要的特征。
        
        loss = loss_fn(outputs, Ys)
        loss.backward()
        optimizer.step()

 
def run_a_eval_epoch(model, valid_dataloader, device):  #评价函数
    true = []
    pred = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(valid_dataloader):
            # ChargeModel.zero_grad()
            bg, keys = batch
            bg = bg.to(device)
            Ys = bg.ndata.pop('charge')
            feats = bg.ndata.pop('h')
            efeats = bg.edata.pop('e')
            outputs = model(bg, feats, efeats)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
    return true, pred


if __name__ == '__main__':
        #if __name__ == '__main__': 是 Python 编程中一个常见的条件语句，它用于确保某些代码块只在脚本作为主程序运行时被执行，而在脚本被作为模块导入时不被执行
        #__name__ 是一个特殊的内置变量，它用于表示当前模块的名称。
#当脚本被直接运行时，Python 解释器将 __name__ 变量设置为 '__main__'。
#当脚本作为模块被导入到其他脚本中时，__name__ 的值将是模块的实际名称（即文件名，不包括扩展名）。
    argparser = argparse.ArgumentParser()
        #argparse.ArgumentParser() 是一个非常强大的工具，用于处理 Python 程序的命令行参数。通过定义和解析这些参数，可以让你的脚本更灵活，便于用户通过命令行传递不同的输入和选项。
    # model training parameters
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
        
#argparse.add_argument() 方法用于向 ArgumentParser 对象中添加命令行参数的定义
#'--gpuid':

#这是定义的参数名称。在命令行中，用户可以使用 --gpuid 来传递这个参数。
#这是一个可选参数，因为它以双破折号 -- 开头。
#type=str:

#这个参数指定了 --gpuid 的值应该被解析为字符串类型。如果用户输入的值不是字符串，argparse 会尝试将其转换为字符串类型。
#default='0':
#这是该参数的默认值。如果用户在命令行中没有提供 --gpuid 参数，那么 argparse 会使用这个默认值 '0'。
#这意味着如果没有指定 --gpuid，程序会自动使用 GPU ID '0'。
#help="gpu id for training model":

#这是对该参数的描述或帮助信息。它会在运行 python script.py --help 时显示，帮助用户理解这个参数的作用。
#必选参数，因为它没有以双破折号 -- 开头。用户在命令行中必须提供这个参数，否则程序将报错。注意，必选参数通常没有默认值。
        
    argparser.add_argument('--lr', type=float, default=10 ** -3.0, help="learning rate")                #学习率，用于训练模型时控制每次参数更新的步长。较小的学习率可能导致训练较慢，但可能更稳定。
    argparser.add_argument('--epochs', type=int, default=50, help="number of epochs ")                  #总迭代次数
    argparser.add_argument('--batch_size', type=int, default=20, help="batch size")                     #批处理样本数
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")     #早停策略中的容忍度
    argparser.add_argument('--patience', type=int, default=50, help="early stopping patience")          #早停策略中的耐心值。指在多少个 epoch 中没有显著性能提升时，训练将提前停止。
    argparser.add_argument('--l2', type=float, default=10 ** -6, help="l2 regularization")              # L2 正则化的系数。用于减少模型过拟合，通过在损失函数中添加权重的平方和来惩罚过大的权重值。
    argparser.add_argument('--repetitions', type=int, default=3, help="the number of independent runs") #执行独立实验的次数。通常用于评估模型在不同随机初始化或数据分割下的稳定性和可靠性。
    argparser.add_argument('--type_of_charge', type=str, default='e4',
                           help="type of charge, only support 'e4', 'e78', 'resp'")                     #电荷类型的选择，支持的值包括 'e4'、'e78' 和 'resp'
    argparser.add_argument('--num_process', type=int, default=4,
                           help="the number of process to generate graph data")                         #用于生成图数据的进程数量。并行处理可以加快数据准备过程。
    argparser.add_argument('--bin_data_file', type=str, default='data_e4.bin',
                           help="the bin data file name of each dataset")                               #数据集的二进制文件名。指定程序读取的输入数据文件。
    args = argparser.parse_args()
    gpuid, lr, epochs, batch_size, tolerance, patience, l2, repetitions = args.gpuid, args.lr, args.epochs, \
                                                                          args.batch_size, args.tolerance, \
                                                                          args.patience, args.l2, args.repetitions
    type_of_charge, num_process, bin_data_file = args.type_of_charge, args.num_process, args.bin_data_file  #从 argparse 解析后的 args 对象中提取了多个参数的值，并将这些值赋给相应的变量。

    # Get the upper-level directory
    path_marker = '/'
    home_path = os.path.abspath(os.path.dirname(os.getcwd()))
#os.getcwd() 获取当前工作目录的路径。
#os.path.dirname() 获取当前工作目录的父目录路径。
#os.path.abspath() 将路径转换为绝对路径，确保路径是完整的。
    model_home_path = home_path + path_marker + 'model_save'
    pred_res_path = home_path + path_marker + 'outputs'
    training_data_path = home_path + path_marker + 'training_data'
    data_keys = os.listdir(training_data_path)
    sdf_files = [training_data_path + path_marker + key for key in data_keys]
# remove the '.sdf' suffix for data_keys
    data_keys = [key.strip('.sdf') for key in data_keys]
#strip('.sdf') 从文件名中去除 .sdf 后缀。注意：strip() 方法会去除字符串两端的字符，因此这种方法只适用于 .sdf 是文件名开头或结尾的情况。
#对于只想去除文件名结尾的 .sdf 后缀，应该使用 rstrip('.sdf') 方法

    dataset = GraphDatasetNew(data_dirs=sdf_files, data_keys=data_keys,
                              cache_bin_file=home_path + path_marker + 'bin_data' + path_marker + bin_data_file,
                              tmp_cache_path=home_path + path_marker + 'tmpfiles',
                              path_marker=path_marker, num_process=num_process)

    num_data = len(dataset)
    print("number of total data is:", num_data)
    frac_list = [0.8, 0.1, 0.1]                             #frac_list 定义了训练集、验证集和测试集的比例。这里的比例是80%训练集、10%验证集和10%测试集。
    frac_list = np.asarray(frac_list)
    lengths = (num_data * frac_list).astype(int)            #通过将总数据量 num_data 乘以比例 frac_list 得到初步的样本数量，并将结果转换为整数。
    lengths[-1] = num_data - np.sum(lengths[:-1])        
#lengths[-1] 是调整后的测试集样本数量。因为整数转换可能导致总数不完全匹配，所以将最后一个数据集的样本数量调整为 num_data 减去前两个数据集样本数量的总和，以确保数据总量一致。
        
    indices = np.random.RandomState(seed=43).permutation(num_data)
#indices 生成一个随机排列的索引数组。np.random.RandomState(seed=43) 用于设定随机数种子，保证结果可重现。permutation(num_data) 返回一个随机排列的整数数组，表示数据的索引。
   
    train_idx, valid_idx, test_idx = [indices[offset - length:offset] for offset, length in
                                      zip(accumulate(lengths), lengths)]

#accumulate 不是 Python 的内置函数，但它是 Python 标准库中的 itertools 模块的一部分。accumulate 用于生成一个迭代器，该迭代器计算输入序列的累积和或累积结果。
#accumulate 计算的是输入列表的累积和：80、80+10=90、80+10+10=100。
#accumulate(lengths) 会计算每个数据集的结束索引。假设 lengths 是 [80, 10, 10]，accumulate(lengths) 将生成一个迭代器，其输出是 [80, 90, 100]，分别表示训练集、验证集和测试集的结束索引
#indices[offset - length:offset] 从 indices 中提取一段子集。
#对于第一个元组 (80, 80)，indices[80 - 80:80] 提取从 0 到 80 的索引，即训练集的索引。
#对于第二个元组 (90, 10)，indices[90 - 10:90] 提取从 80 到 90 的索引，即验证集的索引。
#对于第三个元组 (100, 10)，indices[100 - 10:100] 提取从 90 到 100 的索引，即测试集的索引。 

      stat_res = []
    for repetition_th in range(repetitions):  #在循环中执行多个独立实验
              torch.cuda.empty_cache()
#用于清空 GPU 的缓存，以释放不再需要的内存，避免 GPU 内存溢出。这在训练多个实验时尤其重要，因为每次实验都可能使用 GPU 资源。
        dt = datetime.datetime.now()
        filename = home_path + path_marker + 'model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
        print('Independent run %s' % repetition_th)
        print('model file %s' % filename)
        seed_torch(repetition_th) #用于设置 PyTorch 的随机种子，以确保实验的可重复性。repetition_th 是当前实验的编号，用于生成不同的随机种子，从而确保每次实验的随机性都不同。

        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        test_dataset = Subset(dataset, test_idx)

#ubset 是 PyTorch 提供的一个数据集类，它允许你从一个较大的数据集中提取出一部分数据作为子集。它的基本构造函数是：
#torch.utils.data.Subset(dataset, indices)
#dataset: 原始数据集，可以是 PyTorch 中的任何数据集类（如 Dataset）。
#indices: 要从原始数据集中提取的样本的索引列表。索引可以是一个整数列表，指定了数据集中要提取的样本的位置。

        print('number of train data:', len(train_dataset))
        print('number of valid data:', len(valid_dataset))
        print('number of test data:', len(test_dataset))
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn_new)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=True, collate_fn=collate_fn_new)
        test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=True, collate_fn=collate_fn_new)

#atch_size: 同样是每个批次中加载的样本数量。
#shuffle=True: 在验证过程中通常不需要打乱数据，因为验证集的目的是评估模型的性能。但是在某些情况下，打乱也可以帮助缓解验证过程中的偏差。
#collate_fn=collate_fn_new: 使用相同的自定义函数处理验证数据的批次。 这个函数，之前没注意啊

        ChargeModel = ModifiedChargeModelV2(node_feat_size=36, edge_feat_size=21, num_layers=6,
                                            graph_feat_size=200, dropout=0.1)
        device = torch.device("cuda:%s" % gpuid if torch.cuda.is_available() else "cpu")
        ChargeModel.to(device)
        optimizer = torch.optim.Adam(ChargeModel.parameters(), lr=lr, weight_decay=l2)
        stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance,
                                filename=filename)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            st = time.time()
            # train
            run_a_train_epoch(ChargeModel, loss_fn, train_dataloader, optimizer, device)

            # test
            train_true, train_pred = run_a_eval_epoch(ChargeModel, train_dataloader, device)
            valid_true, valid_pred = run_a_eval_epoch(ChargeModel, valid_dataloader, device)

            train_true = np.concatenate(np.array(train_true), 0)
            train_pred = np.concatenate(np.array(train_pred), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred = np.concatenate(np.array(valid_pred), 0)

            train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
            valid_rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))
#用于计算模型在训练集和验证集上的根均方误差（RMSE）。根均方误差是一种评估回归模型性能的常用指标，它衡量了模型预测值与真实值之间的偏差
#mean_squared_error(train_true, train_pred): 这是 sklearn.metrics 模块中的一个函数，用于计算均方误差（MSE）。均方误差是预测值与真实值之间差异的平方的平均值。函数的参数：
#RMSE 提供了预测值与真实值之间的标准差度量，单位与目标值相同，比 MSE 更直观。
            early_stop = stopper.step(valid_rmse, ChargeModel)
            end = time.time()
            if early_stop:
                break
            print("epoch:%s \t train_rmse:%.4f \t valid_rmse:%.4f \t time:%.3f s" % (
                epoch, train_rmse, valid_rmse, end - st))

        # load the best model
        stopper.load_checkpoint(ChargeModel)  #这个stopper 指代早停函数

        train_true, train_pred = run_a_eval_epoch(ChargeModel, train_dataloader, device)
        valid_true, valid_pred = run_a_eval_epoch(ChargeModel, valid_dataloader, device)
        test_true, test_pred = run_a_eval_epoch(ChargeModel, test_dataloader, device)

        # metrics
        train_true = np.concatenate(np.array(train_true), 0).flatten()
        train_pred = np.concatenate(np.array(train_pred), 0).flatten()
        valid_true = np.concatenate(np.array(valid_true), 0).flatten()
        valid_pred = np.concatenate(np.array(valid_pred), 0).flatten()
        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()
        train_rmse, train_r2, train_mae, train_rp = np.sqrt(mean_squared_error(train_true, train_pred)), \
                                                    r2_score(train_true, train_pred), \
                                                    mean_absolute_error(train_true, train_pred), pearsonr(train_true,
                                                                                                          train_pred)
        valid_rmse, valid_r2, valid_mae, valid_rp = np.sqrt(
            mean_squared_error(valid_true, valid_pred)), \
                                                    r2_score(valid_true, valid_pred), \
                                                    mean_absolute_error(valid_true,
                                                                        valid_pred), pearsonr(
            valid_true, valid_pred)
        test_rmse, test_r2, test_mae, test_rp = np.sqrt(mean_squared_error(test_true, test_pred)), \
                                                r2_score(test_true, test_pred), \
                                                mean_absolute_error(test_true, test_pred), pearsonr(test_true,
                                                                                                    test_pred)
#RMSE (Root Mean Squared Error): 计算模型的预测值与真实值之间的平均平方差的平方根。
#R2 Score (R Squared): 衡量模型预测值与真实值之间的相关程度。它表示模型能够解释数据变差量的比例。R2的值介于0和1之间，值越大表示模型越好。
#MAE (Mean Absolute Error): 计算模型的预测值与真实值之间绝对差值的平均值。它衡量模型预测值的平均偏差
#pearsonr: 使用皮尔逊积矩相关系数来衡量两个变量的线性相关程度。它表示两个变量之间的线性关系强度和方向。皮尔逊相关系数的值介于-1和1之间，接近1或者-1说明相关性强，接近0说明相关性弱
#这将返回一个元组，其中：test_rp[0] 是皮尔逊相关系数（表示线性相关程度）。test_rp[1] 是 p 值（用于检验相关系数是否显著）。

        print('***model performance***')
        print("train_rmse:%.4f \t train_r2:%.4f \t train_mae:%.4f \t train_rp:%.4f" % (
            train_rmse, train_r2, train_mae, train_rp[0]))
        print("valid_rmse:%.4f \t valid_r2:%.4f \t valid_mae:%.4f \t valid_rp:%.4f" % (
            valid_rmse, valid_r2, valid_mae, valid_rp[0]))
        print("test_rmse:%.4f \t test_r2:%.4f \t test_mae:%.4f \t test_rp:%.4f" % (
        test_rmse, test_r2, test_mae, test_rp[0]))
        stat_res.append([repetition_th, 'train', train_rmse, train_r2, train_mae, train_rp[0]])
        stat_res.append([repetition_th, 'valid', valid_rmse, valid_r2, valid_mae, valid_rp[0]])
        stat_res.append([repetition_th, 'test', test_rmse, test_r2, test_mae, test_rp[0]])
#内存循环结束后，这个是外层循环，即每一种独立实验都会保存这样的数据
    stat_res_pd = pd.DataFrame(stat_res, columns=['repetition', 'group', 'rmse', 'r2', 'mae', 'rp'])
    stat_res_pd.to_csv(
        home_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
    print(stat_res_pd[stat_res_pd.group == 'train'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'train'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'valid'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'valid'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'test'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'test'].std().values[-4:])
#分别计算和打印这后四个特征列的均值和标准差。（虽然也只有四列数值）
