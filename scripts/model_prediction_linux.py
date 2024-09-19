import rdkit
from rdkit import Chem
from GraphConstructor import graph_from_mol_for_prediction
from MyUtils import *
torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings('ignore')
import torch
from MyModel import ModifiedChargeModelV2, ModifiedChargeModelV2New
import dgl
import argparse
import datetime


# Get the upper-level directory # 通常指的是获取当前工作目录的上一级目录路径。
path_marker = '/'
home_path = os.path.abspath(os.path.dirname(os.getcwd()))
model_home_path = home_path + path_marker + 'model_save'
pred_res_path = home_path + path_marker + 'outputs'
input_data_path = home_path + path_marker + 'inputs'
batch_size = 500
models = ['e4_3D_0_resp_.pth', 'e78_3D_1_resp_.pth', '2021-07-09_16_45_29_775301.pth']
charges = ['e4', 'e78', 'resp']

#直接给出模型参数了，不同电荷求解用不同的模型参数

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--job_of_name', type=str, default='hello_charge', help="the unique flag for each job")                       # 用于指定每个作业的唯一标识符
    argparser.add_argument('--type_of_charge', type=str, default='e4', help="type of charge, only support 'e4', 'e78', 'resp'")          #指定电荷的类型。
    argparser.add_argument('--input_file', type=str, default='3cl-min.sdf', help="input file name, only support .sdf format")            #指定输入文件的名称。该参数用于指定输入文件的名称，并且文件格式仅支持 .sdf（结构数据文件）格式。
    argparser.add_argument('--correct_charge', action="store_true", help="correct the summation of predicted charge of atoms in same "   #用于指定是否对同一分子中原子的预测电荷总和进行修正，使其为整数。该参数用于选择是否校正原子的电荷总和以确保其为整数。

                                                                         "molecule to make it as an integer or not")
    argparser.add_argument('--device', type=str, default='gpu', help="what type of device was used in the prediction, gpu or cpu")        # 指定用于预测的设备类型。

    args = argparser.parse_args()
    
#parse_args():这是 ArgumentParser 实例的方法，用于解析命令行参数。它会读取传递给程序的命令行参数，并根据你之前定义的参数配置将其转换为适当的格式。
#args:args 是 parse_args() 方法返回的结果，类型为 argparse.Namespace。
#这个 Namespace 对象的属性名称与定义参数时使用的参数名称相同，属性值则是命令行中传递给这些参数的值。
    
    job_of_name, type_of_charge, input_file, correct_charge = args.job_of_name, args.type_of_charge, args.input_file, args.correct_charge
    device = args.device
    cmdline = 'rm -rf %s && mkdir %s' % (pred_res_path + path_marker + job_of_name, pred_res_path + path_marker + job_of_name) #%s 是占位符，用于后续插入具体的目录路径。
    os.system(cmdline)
    pred_res_path = pred_res_path + path_marker + job_of_name

    print('********************************job_of_name:%s, start*************************************************\n' % job_of_name)
    print('time', datetime.datetime.now())
    print(args)
   
    assert (type_of_charge in charges), "type of charge error, only support 'e4', 'e78' or 'resp'"
    assert input_file.endswith('.sdf'), "input file format error, only support .sdf format"
    assert os.path.exists(input_data_path + path_marker + input_file), 'input file %s not exists' % (input_data_path + path_marker + input_file)
    assert device in ['gpu', 'cpu'], 'only gpu or cpu was supported for device'

    data_folds = [input_file]
    type_of_charges = [type_of_charge]

    for data_fold_th, data_fold in enumerate(data_folds):
        for charge_th, charge in enumerate(type_of_charges):
            torch.cuda.empty_cache() #清空未使用的 GPU 内存缓存。
            # get the prediction data-sets:
            sdfs = Chem.SDMolSupplier(input_data_path + path_marker + data_fold, removeHs=False)
            valid_mol_ids = []
            graphs = []
            for i, mol in enumerate(sdfs):
                if mol:
                    try:
                        g = graph_from_mol_for_prediction(mol)
                        graphs.append(g)
                        valid_mol_ids.append(i)
                    except:
                        pass
                else:
                    pass

            # model prediction
            if charge != 'resp':
                ChargeModel = ModifiedChargeModelV2(node_feat_size=36, edge_feat_size=21, num_layers=6,
                                                    graph_feat_size=200,
                                                    dropout=0.1)
            else:
                ChargeModel = ModifiedChargeModelV2New(node_feat_size=36, edge_feat_size=21, num_layers=6,
                                                       graph_feat_size=200,
                                                       dropout=0.1, n_tasks=1 + 65)

            if device == 'gpu' and torch.cuda.is_available():
                # get the gpu device with maximum video memory  # 获取显存最多的 GPU 设备
                outputs = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
                
#这行代码使用 os.popen 执行 shell 命令 nvidia-smi，这个命令是 NVIDIA 提供的工具，用于监控 GPU 的状态。
#-q 选项表示以查询模式运行，-d Memory 表示仅显示与显存相关的信息。
#grep -A4 GPU 用于找到包含 GPU 信息的行，并显示其后 4 行内容。grep Free 用于筛选出包含“Free”字样的行，这些行表示当前空闲的显存量。

                memory_gpu = [int(x.split()[2]) for x in outputs.readlines()]
#outputs.readlines() 读取 shell 命令的所有输出行。
#x.split()[2] 将每一行分割成多个部分，并提取出显存量（通常为每行的第三部分）。
#将显存量转换为整数并存储在 memory_gpu 列表中。
                gpu_id = str(np.argmax(memory_gpu))
                device = torch.device("cuda:%s" % gpu_id)
            elif device == 'gpu' and not torch.cuda.is_available():
                print('no gpu device was available, the device was set as cpu')
                device = torch.device("cpu")
            else:
                device = torch.device("cpu")
            ChargeModel.load_state_dict(
                torch.load(model_home_path + path_marker + models[charges.index(type_of_charge)], map_location='cpu')['model_state_dict'])
            ChargeModel.to(device)
#torch.load 用于加载保存的 PyTorch 模型文件。从加载的模型文件中提取 model_state_dict。通常，保存的模型文件包含一个字典，其中 model_state_dict 键对应着模型的状态字典（即模型的参数
#load_state_dict 方法将加载的状态字典应用到 ChargeModel 实例中，以恢复模型的权重。
            ChargeModel.eval()
            with torch.no_grad():
                pred = []
                num_batch = len(graphs) // batch_size
                for i_batch in range(num_batch):
                    bg = dgl.batch(graphs[batch_size * i_batch:batch_size * (i_batch + 1)])
#部分代码从 graphs 列表中切片出当前批次的图。graphs 是一个图的列表，其中每个元素都是一个 DGL 图对象。
#batch_size * i_batch 和 batch_size * (i_batch + 1) 确定了当前批次图的索引范围。具体来说：
#batch_size * i_batch 是当前批次的起始索引。
#batch_size * (i_batch + 1) 是当前批次的结束索引（不包括该索引）。

                    bg = bg.to(device)
                    feats = bg.ndata.pop('h')
                    efeats = bg.edata.pop('e')
                    outputs = ChargeModel(bg, feats, efeats)
                    if charge != 'resp':
                        pred.append(outputs.data.cpu().numpy())
                    else:
                        pred.append(outputs[:, 0].view(-1, 1).data.cpu().numpy()) #为啥resp的处理比较特别勒

                # last batch
                bg = dgl.batch(graphs[batch_size * num_batch:])
                bg = bg.to(device)
                feats = bg.ndata.pop('h')
                efeats = bg.edata.pop('e')
                outputs = ChargeModel(bg, feats, efeats)
                if charge != 'resp':
                    pred.append(outputs.data.cpu().numpy())
                else:
                    pred.append(outputs[:, 0].view(-1, 1).data.cpu().numpy())
                pred = np.concatenate(np.array(pred), 0)
            pred = iter(pred) 
#iter() 是 Python 内置的函数，用于将可迭代对象（如列表、元组、集合等）转换为迭代器。迭代器是一个对象，提供了对数据的逐个访问方式而不需要一次性将所有数据加载到内存中。

            sdf_file_name = '%s_new_%s.sdf' % (data_fold[:-4], charge)
            output_sdf_file = pred_res_path + path_marker + '%s_new_%s.sdf' % (data_fold[:-4], charge)
            writer = Chem.SDWriter(output_sdf_file)
            for valid_idx in valid_mol_ids:
                mol = sdfs[valid_idx]
                num_atoms = mol.GetNumAtoms()
                mol_pred_charges = []
                for i in range(num_atoms):
                    mol_pred_charges.append(next(pred)[0])
#历分子 mol 中的每个原子。num_atoms 是分子中的原子总数，所以这个循环会执行 num_atoms 次。
#mol_pred_charges.append(next(pred)[0])
#从预测结果的迭代器 pred 中获取下一个预测值，并将其添加到 mol_pred_charges 列表中。这里的 next(pred) 会从迭代器中提取下一个预测值，假设每个预测值是一个包含电荷的数组或张量。
#[0] 是在取出预测值后访问预测值中的第一个元素。假设预测结果是一个包含多个预测值的数组或张量，这里选择第一个元素（通常是电荷的值）。
                # charge correlation
                sum_abs_pred = np.sum(np.abs(mol_pred_charges))  #预测电荷绝对值的总和 sum_abs_pred
                dQ = np.sum(mol_pred_charges) - Chem.rdmolops.GetFormalCharge(mol)

#Chem.rdmolops.GetFormalCharge(mol)：使用 RDKit 库的 GetFormalCharge(mol) 方法计算实际分子的总电荷（正式电荷）。这个方法返回分子的实际电荷值
#dQ 是预测电荷总和与实际电荷之间的差值，即 dQ = 总预测电荷 - 实际电荷。

                Qcorr = np.array(mol_pred_charges) - (np.abs(mol_pred_charges) * dQ) / sum_abs_pred

#np.array(mol_pred_charges)：将 mol_pred_charges 列表转换为 NumPy 数组，以便进行数组运算。
#np.abs(mol_pred_charges) * dQ：计算预测电荷绝对值与 dQ 的乘积。这个步骤是为了根据电荷差异 dQ 对预测电荷进行调整。
#(np.abs(mol_pred_charges) * dQ) / sum_abs_pred：将上述乘积除以 sum_abs_pred，以实现归一化调整。
#np.array(mol_pred_charges) - ...：将上述归一化调整量从原始预测电荷中减去，得到调整后的预测电荷 Qcorr。
#调整预测电荷，使其更好地符合实际分子的总电荷。通过对每个电荷进行比例调整，使得预测电荷的总和接近实际电荷，从而提高预测的准确性。

#是将调整后的预测电荷（如果需要）或原始预测电荷（如果不需要调整）保存到分子对象中，并将分子对象写入一个文件
                if correct_charge:
                    for i in range(num_atoms):
                        mol.GetAtomWithIdx(i).SetProp('molFileAlias', str(Qcorr[i]))
#对每个原子设置一个名为 'molFileAlias' 的属性，其值为调整后的预测电荷 Qcorr[i]。mol.GetAtomWithIdx(i) 用于获取分子的第 i 个原子，然后使用 SetProp 方法设置该原子的属性。
#设置分子的总电荷属性 'charge'，其值为调整后的预测电荷 Qcorr 的总和。np.sum(Qcorr) 计算调整后的预测电荷的总和，并将其转换为整数和字符串形式进行存储。
#需要注意的是，实际的 SDF 文件中原子顺序和 <molFileAlias> 顺序应该是一致的，因为他始终没有把对应的原子打印，我感觉印应该可以自己处理，直接打印原子-电荷对应文件
                    mol.SetProp('charge', str(int(np.sum(Qcorr))))
                else:
                    for i in range(num_atoms):
                        mol.GetAtomWithIdx(i).SetProp('molFileAlias', str(mol_pred_charges[i]))
                    mol.SetProp('charge', str(int(np.sum(Qcorr))))
                writer.write(mol)

            print('the predicted charge was stored in file: %s' % output_sdf_file)
            print('for assessing the predicted charge, please see the script (example):')
            print(home_path + path_marker + 'scripts' + path_marker + 'get_sdf_charge.py')
    print('********************************job_of_name:%s, end*************************************************\n' % job_of_name)
