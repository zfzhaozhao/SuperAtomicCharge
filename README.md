# SuperAtomicCharge
Out-of-the-box Deep Learning Prediction of Atomic Partial Charges by Graph Representation and Transfer Learning.
This source code was tested sucessfully on the basic environment with `conda=4.5.4` and `cuda=11.0`

通过图表示和迁移学习对原子部分电荷进行开箱即用的深度学习预测。 该源代码已在基本环境（conda=4.5.4 和 cuda=11.0）下测试成功

![Image text](https://github.com/zjujdj/SuperAtomicCharge/blob/main/fig/graph_Abstract.png)
## Conda Environment Reproduce
Two methods were provided for reproducing the conda environment used in this study
- **create environment using file packaged by conda-pack**
    
    Download the packaged file [dgl430_v1.tar.gz](https://drive.google.com/file/d/1Rls2ydUSoEjW_rRnvXBzBCcoB4YvcWLQ/view?usp=sharing) 
    and following commands can be used to reproduce the conda environment:
    ```python
    mkdir /opt/conda_env/dgl430_v1
    tar -xvf dgl430_v1.tar.gz -C /opt/conda_env/dgl430_v1
    source activate /opt/conda_env/dgl430_v1
    conda-unpack
    ```
  
- **create environment using files provided in `./envs` directory**
    
    The following commands can be used to reproduce the conda environment:
    ```python
    conda create --prefix=/opt/conda_env/dgl430_v1 --file conda_packages.txt
    source activate /opt/conda_env/dgl430_v1
    pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -r pip_packages.txt

    ```
  
## Usage
Users can directly use our well-trained model (depoisted in `./model_save/` directory) to predicted the corresponding 
atomic partial charges (DDEC4, DDEC78 and RESP). Because this method was based on the 3D molecular structures. Therefore, 
the actual use of this method should be conducted  on optimized molecules containing 3D coordinates, such as molecules 
optimized by MMFF force field, PM7 method and so on, and the inputs should  be a sdf file containing multiple molecules 
with 3D coordinates. The input  example was deposited in `./inputs/3cl-min.sdf` or `./inputs/casp8-min.sdf`. For users 
who want to train their own model using new datasets, we also show a model training example in the following. The 
corresponding training data was deposited in `./training_data`. The label of training data can be assessed using the
script `./scripts/get_sdf_charge.py`. In addtion, we also provided a web server for directly predicting the atomic partial 
charges in the [deepchargepredictor server](http://cadd.zju.edu.cn/deepchargepredictor/)

用户可以直接使用我们在 ./model_save/ 目录中保存的训练好的模型来预测相应的原子部分电荷（DDEC4、DDEC78 和 RESP）。由于该方法基于三维分子结构，因此实际使用时应在包含三维坐标的优化分子上进行，例如通过 MMFF 力场、PM7 方法等优化的分子，输入文件应为包含多个具有三维坐标的分子的 sdf 文件。输入示例可以在 ./inputs/3cl-min.sdf 或 ./inputs/casp8-min.sdf 中找到。对于希望使用新数据集训练自己模型的用户，我们也提供了模型训练示例。相应的训练数据存放在 ./training_data 中。可以使用 ./scripts/get_sdf_charge.py 脚本访问训练数据的标签。此外，我们还提供了一个可以直接预测原子部分电荷的 deepchargepredictor 服务器。

- **step 1: Clone the Repository**
```python
git clone https://github.com/zjujdj/SuperAtomicCharge.git
```

- **step 2: Construction of Conda Environment**
```python
# method1 in 'Conda Environment Reproduce' section
mkdir /opt/conda_env/dgl430_v1
tar -xvf dgl430_v1_.tar.gz -C /opt/conda_env/dgl430_v1
source activate /opt/conda_env/dgl430_v1
conda-unpack

# method2 in 'Conda Environment Reproduce' section
cd ./SuperAtomicCharge/envs
conda create --prefix=/opt/conda_env/dgl430_v1 --file conda_packages.txt
source activate /opt/conda_env/dgl430_v1
pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r pip_packages.txt
```

- **step 3: Charge Prediction**
```python
cd ./SuperAtomicCharge/scripts
nohup python3 -u model_prediction_linux.py --job_of_name=hello_charge --type_of_charge=e4 --input_file=3cl-min.sdf 
--correct_charge --device=cpu > ../outputs/prediction.log 2>&1 &

# model_prediction_linux.py use help
python3 model_prediction_linux.py -h
```

- **step 4: Model Training Example**
```python
cd ./SuperAtomicCharge/scripts
nohup python3 -u model_train.py --gpuid=0 --lr=0.0001 --epochs=5 --batch_size=20 --tolerance=0 --patience=3 --l2=0.000001 
--repetitions=2 --type_of_charge=e4 --num_process=4 --bin_data_file=data_e4.bin > ../outputs/training.log 2>&1 &

# model_prediction_linux.py use help
python model_train.py -h
```

## Acknowledgement
Some scripts were based on the [dgl project](https://github.com/awslabs/dgl-lifesci). 
We'd like to show our sincere thanks to them.
