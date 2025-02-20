import torch
import numpy as np
import scipy.sparse as sp
from pygcn.models import GCN
from pygcn.utils import load_data, sparse_mx_to_torch_sparse_tensor

# 加载数据集
path = "../data/eng/"
dataset = "knowledge_points"
adj, features, labels, idx_train, idx_val, idx_test = load_data(path, dataset)

# 加载模型
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5)  # 根据您的模型参数进行设置
model.load_state_dict(torch.load('gcn_model.pth'))
model.eval()  # 设置模型为评估模式

# 将特征和邻接矩阵移动到 GPU（如果可用）
if torch.cuda.is_available():
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()

# 进行预测
with torch.no_grad():  # 不需要计算梯度
    output = model(features, adj)
    preds = output.argmax(dim=1)  # 获取每个节点的预测类别

# 输出预测结果
print("Predicted classes for the test set:")
print(preds[idx_test].cpu().numpy())  # 将预测结果移回 CPU 并转换为 NumPy 数组 