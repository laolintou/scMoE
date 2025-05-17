import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import scanpy as sc
import scvi


dataset_name="MYE"
if dataset_name=="MergedMonkey":
    cell_type_key="CellType"
elif dataset_name=="MYE":
    cell_type_key="cell_type"
for fold in range(5):
    # 读取数据
    train_adata = sc.read_h5ad(f'../data/{dataset_name}/cross_validation/fold_{fold}/train_adata.h5ad')
    test_adata = sc.read_h5ad(f'../data/{dataset_name}/cross_validation/fold_{fold}/test_adata.h5ad')

    # 设置LDVAE模型 (LinearSCVI)
    scvi.model.LinearSCVI.setup_anndata(train_adata)
    model = scvi.model.LinearSCVI(train_adata)
    print("LDVAE")
    model.train(max_epochs=50,early_stopping=True,use_gpu=True)

    # 获取latent representation
    train_latent = model.get_latent_representation()
    test_latent = model.get_latent_representation(test_adata)

    # 找到每个测试细胞在训练集中的K个最近邻
    k = 1  # 设定K值
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(train_latent)
    distances, indices = nbrs.kneighbors(test_latent)

    # 假设cell_type列名为'cell_type'
    predicted_labels = []
    for idx in indices:
        # 使用np.unique来找到最常见的细胞类型标签
        values, counts = np.unique(train_adata.obs[cell_type_key].values[idx], return_counts=True)
        mode_value = values[np.argmax(counts)]
        predicted_labels.append(mode_value)

    # 将预测结果存入test_adata的obs中
    test_adata.obs['predicted_cell_type'] = predicted_labels

    # 计算准确率 (Accuracy) 和 F1分数 (F1 Score)
    accuracy = accuracy_score(test_adata.obs[cell_type_key], test_adata.obs['predicted_cell_type'])
    precision = precision_score(test_adata.obs[cell_type_key], test_adata.obs['predicted_cell_type'], average='macro')
    recall = recall_score(test_adata.obs[cell_type_key], test_adata.obs['predicted_cell_type'], average='macro')
    f1 = f1_score(test_adata.obs[cell_type_key], test_adata.obs['predicted_cell_type'], average='macro')

    print(f"{fold} Accuracy: {accuracy}")
    print(f"{fold} precision: {precision}")
    print(f"{fold} recall: {recall}")
    print(f"{fold} F1 Score: {f1}")