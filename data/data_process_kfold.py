import scanpy as sc
from sklearn.model_selection import KFold, ShuffleSplit
import os
import numpy as np


def get_HVG(adata, data_is_raw):
    sc.pp.highly_variable_genes(
        adata,
        layer=None,
        n_top_genes=2000,
        batch_key=None,
        flavor="seurat_v3" if data_is_raw else "cell_ranger",
        subset=True,
    )
    return adata


# 读取数据并进行HVG筛选
dataset_name = "MYE"
adata = sc.read(f"./{dataset_name}/adata.h5ad")
adata = get_HVG(adata, False)

# 创建保存交叉验证结果的目录
cv_dir = f"./{dataset_name}/cross_validation"
os.makedirs(cv_dir, exist_ok=True)

# 设置五折交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 对数据进行五折交叉验证分割
for fold, (train_val_idx, test_idx) in enumerate(kf.split(adata)):
    # 创建当前折叠的目录
    fold_dir = f"{cv_dir}/fold_{fold}"
    os.makedirs(fold_dir, exist_ok=True)

    # 分割测试集
    test_adata = adata[test_idx].copy()

    # 进一步将train_val分割为训练集和验证集
    val_size = int(len(train_val_idx) * 0.125)  # 0.125 * 0.8 = 0.1
    ss = ShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    train_idx, val_idx = next(ss.split(train_val_idx))

    # 获取实际的索引
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    # 分割数据
    train_adata = adata[train_idx].copy()
    val_adata = adata[val_idx].copy()

    # 保存训练集、验证集和测试集
    train_adata.write(f"{fold_dir}/train_adata.h5ad")
    val_adata.write(f"{fold_dir}/val_adata.h5ad")
    test_adata.write(f"{fold_dir}/test_adata.h5ad")

    print(f"Fold {fold} 分割完成:")
    print(f"  训练集大小: {len(train_adata)} ({len(train_adata) / len(adata):.1%})")
    print(f"  验证集大小: {len(val_adata)} ({len(val_adata) / len(adata):.1%})")
    print(f"  测试集大小: {len(test_adata)} ({len(test_adata) / len(adata):.1%})")

print(f"五折交叉验证数据已保存至 {cv_dir}")