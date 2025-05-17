import scanpy as sc
import pandas as pd
import numpy as np


def create_anndata_from_csv(metadata_file, expression_file):
    """从CSV文件创建AnnData对象

    参数:
    metadata_file: 元数据CSV文件路径
    expression_file: 表达矩阵CSV文件路径
    """
    # 读取元数据
    print(f"正在读取元数据: {metadata_file}")
    obs = pd.read_csv(metadata_file, index_col=0)

    # 读取表达矩阵
    print(f"正在读取表达矩阵: {expression_file}")
    exp_df = pd.read_csv(expression_file, index_col=0)

    # 检查细胞ID是否匹配
    obs_cells = set(obs.index)
    exp_cells = set(exp_df.index)

    common_cells = obs_cells.intersection(exp_cells)
    if len(common_cells) == 0:
        raise ValueError("元数据和表达矩阵中没有共同的细胞ID，请检查ID是否匹配。")

    # 如果存在不匹配的细胞，只保留共同的细胞
    if len(common_cells) < len(obs_cells) or len(common_cells) < len(exp_cells):
        print(f"警告: 元数据和表达矩阵中的细胞ID不完全匹配。将只保留共同的 {len(common_cells)} 个细胞。")
        obs = obs.loc[common_cells]
        exp_df = exp_df.loc[common_cells]

    # 创建基因元数据
    var = pd.DataFrame(index=exp_df.columns)
    var['gene_symbol'] = exp_df.columns

    # 创建AnnData对象
    adata = sc.AnnData(
        X=exp_df.values,
        obs=obs,
        var=var,
        obsm={'X_original': exp_df.values.copy()},  # 保存原始表达矩阵
        dtype='float32'
    )

    # 设置细胞和基因名称
    adata.obs_names = exp_df.index
    adata.var_names = exp_df.columns

    return adata


if __name__ == "__main__":
    # 文件路径
    # aa=sc.read("/media/fei/Data/zxy/scMoE/dataset/Mye/reference_adata.h5ad")
    # metadata_file = "./MYE/GSE154763_MYE_metadata.csv"
    # normalized_expression_file = "./MYE/GSE154763_MYE_normalized_expression.csv"
    # adata = create_anndata_from_csv(metadata_file, normalized_expression_file)
    # # 保存为h5ad格式

    train_adata=sc.read("/media/fei/Data/zxy/scMoE/dataset/Mye/reference_adata.h5ad")
    test_adata=sc.read("/media/fei/Data/zxy/scMoE/dataset/Mye/query_adata.h5ad")
    adata=train_adata.concatenate(test_adata)
    output_file = "./MYE/adata.h5ad"
    adata.write_h5ad(output_file)
    print(f"AnnData 对象已保存为 {output_file}")    