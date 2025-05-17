import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

dataset_name="MYE"
for i in range(5):
    df= pd.read_csv(f'/media/fei/Data/zxy/singleR/{dataset_name}/fold{i}_predictions.csv')
    celltype_label=df["True_CellType"]
    predictions=df["Predicted_CellType"]
    accuracy = accuracy_score(celltype_label, predictions)
    f1 = f1_score(celltype_label, predictions, average="macro")
    precision = precision_score(celltype_label, predictions, average="macro")
    recall = recall_score(celltype_label, predictions, average="macro")
    print(f"fold {i}:\n acc:{accuracy},precision:{precision},recall:{recall},f1:{f1}")