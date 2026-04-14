"""
模型训练与评估脚本
加载预处理好的小数据集，一键复现论文中的所有实验结果
"""
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from xgboost import XGBClassifier
from statsmodels.stats.contingency_tables import mcnemar

# ====================== 配置参数 ======================
INPUT_CSV = "carrada_radar_4d_dataset.csv"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 危险等级权重
W_DISTANCE = 0.3
W_VELOCITY = 0.3
W_ANGLE = 0.2
W_RCS = 0.2

# 随机种子（保证结果可复现）
SEED = 42

# ====================== 计算危险等级 ======================
def calculate_hazard_level(df):
    """计算每个样本的危险等级：低/中/高"""
    scaler = MinMaxScaler()
    feat_norm = scaler.fit_transform(df[["distance", "velocity", "angle", "rcs"]])
    
    H = (W_DISTANCE * (1 / (feat_norm[:, 0] + 1e-6)) +
         W_VELOCITY * np.abs(feat_norm[:, 1]) +
         W_ANGLE * np.abs(feat_norm[:, 2]) +
         W_RCS * feat_norm[:, 3])
    
    q1 = np.quantile(H, 0.33)
    q2 = np.quantile(H, 0.66)
    hazard_level = np.where(H < q1, 0, np.where(H < q2, 1, 2))
    df["hazard"] = hazard_level
    return df

# ====================== 模型训练与评估 ======================
def train_and_evaluate(X, y, task_name):
    """通用训练评估函数，返回5折交叉验证结果"""
    unique_y = np.unique(y)
    print(f"🏷️  {task_name} 任务标签: {unique_y}")
    
    # 数据集划分：8:1:1
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED, stratify=y
    )
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(kf.split(X_train_val))
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=SEED, eval_metric="mlogloss")
    }
    
    cv_results = {name: {"acc": [], "pre": [], "rec": [], "f1": []} for name in models}
    test_results = {}
    trained_models = {}
    test_preds = {}
    
    print(f"\n--- 开始【{task_name}】5折交叉验证 ---")
    for train_idx, val_idx in tqdm(folds, desc="5折训练进度"):
        X_tr, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_tr, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            acc = accuracy_score(y_val, y_pred)
            pre, rec, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average="macro", zero_division=0
            )
            
            cv_results[name]["acc"].append(acc)
            cv_results[name]["pre"].append(pre)
            cv_results[name]["rec"].append(rec)
            cv_results[name]["f1"].append(f1)
    
    print(f"\n--- 开始【{task_name}】测试集评估 ---")
    for name, model in models.items():
        model.fit(X_train_val, y_train_val)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        pre, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        
        test_results[name] = {"acc": acc, "pre": pre, "rec": rec, "f1": f1}
        trained_models[name] = model
        test_preds[name] = y_pred
    
    return cv_results, test_results, trained_models, test_preds, X_test, y_test

# ====================== 生成论文表格 ======================
def generate_paper_tables(df, 
                          cls_cv, cls_models, cls_preds, X_test_cls, y_test_cls,
                          haz_cv, haz_models, haz_preds, X_test_haz, y_test_haz):
    result_file = os.path.join(OUTPUT_DIR, "paper_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        # 表1：5折交叉验证性能
        f.write("="*80 + "\n")
        f.write("论文表1：各模型双任务性能（5折交叉验证）\n")
        f.write("="*80 + "\n\n")
        
        f.write("--- 障碍物分类任务 ---\n")
        f.write(f"{'模型':<10} {'准确率':<15} {'精确率':<15} {'召回率':<15} {'F1':<15}\n")
        f.write("-"*80 + "\n")
        for name in ["KNN", "SVM", "RF", "XGBoost"]:
            acc = f"{np.mean(cls_cv[name]['acc']):.3f}±{np.std(cls_cv[name]['acc']):.3f}"
            pre = f"{np.mean(cls_cv[name]['pre']):.3f}±{np.std(cls_cv[name]['pre']):.3f}"
            rec = f"{np.mean(cls_cv[name]['rec']):.3f}±{np.std(cls_cv[name]['rec']):.3f}"
            f1 = f"{np.mean(cls_cv[name]['f1']):.3f}±{np.std(cls_cv[name]['f1']):.3f}"
            f.write(f"{name:<10} {acc:<15} {pre:<15} {rec:<15} {f1:<15}\n")
        
        f.write("\n--- 危险等级评估任务 ---\n")
        f.write(f"{'模型':<10} {'准确率':<15} {'精确率':<15} {'召回率':<15} {'F1':<15}\n")
        f.write("-"*80 + "\n")
        for name in ["KNN", "SVM", "RF", "XGBoost"]:
            acc = f"{np.mean(haz_cv[name]['acc']):.3f}±{np.std(haz_cv[name]['acc']):.3f}"
            pre = f"{np.mean(haz_cv[name]['pre']):.3f}±{np.std(haz_cv[name]['pre']):.3f}"
            rec = f"{np.mean(haz_cv[name]['rec']):.3f}±{np.std(haz_cv[name]['rec']):.3f}"
            f1 = f"{np.mean(haz_cv[name]['f1']):.3f}±{np.std(haz_cv[name]['f1']):.3f}"
            f.write(f"{name:<10} {acc:<15} {pre:<15} {rec:<15} {f1:<15}\n")
        
        # 表2：特征重要性
        f.write("\n\n" + "="*80 + "\n")
        f.write("论文表2：特征重要性排序（基于XGBoost）\n")
        f.write("="*80 + "\n\n")
        
        xgb_cls = cls_models["XGBoost"]
        imp = xgb_cls.feature_importances_
        feats = ["距离", "相对速度", "方位角", "回波强度(RCS)"]
        sorted_idx = np.argsort(imp)[::-1]
        
        f.write(f"{'特征':<20} {'得分':<10} {'排名':<5}\n")
        f.write("-"*40 + "\n")
        for rank, idx in enumerate(sorted_idx, 1):
            f.write(f"{feats[idx]:<20} {imp[idx]:.3f}{'':<7} {rank:<5}\n")
        
        # 表3：混淆矩阵
        f.write("\n\n" + "="*80 + "\n")
        f.write("论文表3：XGBoost障碍物分类混淆矩阵\n")
        f.write("="*80 + "\n\n")
        
        cm = confusion_matrix(y_test_cls, cls_preds["XGBoost"])
        labels = ["行人", "自行车", "车辆"]
        f.write(f"{'真实\\预测':<10} " + " ".join([f"{l:<10}" for l in labels]) + "\n")
        f.write("-"*(10 + 10*len(labels)) + "\n")
        for i in range(len(labels)):
            row = " ".join([f"{cm[i][j]:<10}" for j in range(len(labels))])
            f.write(f"{labels[i]:<10} {row}\n")
        
        # 表4：危险等级性能
        f.write("\n\n" + "="*80 + "\n")
        f.write("论文表4：危险等级分类性能（XGBoost）\n")
        f.write("="*80 + "\n\n")
        
        pre_haz, rec_haz, f1_haz, _ = precision_recall_fscore_support(
            y_test_haz, haz_preds["XGBoost"], average=None, zero_division=0
        )
        hazard_labels = ["低危险", "中危险", "高危险"]
        
        f.write(f"{'等级':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}\n")
        f.write("-"*45 + "\n")
        for i in range(3):
            f.write(f"{hazard_labels[i]:<10} {pre_haz[i]:.3f}{'':<5} {rec_haz[i]:.3f}{'':<5} {f1_haz[i]:.3f}{'':<5}\n")
        
        # 表5：推理时间
        f.write("\n\n" + "="*80 + "\n")
        f.write("论文表5：模型推理时间（ms）\n")
        f.write("="*80 + "\n\n")
        
        N = 1000
        sample = X_test_cls[:1]
        f.write(f"{'模型':<10} {'时间(ms)':<10}\n")
        f.write("-"*25 + "\n")
        for name in ["KNN", "SVM", "RF", "XGBoost"]:
            model = cls_models[name]
            start = time.time()
            for _ in range(N):
                model.predict(sample)
            end = time.time()
            t = (end - start) / N * 1000
            f.write(f"{name:<10} {t:.3f}\n")
        
        # 显著性检验
        f.write("\n\n" + "="*80 + "\n")
        f.write("统计显著性检验结果（McNemar检验，α=0.05）\n")
        f.write("="*80 + "\n\n")
        
        def mcnemar_test(y_true, y_a, y_b):
            table = np.zeros((2,2), dtype=int)
            for t, a, b in zip(y_true, y_a, y_b):
                a_c = 1 if a == t else 0
                b_c = 1 if b == t else 0
                table[a_c, b_c] += 1
            res = mcnemar(table, exact=False)
            return res.pvalue
        
        p_knn = mcnemar_test(y_test_cls, cls_preds["XGBoost"], cls_preds["KNN"])
        p_svm = mcnemar_test(y_test_cls, cls_preds["XGBoost"], cls_preds["SVM"])
        
        f.write("--- 障碍物分类任务 ---\n")
        f.write(f"XGBoost vs KNN: p-value = {p_knn:.4f}")
        f.write(" (显著, p<0.05)\n" if p_knn < 0.05 else " (不显著)\n")
        f.write(f"XGBoost vs SVM: p-value = {p_svm:.4f}")
        f.write(" (显著, p<0.05)\n" if p_svm < 0.05 else " (不显著)\n")
        
        p_knn_h = mcnemar_test(y_test_haz, haz_preds["XGBoost"], haz_preds["KNN"])
        p_svm_h = mcnemar_test(y_test_haz, haz_preds["XGBoost"], haz_preds["SVM"])
        
        f.write("\n--- 危险等级评估任务 ---\n")
        f.write(f"XGBoost vs KNN: p-value = {p_knn_h:.4f}")
        f.write(" (显著, p<0.05)\n" if p_knn_h < 0.05 else " (不显著)\n")
        f.write(f"XGBoost vs SVM: p-value = {p_svm_h:.4f}")
        f.write(" (显著, p<0.05)\n" if p_svm_h < 0.05 else " (不显著)\n")
    
    print(f"\n✅ 所有论文图表数据已保存至: {result_file}")
    return result_file

# ====================== 主流程 ======================
if __name__ == "__main__":
    print("="*80)
    print("雷达障碍物分类与危险等级评估 - 实验复现")
    print("="*80 + "\n")
    
    # 1. 加载预处理好的数据集
    print("正在加载预处理数据集...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"找不到预处理数据集 {INPUT_CSV}！\n"
                                f"请先确保你已经下载了我们处理好的小数据集，或者运行 load_carrada.py 自己处理原始数据。")
    
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ 加载完成！共 {len(df)} 个有效样本")
    
    # 2. 计算危险等级
    df = calculate_hazard_level(df)
    
    # 3. 数据归一化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[["distance", "velocity", "angle", "rcs"]].values)
    y_cls = df["label"].values
    y_haz = df["hazard"].values
    
    # 4. 训练分类模型
    cls_cv, _, cls_models, cls_preds, X_test_cls, y_test_cls = train_and_evaluate(
        X, y_cls, "障碍物分类"
    )
    
    # 5. 训练危险等级模型
    haz_cv, _, haz_models, haz_preds, X_test_haz, y_test_haz = train_and_evaluate(
        X, y_haz, "危险等级评估"
    )
    
    # 6. 生成论文表格
    generate_paper_tables(
        df, 
        cls_cv, cls_models, cls_preds, X_test_cls, y_test_cls,
        haz_cv, haz_models, haz_preds, X_test_haz, y_test_haz
    )
    
    print("\n🎉 实验全部完成！你可以直接复制 results/paper_results.txt 到你的论文中！")
