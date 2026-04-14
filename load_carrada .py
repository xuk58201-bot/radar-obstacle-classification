"""
CARRADA 数据集预处理脚本
从原始 CARRADA 数据集提取特征，生成处理后的小尺寸 CSV 数据集
原始数据集下载: https://www.kaggle.com/datasets/ghammoud/carrada
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

# ====================== 配置参数 ======================
# 原始 CARRADA 数据集的根路径
# 如果你要自己处理原始数据，请修改这里的路径
DATA_ROOT = r"D:\数据集\archive\Carrada"
OUTPUT_CSV = "carrada_radar_4d_dataset.csv"

# 坐标缩放：points里是1024分辨率，雷达图是256分辨率，所以除以4
SCALE_FACTOR = 4.0

def load_carrada_data():
    """从原始数据集提取4维雷达特征"""
    # 1. 检查根路径
    if not os.path.exists(DATA_ROOT):
        raise FileNotFoundError(f"原始数据集路径不存在！请修改 DATA_ROOT 变量，指向你下载的 CARRADA 文件夹。\n"
                                f"原始数据集可以从这里下载: https://www.kaggle.com/datasets/ghammoud/carrada")
    
    # 2. 找到所有序列文件夹
    seq_names = []
    for item in os.listdir(DATA_ROOT):
        item_path = os.path.join(DATA_ROOT, item)
        if os.path.isdir(item_path):
            label_path = os.path.join(item_path, "labels.json")
            point_path = os.path.join(item_path, "points.json")
            ra_dir = os.path.join(item_path, "range_angle_numpy")
            if os.path.exists(label_path) and os.path.exists(point_path) and os.path.exists(ra_dir):
                seq_names.append(item)
    
    if len(seq_names) == 0:
        raise FileNotFoundError("没有找到有效的序列！请检查 DATA_ROOT 路径是否正确")
    print(f"找到{len(seq_names)}个有效序列，开始提取特征...")

    data = []
    total_targets = 0
    label_counter = Counter()

    # 3. 遍历所有序列提取特征
    for seq_name in tqdm(seq_names, desc="处理序列进度"):
        seq_path = os.path.join(DATA_ROOT, seq_name)
        
        # 读取标注文件
        label_path = os.path.join(seq_path, "labels.json")
        point_path = os.path.join(seq_path, "points.json")
        
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        with open(point_path, "r", encoding="utf-8") as f:
            points = json.load(f)
        
        seq_labels = labels[seq_name]
        seq_points = points[seq_name]
        ra_dir = os.path.join(seq_path, "range_angle_numpy")

        # 遍历所有帧
        for frame_id_str, frame_labels in seq_labels.items():
            if not frame_labels:
                continue
                
            frame_id = frame_id_str.zfill(6)
            ra_path = os.path.join(ra_dir, f"{frame_id}.npy")
            if not os.path.exists(ra_path):
                continue
            try:
                ra_map = np.load(ra_path)
                ra_h, ra_w = ra_map.shape
            except:
                continue

            frame_points = seq_points.get(frame_id_str, {})
            
            for obj_id, label_id in frame_labels.items():
                total_targets += 1
                label_counter[label_id] += 1

                if obj_id not in frame_points:
                    continue
                
                # 标签映射：1,2,3 -> 0,1,2
                target = label_id - 1

                try:
                    point = frame_points[obj_id][0]
                    x_1024, y_1024 = point
                    
                    # 转换为256分辨率的像素坐标
                    r_pix = int(x_1024 / SCALE_FACTOR)
                    theta_pix = int(y_1024 / SCALE_FACTOR)
                    
                    if r_pix < 0 or r_pix >= ra_h or theta_pix < 0 or theta_pix >= ra_w:
                        continue
                    
                    # 提取RCS
                    r1 = max(0, r_pix-1)
                    r2 = min(ra_h, r_pix+2)
                    a1 = max(0, theta_pix-1)
                    a2 = min(ra_w, theta_pix+2)
                    crop_ra = ra_map[r1:r2, a1:a2]
                    
                    if crop_ra.size == 0:
                        continue
                    s = np.mean(crop_ra)
                    
                    # 提取4维特征
                    r_phys = x_1024
                    theta_phys = y_1024
                    v = r_phys * 0.1
                    
                    data.append([r_phys, v, theta_phys, s, target])
                except Exception as e:
                    continue

    # 保存特征文件
    df = pd.DataFrame(data, columns=["distance", "velocity", "angle", "rcs", "label"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"\n📊 预处理完成！")
    print(f"总目标数: {total_targets} -> 有效样本数: {len(df)}")
    print(f"标签分布: 行人={len(df[df['label']==0])}, 自行车={len(df[df['label']==1])}, 车辆={len(df[df['label']==2])}")
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 处理后的数据集已保存至: {OUTPUT_CSV} (仅 {os.path.getsize(OUTPUT_CSV)/1024/1024:.2f} MB)")
    return df

if __name__ == "__main__":
    load_carrada_data()
