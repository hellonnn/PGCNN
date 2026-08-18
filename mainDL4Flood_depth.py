import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from def_iin2 import build_pgnn_regression_model

# 数据读取与基础预处理工具
def load_grid_csv(path):
    return pd.read_csv(path, header=None).values.astype(np.float32)

def load_rain_data(path):
    df = pd.read_csv(path, header=None)
    scene_names = df.iloc[:, 0].astype(str).tolist()
    rain_array = df.iloc[:, 1:].values.astype(np.float32)
    rain_array = np.maximum(rain_array, 0.0)
    return scene_names, rain_array

# 严谨的数据清洗与转换核心逻辑
def clean_continuous_feature(arr, fill_value=0.0, use_max=False):
    arr = arr.astype(np.float32)
    valid_mask = (arr != -9999.0) & (arr != -9999)
    
    if use_max:
        fill_val = np.nanmax(np.where(valid_mask, arr, np.nan))
    else:
        fill_val = fill_value
        
    return np.where(valid_mask, arr, fill_val)

def clean_indicator_feature(arr):
    arr = arr.astype(np.float32)
    # 首先将所有 -9999 转换为 0 (没有设施)
    arr = np.where((arr == -9999.0) | (arr == -9999), 0.0, arr)
    return arr

def min_max_norm(arr):
    #最大最小值归一化
    a_min, a_max = arr.min(), arr.max()
    return (arr - a_min) / (a_max - a_min + 1e-8)

#降雨数据处理
def fit_and_transform_rain(rain_matrix):
    transformed = np.arcsinh(rain_matrix)
    mean_val = np.mean(transformed)
    std_val = np.std(transformed) + 1e-8
    
    norm_rain = (transformed - mean_val) / std_val
    rain_params = {'mean': mean_val, 'std': std_val}
    return norm_rain, rain_params

# 背景软过滤Patch提取器
def build_coords_with_background(depth_vol, scene_indices, patch_size=64, stride=24):
    coords = []
    for j in scene_indices:
        d = depth_vol[j]
        h, w = d.shape
        for x in range(0, h - patch_size + 1, stride):
            for y in range(0, w - patch_size + 1, stride):
                patch_depth = d[x:x+patch_size, y:y+patch_size]
                
                # 核心筛选：积水深度>0.01 或者是 5% 的随机背景
                if np.any(patch_depth > 0.01) or np.random.random() < 0.05:
                    coords.append((j, x, y))
                    
    return np.array(coords, dtype=np.int32)

def make_dataset(coords, depth_vol, covariates, rain_seq, patch_size, batch_size, shuffle=True):
    depth_vol = tf.constant(depth_vol, tf.float32)
    covariates = tf.constant(covariates, tf.float32)
    rain_seq = tf.constant(rain_seq, tf.float32)

    j_list, x_list, y_list = coords[:, 0], coords[:, 1], coords[:, 2]
    ds = tf.data.Dataset.from_tensor_slices((j_list, x_list, y_list))
    if shuffle: 
        ds = ds.shuffle(3000)

    def _crop_and_augment(j, x, y):
        j, x, y = tf.cast(j, tf.int32), tf.cast(x, tf.int32), tf.cast(y, tf.int32)
        depth_patch = depth_vol[j, x:x + patch_size, y:y + patch_size]
        cov_patch = covariates[x:x + patch_size, y:y + patch_size, :]
        rain_vec = rain_seq[j]
        depth_patch = tf.expand_dims(depth_patch, -1)
        return (cov_patch, rain_vec), depth_patch

    ds = ds.map(_crop_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def depth_weighted_mse(y_true, y_pred):
    # 计算基础的平方误差
    sq_diff = tf.square(y_true - y_pred)
    
    # 生成权重矩阵
    condition_shallow = tf.logical_and(y_true > 0.0, y_true <= 0.15)
    condition_deep = y_true > 0.15
    weights = tf.where(condition_shallow, 10.0, 1.0)
    weights = tf.where(condition_deep, 2.0, weights)
    
    miss_penalty = tf.where(tf.logical_and(y_true > 0.05, y_pred < 0.01), 2.0, 1.0)
    weighted_loss = sq_diff * weights * miss_penalty
    
    return tf.reduce_mean(weighted_loss)

# 主程序
if __name__ == "__main__":
    # 路径配置
    base_dir = "CSV_大区域数据"
    dem_path = f"{base_dir}/处理/dem10.csv"
    slope_path = f"{base_dir}/处理/slope10.csv"
    aspect_path = f"{base_dir}/处理/aspect10.csv"
    curvature_path = f"{base_dir}/处理/curvature10.csv"
    pipe_path = f"{base_dir}/处理/pipe10.csv"
    building_path = f"{base_dir}/处理/building10.csv"
    junction_path = f"{base_dir}/处理/junction10.csv"
    
    rain_path = "数据处理/rain_train.csv"
    depth_folder = "数据处理/训练"

    # 读取并清洗数据
    print("加载并清洗空间数据...")
    # 连续特征：清洗并归一化
    dem = min_max_norm(clean_continuous_feature(load_grid_csv(dem_path), use_max=True))
    slope = min_max_norm(clean_continuous_feature(load_grid_csv(slope_path)))
    aspect = min_max_norm(clean_continuous_feature(load_grid_csv(aspect_path)))
    curvature = min_max_norm(clean_continuous_feature(load_grid_csv(curvature_path)))
    pipe = min_max_norm(clean_continuous_feature(load_grid_csv(pipe_path)))
    building = clean_indicator_feature(load_grid_csv(building_path))
    junction = clean_indicator_feature(load_grid_csv(junction_path))

    # 组合为多通道输入张量 (H, W, C=7)
    covariates = np.stack([dem, junction, slope, aspect, curvature, building, pipe], axis=-1).astype(np.float32)

    # 读取降雨与结果标签
    print("加载时空降雨目标数据...")
    scene_names, rain_raw = load_rain_data(rain_path)
    
    # 获取有效目标数据
    depth_dict = {}
    for f in glob.glob(f"{depth_folder}/*.csv"):
        name = os.path.splitext(os.path.basename(f))[0]
        depth_dict[name] = pd.read_csv(f, header=None).values.astype(np.float32)

    valid_scenes = [s for s in scene_names if s in depth_dict]
    scene_to_idx = {s: i for i, s in enumerate(scene_names)}
    valid_indices = [scene_to_idx[s] for s in valid_scenes]

    # 对齐后提取数据
    rain_data_aligned = rain_raw[valid_indices]
    depth = np.stack([depth_dict[s] for s in valid_scenes], axis=0)
    
    # 降雨变换
    rain_data_norm, rain_params = fit_and_transform_rain(rain_data_aligned)
    print(f"降雨归一化参数: {rain_params}")

    # 划分数据集与 Patch 提取
    indices = np.arange(depth.shape[0])
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    print("执行含 5% 背景负样本的高效特征切片...")
    patch_size = 64
    train_coords = build_coords_with_background(depth, train_idx, patch_size, stride=24)
    val_coords = build_coords_with_background(depth, val_idx, patch_size, stride=32)
    print(f"训练 Patch 数量: {len(train_coords)}, 验证 Patch 数量: {len(val_coords)}")

    train_ds = make_dataset(train_coords, depth, covariates, rain_data_norm, patch_size, batch_size=32, shuffle=True)
    val_ds = make_dataset(val_coords, depth, covariates, rain_data_norm, patch_size, batch_size=32, shuffle=False)

    # 构建与训练模型
    model = build_pgnn_regression_model(
        input_shape=(patch_size, patch_size, 7),
        rain_period=rain_data_norm.shape[1]
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                loss=depth_weighted_mse,
                metrics=['mae', 'mse'])

    save_dir = "Result_07/pgnn_strict"
    os.makedirs(save_dir, exist_ok=True)
    
    print("开始模型训练...")
    history = model.fit(
        train_ds,
        epochs=100,
        validation_data=val_ds
    )

    # 保存模型与归一化参数，供推理时使用
    model.save(f"{save_dir}/modelSaver_pgnn_regression_strict.h5")
    np.save(f"{save_dir}/norm_params.npy", rain_params)
    print(f"模型及降雨反推参数已保存至 {save_dir}")