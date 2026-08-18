import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from def_iin2 import build_pgnn_regression_model

# 字体与绘图风格
font_path = 'Times New Roman.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
sns.set_theme(style="ticks", context="paper", font="Times New Roman")
# 参数设置与路径
DEPTH_THRESHOLDS = [0.15, 0.5]
CLASS_NAMES = ["NoFlood", "Low", "Medium", "High"]

# 空间特征 CSV 路径
base_dir = "CSV_大区域数据/处理"
dem_path = f"{base_dir}/dem10.csv"
junction_path = f"{base_dir}/junction10.csv"
slope_path = f"{base_dir}/slope10.csv"
aspect_path = f"{base_dir}/aspect10.csv"
curvature_path = f"{base_dir}/curvature10.csv"
building_path = f"{base_dir}/building10.csv"
pipe_path = f"{base_dir}/pipe10.csv"

# 单个场景的积水深度 CSV
RAIN_SCENARIO_NAME = "Rain100Type3"
valid_path = f"数据处理/测试/{RAIN_SCENARIO_NAME}.csv"

# 模型和参数路径 
MODEL_DIR = "Result_07/pgnn_strict"
model_path = f"{MODEL_DIR}/modelSaver_pgnn_regression_strict.h5"
norm_params_path = f"{MODEL_DIR}/norm_params.npy"

rowNum = 656
colNum = 650
rain_period = 12
patch_row_col_Num = 64
inpFea_num = 7

PATCH = patch_row_col_Num
STRIDE = patch_row_col_Num // 2
EPS = 1e-6

# 当前测试场景的降雨输入
rain_vec_raw = np.array([0.61,0.75,0.95,1.24,1.68,2.4,3.69,6.26,12.15,25.18,33.27,19.85], dtype=np.float32)

out_dir = 'Result_07/pgnn_strict'
os.makedirs(out_dir, exist_ok=True)

# 数据清洗与处理工具函数
def load_grid_csv(path):
    return pd.read_csv(path, header=None).values.astype(np.float32)

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
    return np.where((arr == -9999.0) | (arr == -9999), 0.0, arr)

def min_max_norm(arr):
    a_min, a_max = arr.min(), arr.max()
    return (arr - a_min) / (a_max - a_min + 1e-8)

def transform_rain_inference(rain_vec, params_path):
    rain_params = np.load(params_path, allow_pickle=True).item()
    rain_vec = np.maximum(rain_vec, 0.0)  # 保证非负
    transformed = np.arcsinh(rain_vec)
    return (transformed - rain_params['mean']) / rain_params['std']

def classify_depth(depth_data):
    classified = np.zeros_like(depth_data, dtype=np.uint8)
    mask1 = (depth_data > 0) & (depth_data <= DEPTH_THRESHOLDS[0])
    classified[mask1] = 1
    mask2 = (depth_data > DEPTH_THRESHOLDS[0]) & (depth_data <= DEPTH_THRESHOLDS[1])
    classified[mask2] = 2
    mask3 = depth_data > DEPTH_THRESHOLDS[1]
    classified[mask3] = 3
    return classified

def get_starts(total_len, patch_len, stride):
    starts = list(range(0, total_len - patch_len + 1, stride))
    if starts[-1] != total_len - patch_len:
        starts.append(total_len - patch_len)
    return starts

def set_nature_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', length=5, width=1.2)

print("开始加载并严格清洗空间特征...")
# 连续特征：清洗并归一化
dem = min_max_norm(clean_continuous_feature(load_grid_csv(dem_path), use_max=True))
slope = min_max_norm(clean_continuous_feature(load_grid_csv(slope_path)))
aspect = min_max_norm(clean_continuous_feature(load_grid_csv(aspect_path)))
curvatu = min_max_norm(clean_continuous_feature(load_grid_csv(curvature_path)))
pipe = min_max_norm(clean_continuous_feature(load_grid_csv(pipe_path)))
building = clean_indicator_feature(load_grid_csv(building_path))
junctn = clean_indicator_feature(load_grid_csv(junction_path))
validdepth = load_grid_csv(valid_path)
validdepth = clean_continuous_feature(validdepth, fill_value=0.0)
print("加载降雨参数并执行稳健变换...")
rain_vec_norm = transform_rain_inference(rain_vec_raw, norm_params_path)

# 生成滑窗 Patch (顺序与训练一致)
print("执行滑窗提取...")
row_starts = get_starts(rowNum, PATCH, STRIDE)
col_starts = get_starts(colNum, PATCH, STRIDE)
num_patches = len(row_starts) * len(col_starts)

pred_feat_whole = np.zeros((num_patches, PATCH, PATCH, inpFea_num), dtype=np.float32)
pred_rain_whole = np.zeros((num_patches, rain_period), dtype=np.float32)
patch_pos = []

idx = 0
for y in row_starts:
    for x in col_starts:
        pred_feat_whole[idx, :, :, 0] = dem[y:y + PATCH, x:x + PATCH]
        pred_feat_whole[idx, :, :, 1] = junctn[y:y + PATCH, x:x + PATCH]
        pred_feat_whole[idx, :, :, 2] = slope[y:y + PATCH, x:x + PATCH]
        pred_feat_whole[idx, :, :, 3] = aspect[y:y + PATCH, x:x + PATCH]
        pred_feat_whole[idx, :, :, 4] = curvatu[y:y + PATCH, x:x + PATCH]
        pred_feat_whole[idx, :, :, 5] = building[y:y + PATCH, x:x + PATCH]
        pred_feat_whole[idx, :, :, 6] = pipe[y:y + PATCH, x:x + PATCH]
        pred_rain_whole[idx, :] = rain_vec_norm
        patch_pos.append((y, x))
        idx += 1


# 加载模型并推理预测
print("构建模型架构...")
new_model = build_pgnn_regression_model(
    input_shape=(PATCH, PATCH, inpFea_num),
    rain_period=rain_period
)

print(f"加载模型权重: {model_path}")
new_model.load_weights(model_path)

start = datetime.datetime.now()
predictions = new_model.predict([pred_feat_whole, pred_rain_whole], batch_size=32, verbose=1)
predictions = np.squeeze(predictions, axis=-1) 

# 平滑拼接，消除 Patch 边界缝隙
win1d = np.hanning(PATCH).astype(np.float32)
win2d = np.outer(win1d, win1d).astype(np.float32)
win2d = np.maximum(win2d, EPS)

depth_sum = np.zeros((rowNum, colNum), dtype=np.float32)
weight_sum = np.zeros((rowNum, colNum), dtype=np.float32)

for i, (y, x) in enumerate(patch_pos):
    pred_patch = predictions[i]
    weighted_pred = pred_patch * win2d
    depth_sum[y:y + PATCH, x:x + PATCH] += weighted_pred
    weight_sum[y:y + PATCH, x:x + PATCH] += win2d

final_pred_depth = depth_sum / np.maximum(weight_sum, EPS)
final_pred_depth = np.maximum(final_pred_depth, 0.0)  # 防止出现负数水深

end = datetime.datetime.now()
print(f"推理完成! 耗时: {(end - start).total_seconds():.2f} 秒")

# 7) 保存最终连续水深
csv_path = os.path.join(out_dir, f'predicted_depth_map_{RAIN_SCENARIO_NAME}.csv')
np.savetxt(csv_path, final_pred_depth, fmt='%.4f', delimiter=',')
print(f"预测水深矩阵已保存至 {csv_path}")

# 评价图 1：回归散点密度图 (Hexbin)
valid_idx = np.isfinite(validdepth)
y_true_flat = validdepth[valid_idx]
y_pred_flat = final_pred_depth[valid_idx]
mask = (y_true_flat > 0.01)
y_true_flat = y_true_flat[mask]
y_pred_flat = y_pred_flat[mask]

if len(y_true_flat) < 2:
    print("有效样本过少，跳过散点图绘制。")
else:
    # 评价指标
    r2 = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)

    stats_text = (
        f"$R^2$ = {r2:.3f}\n"
        f"MAE = {mae:.3f} m\n"
        f"RMSE = {rmse:.3f} m"
    )
    g = sns.JointGrid(
        x=y_true_flat,
        y=y_pred_flat,
        height=6,
        space=0
    )
    # 主散点图
    sns.scatterplot(
        x=y_true_flat,
        y=y_pred_flat,
        s=4,
        alpha=0.35,
        color="#2a7886",
        edgecolor=None,
        ax=g.ax_joint
    )
    AXIS_MAX = 3.0 
    bin_width = 0.05 

    sns.histplot(
        y_true_flat,
        binwidth=bin_width,
        binrange=(0, AXIS_MAX),
        kde=True,
        color="#2a7886",
        alpha=0.35,
        ax=g.ax_marg_x
    )

    sns.histplot(
        y=y_pred_flat,
        binwidth=bin_width,
        binrange=(0, AXIS_MAX),
        kde=True,
        color="#2a7886",
        alpha=0.35,
        ax=g.ax_marg_y
    )

    g.ax_joint.plot(
        [0, AXIS_MAX],
        [0, AXIS_MAX],
        '--',
        color='black',
        lw=1.5,
        label='1:1 Line'
    )

    g.ax_joint.set_xlim(0, AXIS_MAX)
    g.ax_joint.set_ylim(0, AXIS_MAX)
    g.ax_joint.set_aspect('equal', adjustable='box')

    # Nature风格刻度
    g.ax_joint.set_xticks(np.arange(0, AXIS_MAX + 0.001, 0.5))
    g.ax_joint.set_yticks(np.arange(0, AXIS_MAX + 0.001, 0.5))

    # 指标文本
    g.ax_joint.text(
        0.05,
        0.95,
        stats_text,
        transform=g.ax_joint.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor='none',
            alpha=0.85
        )
    )

    # 标签
    g.ax_joint.set_xlabel(
        "Simulated Water Depth (m)",
        fontsize=13,
        fontweight="bold"
    )

    g.ax_joint.set_ylabel(
        "Predicted Water Depth (m)",
        fontsize=13,
        fontweight="bold"
    )
    sns.despine(ax=g.ax_joint)

    scatter_path = os.path.join(
        out_dir,
        f"regression_scatter_nature_{RAIN_SCENARIO_NAME}.png"
    )

    plt.savefig(
        scatter_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(f" Nature风格散点图已保存至 {scatter_path}")
    
    
    