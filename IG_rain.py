import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tensorflow.keras.layers import (Input, Conv2D, Conv1D, Dense, BatchNormalization, 
                                     Activation, Multiply, Add, Concatenate, UpSampling2D, 
                                     MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, 
                                     Reshape, Lambda, Subtract, ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal

font_path = 'Times New Roman.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12

# 路径与参数设置
base_dir = "CSV_大区域数据/处理"
dem_path = f"{base_dir}/dem10.csv"
junction_path = f"{base_dir}/junction10.csv"
slope_path = f"{base_dir}/slope10.csv"
aspect_path = f"{base_dir}/aspect10.csv"
curvature_path = f"{base_dir}/curvature10.csv"
building_path = f"{base_dir}/building10.csv"
pipe_path = f"{base_dir}/pipe10.csv"

RAIN_SCENARIO_NAME = "Rain10Type7"
valid_path = f"数据处理/训练/{RAIN_SCENARIO_NAME}.csv"

MODEL_DIR = "Result_07/pgnn_strict"
model_path = f"{MODEL_DIR}/modelSaver_pgnn_regression_strict.h5"
norm_params_path = f"{MODEL_DIR}/norm_params.npy"

rain_period = 12
patch_size = 64
inpFea_num = 7
rain_vec_raw = np.array([0.46,0.64,0.96,1.63,3.41,9.82,17.16,7.5,4.03,4.99,12.48,15.64], dtype=np.float32)

os.makedirs(MODEL_DIR, exist_ok=True)
# 核心模型构建
def compute_dem_gradients(tensor):
    dy, dx = tf.image.image_gradients(tensor)
    return tf.concat([dx, dy], axis=-1)

def physics_interaction_module(dem, junction, slope, aspect, curvature, pipe):
    dem_grad = Lambda(compute_dem_gradients, name="dem_gradient")(dem)
    gather_inputs = Concatenate()([slope, curvature, dem_grad, aspect])
    gather_potential = Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer=he_normal())(gather_inputs)
    drain_inputs = Concatenate()([junction, pipe])
    drain_resistance = Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer=he_normal())(drain_inputs)
    net_risk = Subtract(name="mass_balance_sub")([gather_potential, drain_resistance])
    net_risk = ReLU(name="mass_balance_relu")(net_risk)
    phys_feat = Concatenate()([dem, net_risk, gather_potential, drain_resistance])
    phys_feat = Conv2D(64, (3,3), padding='same', activation='relu')(phys_feat)
    return phys_feat

def rainfall_temporal_extractor(rain_input):
    x = Reshape((rain_input.shape[1], 1))(rain_input)
    x = Conv1D(16, kernel_size=3, padding='same', activation='relu', kernel_initializer=he_normal())(x)
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer=he_normal())(x)
    seq_feat = GlobalAveragePooling1D()(x)
    accumulation = Lambda(lambda t: tf.reduce_sum(t, axis=1, keepdims=True), name="rain_accumulation")(rain_input)
    combined = Concatenate()([seq_feat, accumulation])
    rain_state = Dense(64, activation='relu', kernel_initializer=he_normal())(combined)
    return rain_state

def conv_block(x, filters):
    x = Conv2D(filters, (3,3), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3,3), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def aspp_module(x, filters):
    b0 = Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    b1 = Conv2D(filters, (3,3), padding='same', dilation_rate=2, activation='relu')(x)
    b2 = Conv2D(filters, (3,3), padding='same', dilation_rate=4, activation='relu')(x)
    shape_before = tf.shape(x)
    b3 = GlobalAveragePooling2D()(x)
    b3 = Reshape((1, 1, x.shape[-1]))(b3)
    b3 = Conv2D(filters, (1,1), padding='same', activation='relu')(b3)
    b3 = Lambda(lambda tensors: tf.image.resize(tensors[0], tensors[1][1:3]))([b3, shape_before])
    out = Concatenate()([b0, b1, b2, b3])
    return Conv2D(filters, (1,1), padding='same', activation='relu')(out)

def build_pgnn_regression_model(input_shape=(64, 64, 7), rain_period=12):
    spatial_input = Input(shape=input_shape, name="spatial_input")
    rain_input = Input(shape=(rain_period,), name="rain_input")
    
    dem       = Lambda(lambda x: x[..., 0:1], name="dem")(spatial_input)
    junction  = Lambda(lambda x: x[..., 1:2], name="junction")(spatial_input)
    slope     = Lambda(lambda x: x[..., 2:3], name="slope")(spatial_input)
    aspect    = Lambda(lambda x: x[..., 3:4], name="aspect")(spatial_input)
    curvature = Lambda(lambda x: x[..., 4:5], name="curvature")(spatial_input)
    building  = Lambda(lambda x: x[..., 5:6], name="building")(spatial_input)
    pipe      = Lambda(lambda x: x[..., 6:7], name="pipe")(spatial_input)
    
    phys_feat = physics_interaction_module(dem, junction, slope, aspect, curvature, pipe)
    rain_feat_vec = rainfall_temporal_extractor(rain_input)
    
    rain_broadcast = Lambda(
        lambda tensors: tf.tile(tf.reshape(tensors[0], [-1, 1, 1, 64]), 
                                [1, tf.shape(tensors[1])[1], tf.shape(tensors[1])[2], 1]),
        name="spatio_temporal_broadcast"
    )([rain_feat_vec, phys_feat])
    
    fused = Concatenate()([phys_feat, rain_broadcast])
    x = Conv2D(64, (3,3), padding='same', activation='relu')(fused)
    
    enc1 = conv_block(x, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(enc1)
    enc2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(enc2)
    bottleneck = aspp_module(pool2, 256)
    up2 = UpSampling2D(size=(2, 2))(bottleneck)
    concat2 = Concatenate()([up2, enc2])
    dec2 = conv_block(concat2, 128)
    up1 = UpSampling2D(size=(2, 2))(dec2)
    concat1 = Concatenate()([up1, enc1])
    dec1 = conv_block(concat1, 64)
    
    prob_mask = Conv2D(1, (1, 1), activation='sigmoid', name="prob_mask")(dec1)
    raw_depth = Conv2D(1, (1, 1), activation='softplus', name="raw_depth")(dec1)
    gated_depth = Multiply(name="gated_depth")([prob_mask, raw_depth])
    
    inv_building_mask = Lambda(lambda x: tf.cast(1.0, x.dtype) - x, name="inv_building_mask")(building)
    final_output = Multiply(name="depth_output")([gated_depth, inv_building_mask])
    
    return Model(inputs=[spatial_input, rain_input], outputs=final_output)

# 数据加载与局部 Patch 提取
def load_grid_csv(path):
    return pd.read_csv(path, header=None).values.astype(np.float32)

def clean_continuous(arr, fill_value=0.0, use_max=False):
    arr = arr.astype(np.float32)
    valid_mask = (arr != -9999.0) & (arr != -9999)
    fill_val = np.nanmax(np.where(valid_mask, arr, np.nan)) if use_max else fill_value
    return np.where(valid_mask, arr, fill_val)

def clean_indicator(arr):
    return np.where((arr == -9999.0) | (arr == -9999), 0.0, arr.astype(np.float32))

def min_max_norm(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

print("加载真值数据以定位高风险积水区域...")
validdepth = clean_continuous(load_grid_csv(valid_path), fill_value=0.0)

# 切出完整的 64x64 patch
max_y, max_x = np.unravel_index(np.argmax(validdepth), validdepth.shape)
start_y = max(0, min(max_y - patch_size // 2, validdepth.shape[0] - patch_size))
start_x = max(0, min(max_x - patch_size // 2, validdepth.shape[1] - patch_size))
print(f"锁定高风险区域坐标: 起点 ({start_y}, {start_x})")
dem = min_max_norm(clean_continuous(load_grid_csv(dem_path), use_max=True))[start_y:start_y+patch_size, start_x:start_x+patch_size]
junctn = clean_indicator(load_grid_csv(junction_path))[start_y:start_y+patch_size, start_x:start_x+patch_size]
slope = min_max_norm(clean_continuous(load_grid_csv(slope_path)))[start_y:start_y+patch_size, start_x:start_x+patch_size]
aspect = min_max_norm(clean_continuous(load_grid_csv(aspect_path)))[start_y:start_y+patch_size, start_x:start_x+patch_size]
curvatu = min_max_norm(clean_continuous(load_grid_csv(curvature_path)))[start_y:start_y+patch_size, start_x:start_x+patch_size]
building = clean_indicator(load_grid_csv(building_path))[start_y:start_y+patch_size, start_x:start_x+patch_size]
pipe = min_max_norm(clean_continuous(load_grid_csv(pipe_path)))[start_y:start_y+patch_size, start_x:start_x+patch_size]

spatial_patch = np.stack([dem, junctn, slope, aspect, curvatu, building, pipe], axis=-1)
spatial_patch = np.expand_dims(spatial_patch, axis=0) # [1, 64, 64, 7]

# 降雨处理 (真实降雨及零降雨基线)
rain_params = np.load(norm_params_path, allow_pickle=True).item()
# 真实降雨归一化
rain_transformed = np.arcsinh(np.maximum(rain_vec_raw, 0.0))
rain_vec_norm = (rain_transformed - rain_params['mean']) / rain_params['std']
rain_patch = np.expand_dims(rain_vec_norm, axis=0) # [1, 12]
# 无雨基线归一化：降雨量为0时的归一化值
baseline_transformed = np.arcsinh(0.0)
rain_baseline_norm = (baseline_transformed - rain_params['mean']) / rain_params['std']
baseline_patch = np.ones((1, rain_period), dtype=np.float32) * rain_baseline_norm

# 积分梯度法 (Integrated Gradients) 核心计算
print("加载模型并计算积分梯度 (Integrated Gradients)...")
model = build_pgnn_regression_model()
model.load_weights(model_path)

def compute_integrated_gradients(model, spatial_input, rain_input, baseline_input, steps=50):
    # 生成积分路径
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
    alphas = tf.cast(alphas, tf.float32)
    # 沿着路径生成一系列插值的降雨输入
    rain_diff = rain_input - baseline_input
    alphas_reshaped = tf.reshape(alphas, [-1, 1])
    # [steps+1, 12] 的插值矩阵
    interpolated_rain = baseline_input + alphas_reshaped * rain_diff
    # 空间特征复制 steps+1 份以匹配 Batch Size
    repeated_spatial = tf.tile(spatial_input, [steps+1, 1, 1, 1])
    # 计算各个插值点关于输入的梯度
    with tf.GradientTape() as tape:
        tape.watch(interpolated_rain)
        # 前向传播
        predictions = model([repeated_spatial, interpolated_rain])
        # 目标损失：整个 Patch 预测水深的总体积
        loss = tf.reduce_sum(predictions, axis=(1, 2, 3))
    # 获取预测结果对于降雨输入的梯度 (shape: [steps+1, 12])
    grads = tape.gradient(loss, interpolated_rain)
    avg_grads = tf.reduce_mean(grads[:-1], axis=0, keepdims=True)
    
    # 计算 IG 得分： (输入 - 基线) * 平均梯度
    integrated_gradients = rain_diff * avg_grads
    
    # 取其绝对贡献，并归一化到 0-1 用于绘图对比
    ig_scores = tf.abs(integrated_gradients)
    if tf.reduce_max(ig_scores) != 0:
        ig_scores /= tf.reduce_max(ig_scores)
        
    return ig_scores.numpy()[0]

# 计算 IG 权重
ig_heatmap = compute_integrated_gradients(model, spatial_patch, rain_patch, baseline_patch, steps=50)
print("生成可视化报告...")

plt.rcParams['font.size'] = 16  
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in' 
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

time_steps = np.arange(1, rain_period + 1)
fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=600)

color_rain = '#4C72B0'       
color_rain_edge = '#354F7A'  
color_ig = '#C44E52'        

# 背景热力图映射
heatmap_2d = np.expand_dims(ig_heatmap, axis=0)
ax1.imshow(heatmap_2d, aspect='auto', cmap='Reds', alpha=0.15,
           extent=[0.5, rain_period + 0.5, 0, np.max(rain_vec_raw) * 1.25])

# 左轴：原始降雨量
ax1.bar(time_steps, rain_vec_raw, width=0.6, color=color_rain, 
        edgecolor=color_rain_edge, linewidth=1.2, label='Rainfall (mm)', zorder=2)
ax1.set_xlabel('Time Step', fontsize=14, fontweight='normal', labelpad=8)
ax1.set_ylabel('Rainfall Intensity (mm)', fontsize=14, color=color_rain, fontweight='normal', labelpad=8)
ax1.tick_params(axis='y', labelcolor=color_rain, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_ylim(0, np.max(rain_vec_raw) * 1.25)
ax1.set_xticks(time_steps)

# 右轴：IG 贡献得分曲线
ax2 = ax1.twinx()
ax2.plot(time_steps, ig_heatmap, color=color_ig, marker='o', 
         linestyle='-', linewidth=2.5, markersize=8, 
         markeredgecolor='white', markeredgewidth=1.5,
         label='Attribution Score (IG)', zorder=3)
ax2.set_ylabel('IG Attribution Score', fontsize=14, color=color_ig, fontweight='normal', labelpad=8)
ax2.tick_params(axis='y', labelcolor=color_ig, labelsize=12)
ax2.set_ylim(0, 1.1)

# 边框、图例与细节修饰
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.spines['left'].set_linewidth(1.2)
ax2.spines['right'].set_linewidth(1.2)
plt.title("Rainfall Temporal Attribution (Integrated Gradients)", 
          fontsize=15, pad=15, fontweight='bold', color='#222222')
plt.tight_layout()

save_path_png = os.path.join(MODEL_DIR, f"Rain_Attribution_IG_{RAIN_SCENARIO_NAME}.png")
plt.savefig(save_path_png, bbox_inches='tight')
plt.close()

print(f"高级版图表已保存至:\nPNG: {save_path_png}")