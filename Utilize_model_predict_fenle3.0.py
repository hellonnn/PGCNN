import keras
import math
import numpy as np
from dataProcess import depth_data_process, data_process_pro, raindata_process, normalization, normalization_reverse, data_process
import pylab as plt
import datetime
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl
import os
import seaborn as sns
# from def_iin2 import gaussian_ablate        #后加的
plt.rcParams['font.family'] = 'Liberation Sans'   # 与 Arial 几乎一模一样

# ==================== 更新：定义水深分类阈值（与训练时一致）==================== #
DEPTH_THRESHOLDS = [0.15, 0.5]  # 分类阈值(米) - 必须与训练时一致
NUM_CLASSES = len(DEPTH_THRESHOLDS) + 2  # 现在有4类

# 定义类别颜色映射（用于可视化）
CLASS_COLORS = [
    [0.8, 0.8, 0.8, 1],    # 0类：灰色（不可能积水）
    [1, 1, 1, 1],           # 1类：白色（无水）
    [0.5, 0.8, 1, 1],       # 2类：浅蓝色（浅水）
    [0, 0.4, 1, 1],         # 3类：蓝色（深水）
]
CLASS_NAMES = [
    "NoFlood",  # 0类
    "Low",    # 1类
    f"Medium",  # 2类
    f"High"              # 3类
]

# 数据库存放路径
dem_path = "大区域数据/dem10.asc"
rain_path = "大区域数据/pra.asc"
junctn_path = "大区域数据/junctn10.asc"
slope_path = "大区域数据/slope10.asc"
aspect_path = "大区域数据/aspect10.asc"
curvatu_path = "大区域数据/curvatu10.asc"
building_path = "大区域数据/building10.asc"
pipe_path = "大区域数据/pipe10.asc"
valid_path = "rain_pr/Rain100Type6.asc"  # validSet数据

# 定义研究区的行列数
rowNum = 656
colNum = 650
scnNum = 6
RAIN_SCENARIO_INDEX = 5  # 手动设置需要预测的降雨场景索引（0~5）
rain_period = 12
patch_row_col_Num = 64
sample_per_scn = 5
sample_total = sample_per_scn * scnNum
inpFea_num = 7

# 创建存放数据的矩阵
dem = np.empty([rowNum, colNum], dtype=float)
junctn = np.empty([rowNum, colNum], dtype=float)
slope = np.empty([rowNum, colNum], dtype=float)
raw_rain_data = np.empty([scnNum, rain_period], dtype=float)
aspect = np.empty([rowNum, colNum], dtype=float)
curvatu = np.empty([rowNum, colNum], dtype=float)
building = np.empty([rowNum, colNum], dtype=float)
pipe = np.empty([rowNum, colNum], dtype=float)
validdepth = np.empty([rowNum, colNum], dtype=float)

# 读取数据
dem = data_process(dem_path)
raw_rain_data = raindata_process(rain_path)
junctn = data_process(junctn_path)
slope = data_process(slope_path)
aspect = data_process(aspect_path)
curvatu = data_process(curvatu_path)
building = data_process(building_path)
pipe = data_process(pipe_path)
validdepth = data_process_pro(valid_path)

# 原始数据预处理
np.place(dem, dem == -9999, np.max(dem))
np.place(slope, slope == -9999, 0)
np.place(junctn, junctn == 0, 1)
np.place(junctn, junctn == -9999, 0)
np.place(aspect, aspect == -9999, 0)
np.place(curvatu, curvatu == -9999, 0)
np.place(building, building == -9999, 0)
np.place(pipe, pipe == -9999, 0)

# 数据标准化
dem = normalization(dem)
rain_data = normalization(raw_rain_data)
slope = normalization(slope)
aspect = normalization(aspect)
curvatu = normalization(curvatu)
pipe = normalization(pipe)

# ==================== 更新：定义新的水深分类函数 ==================== #
def classify_depth(depth_data, impossible_mask):
    """将连续水深值转换为分类索引（4类）"""
    classified = np.zeros_like(depth_data, dtype=np.uint8)  # 默认0类：不可能积水
    
    # 只有在"可能积水"区域才分类
    valid = ~impossible_mask
    
    # 1类：0 < depth ≤ 0.15m
    mask1 = valid & (depth_data > 0) & (depth_data <= DEPTH_THRESHOLDS[0])
    classified[mask1] = 1
    
    # 2类：0.15m < depth ≤ 0.5m
    mask2 = valid & (depth_data > DEPTH_THRESHOLDS[0]) & (depth_data <= DEPTH_THRESHOLDS[1])
    classified[mask2] = 2
    
    # 3类：depth > 0.5m
    mask3 = valid & (depth_data > DEPTH_THRESHOLDS[1])
    classified[mask3] = 3
    
    return classified

# 创建不可能积水掩膜（建筑物或无效DEM区域）
impossible_mask = (building == 1) | (dem == -9999)

# 将验证集水深转换为分类索引
valid_class = classify_depth(validdepth, impossible_mask)

# 将整个研究区分成patch
patch_num_row = math.floor(rowNum / patch_row_col_Num)
patch_num_col = math.floor(colNum / patch_row_col_Num)

pred_dem_whole = np.empty([(patch_num_row + 1) * (patch_num_col + 1), patch_row_col_Num, patch_row_col_Num, inpFea_num], dtype=float)

# 填充主数据
for j in range(patch_num_row):
    for k in range(patch_num_col):
        start_row = j * patch_row_col_Num
        end_row = (j + 1) * patch_row_col_Num
        start_col = k * patch_row_col_Num
        end_col = (k + 1) * patch_row_col_Num
        
        pred_dem_whole[j * patch_num_col + k, :, :, 0] = dem[start_row:end_row, start_col:end_col]
        pred_dem_whole[j * patch_num_col + k, :, :, 1] = junctn[start_row:end_row, start_col:end_col]
        pred_dem_whole[j * patch_num_col + k, :, :, 2] = slope[start_row:end_row, start_col:end_col]
        pred_dem_whole[j * patch_num_col + k, :, :, 3] = aspect[start_row:end_row, start_col:end_col]
        pred_dem_whole[j * patch_num_col + k, :, :, 4] = curvatu[start_row:end_row, start_col:end_col]
        pred_dem_whole[j * patch_num_col + k, :, :, 5] = building[start_row:end_row, start_col:end_col]
        pred_dem_whole[j * patch_num_col + k, :, :, 6] = pipe[start_row:end_row, start_col:end_col]

# 补全右边和下边
remainder_col = colNum % patch_row_col_Num
remainder_row = rowNum % patch_row_col_Num

# 右侧补丁
for j in range(patch_num_row):
    start_row = j * patch_row_col_Num
    end_row = (j + 1) * patch_row_col_Num
    start_col = colNum - patch_row_col_Num
    end_col = colNum
    
    idx = patch_num_row * patch_num_col + j
    pred_dem_whole[idx, :, :, 0] = dem[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 1] = junctn[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 2] = slope[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 3] = aspect[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 4] = curvatu[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 5] = building[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 6] = pipe[start_row:end_row, start_col:end_col]

# 底部补丁
for k in range(patch_num_col):
    start_row = rowNum - patch_row_col_Num
    end_row = rowNum
    start_col = k * patch_row_col_Num
    end_col = (k + 1) * patch_row_col_Num
    
    idx = patch_num_row * (patch_num_col + 1) + k
    pred_dem_whole[idx, :, :, 0] = dem[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 1] = junctn[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 2] = slope[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 3] = aspect[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 4] = curvatu[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 5] = building[start_row:end_row, start_col:end_col]
    pred_dem_whole[idx, :, :, 6] = pipe[start_row:end_row, start_col:end_col]

# 右下角补丁
start_row = rowNum - patch_row_col_Num
end_row = rowNum
start_col = colNum - patch_row_col_Num
end_col = colNum
idx = (patch_num_row + 1) * (patch_num_col + 1) - 1
pred_dem_whole[idx, :, :, 0] = dem[start_row:end_row, start_col:end_col]
pred_dem_whole[idx, :, :, 1] = junctn[start_row:end_row, start_col:end_col]
pred_dem_whole[idx, :, :, 2] = slope[start_row:end_row, start_col:end_col]
pred_dem_whole[idx, :, :, 3] = aspect[start_row:end_row, start_col:end_col]
pred_dem_whole[idx, :, :, 4] = curvatu[start_row:end_row, start_col:end_col]
pred_dem_whole[idx, :, :, 5] = building[start_row:end_row, start_col:end_col]
pred_dem_whole[idx, :, :, 6] = pipe[start_row:end_row, start_col:end_col]

# 创建降雨输入数据
pred_rain_whole = np.empty([(patch_num_row + 1) * (patch_num_col + 1), rain_period], dtype=float)
for j in range((patch_num_row + 1) * (patch_num_col + 1)):
    pred_rain_whole[j, :] = rain_data[RAIN_SCENARIO_INDEX, :]  # 关键修改：通过索引选择场景

# 存储合成结果 - 现在存储分类索引
pred_class_whole = np.zeros([rowNum, colNum], dtype=np.uint8)

start = datetime.datetime.now()

# 在调用模型的脚本头部重新定义（或导入）这个函数
import tensorflow as tf

CLASS_WEIGHTS = tf.constant([1.0, 5.0, 10.0, 20.0], dtype=tf.float32)

def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1)
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False)
    weight_map = tf.gather(CLASS_WEIGHTS, y_true)
    return tf.reduce_mean(ce * weight_map)

model_path = 'result3/ALL/modelSaver_pgnn_classification_ok.h5'  # 注意使用4分类模型

# 加载模型时把函数塞到 custom_objects
new_model = keras.models.load_model(
    model_path,
    custom_objects={'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy}
)


# 预测
predictions = new_model.predict([pred_dem_whole, pred_rain_whole])

# 获取类别索引 (取概率最大的类别)
pred_classes = np.argmax(predictions, axis=-1)

# 合成整个预测区域
for j in range(patch_num_row):
    for k in range(patch_num_col):
        start_row = j * patch_row_col_Num
        end_row = (j + 1) * patch_row_col_Num
        start_col = k * patch_row_col_Num
        end_col = (k + 1) * patch_row_col_Num
        pred_class_whole[start_row:end_row, start_col:end_col] = pred_classes[j * patch_num_col + k]

# 右侧补丁
for j in range(patch_num_row):
    start_row = j * patch_row_col_Num
    end_row = (j + 1) * patch_row_col_Num
    start_col = colNum - patch_row_col_Num
    end_col = colNum
    idx = patch_num_row * patch_num_col + j
    pred_class_whole[start_row:end_row, start_col:end_col] = pred_classes[idx][:, :patch_row_col_Num]

# 底部补丁
for k in range(patch_num_col):
    start_row = rowNum - patch_row_col_Num
    end_row = rowNum
    start_col = k * patch_row_col_Num
    end_col = (k + 1) * patch_row_col_Num
    idx = patch_num_row * (patch_num_col + 1) + k
    pred_class_whole[start_row:end_row, start_col:end_col] = pred_classes[idx][:patch_row_col_Num, :]

# 右下角补丁
start_row = rowNum - patch_row_col_Num
end_row = rowNum
start_col = colNum - patch_row_col_Num
end_col = colNum
idx = (patch_num_row + 1) * (patch_num_col + 1) - 1
pred_class_whole[start_row:end_row, start_col:end_col] = pred_classes[idx][:patch_row_col_Num, :patch_row_col_Num]

end = datetime.datetime.now()
print(f"预测完成，用时: {(end - start).total_seconds():.2f}秒")

# 保存预测结果（分类索引）
np.savetxt(r'result4/测试/DHC_class_predict_4.txt', pred_class_whole, fmt='%d', delimiter=' ')                       #----------------------------------------------------------

# ==================== 更新：可视化4分类结果 ==================== #
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#"NoFlood","Low", "Medium","High"
def plot_classification_result(prediction, ground_truth, prefix):
    # ------------------------------------------------
    # 1. 手动设置颜色与类别名
    # ------------------------------------------------
    my_colors = [
        "#E7E7E7",  # 类别 0：
        "#aed5f5",  # 类别 1：
        "#615ff1",  # 类别 2：
        "#960ff0",  # 类别 3：
    ]
    my_names = ['NoFlood', 'Low', 'Medium', 'High']  
    cmap = ListedColormap(my_colors)
    bounds = [0, 1, 2, 3, 4]
    norm = BoundaryNorm(bounds, cmap.N)

    # ------------------------------------------------
    # 2. 画预测图（左侧 colorbar）
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(prediction, cmap=cmap, norm=norm)
    ax.set_title('Model Prediction(TEST4)', fontsize=18)           #--------------------------------------------------
    ax.tick_params(axis='both', labelsize=14)

    try:
        cbar = fig.colorbar(im, ax=ax, location='left',
                            shrink=1.0, pad=0.08, fraction=0.046)
    except TypeError:                         # 旧版 matplotlib
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('left', size='5%', pad=0.5)
        cbar = fig.colorbar(im, cax=cax)

    cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(my_names)
    plt.tight_layout()
    plt.savefig(f'{prefix}_pred.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------
    # 3. 画真实标签图（右侧 colorbar）
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(ground_truth, cmap=cmap, norm=norm)
    ax.set_title('Ground Truth(TEST4)', fontsize=18)                #--------------------------------------------------
    ax.tick_params(axis='both', labelsize=14)

    try:
        cbar = fig.colorbar(im, ax=ax, location='right',
                            shrink=1.0, pad=0.08, fraction=0.046)
    except TypeError:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.5)
        cbar = fig.colorbar(im, cax=cax)

    cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(my_names)
    plt.tight_layout()
    plt.savefig(f'{prefix}_gt.png', dpi=300, bbox_inches='tight')
    plt.close()

# 调用示例
plot_classification_result(
    pred_class_whole,
    valid_class,
    "result4/测试/classification_comparison_ok4"          #------------------------------------------
)
    
# ==================== 更新：计算4分类准确率 ==================== #
def calculate_class_accuracy(pred, true):
    """计算各类别准确率和总体准确率"""
    valid_mask = (true >= 0)  # 假设无效值为-1或类似
    pred_valid = pred[valid_mask]
    true_valid = true[valid_mask]
    
    overall_acc = np.mean(pred_valid == true_valid)
    
    class_acc = {}
    for i in range(NUM_CLASSES):
        mask = (true_valid == i)
        if np.sum(mask) > 0:
            class_acc[i] = np.mean(pred_valid[mask] == i)
        else:
            class_acc[i] = 0.0
    
    return overall_acc, class_acc


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def plot_diff_map_new(prediction, ground_truth, filename):
    """
    不通过减法，而是显式枚举 4×4 组合，把误差等级可视化
    0 = 完全一致
    1 = 相差 1 类
    2 = 相差 2 类
    3 = 相差 3 类
    """
    # 1. 预先生成 4×4 误差等级表（行列：真实→预测）
    #    0 1 2 3
    # 0  0 1 2 3
    # 1  1 0 1 2
    # 2  2 1 0 1
    # 3  3 2 1 0
    diff_table = np.array([[0, 1, 2, 3],
                           [1, 0, 1, 2],
                           [2, 1, 0, 1],
                           [3, 2, 1, 0]], dtype=np.uint8)

    # 2. 用矢量化方式查表：diff_map[i,j] = diff_table[gt[i,j], pred[i,j]]
    diff_map = diff_table[ground_truth, prediction]

    # 3. 保存 txt
    txt_path = os.path.splitext(filename)[0] + '.txt'
    np.savetxt(txt_path, diff_map, fmt='%d')

    # 4. 颜色映射
    diff_colors = ["#ffffff",   # 0 完全一致
                   "#96B11E",   # 1 相差 1 类
                   "#2440da",   # 2 相差 2 类
                   "#e21e1e"]   # 3 相差 3 类
    cmap_diff = mpl.colors.ListedColormap(diff_colors)
    bounds_diff = [0, 1, 2, 3, 4]
    norm_diff = mpl.colors.BoundaryNorm(bounds_diff, cmap_diff.N)

    # 5. 绘图
    plt.figure(figsize=(8, 8))
    im = plt.imshow(diff_map, cmap=cmap_diff, norm=norm_diff)

    # ===== 1. 标题字号调大 =====
    plt.title("Error Category Map (TEST4)", fontsize=16)   # 原来是 14   #----------------------------------------------

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(["Perfect match",
                         "One-class deviation",
                         "Two-class deviation",
                         "Three-class deviation"])

    # ===== 2. colorbar 刻度字号调大 =====
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(12)

    # ===== 3. 坐标轴刻度字号调大 =====
    plt.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# 新增差值图
plot_diff_map_new(pred_class_whole,
                  valid_class,
                  "result4/测试/error_map_ok4.png")                 #----------------------------------------------------------
# 计算并打印准确率
overall_acc, class_acc = calculate_class_accuracy(pred_class_whole, valid_class)

print(f"\n总体准确率: {overall_acc:.4f}")
for cls, acc in class_acc.items():
    print(f"类别 {cls} ({CLASS_NAMES[cls]}): {acc:.4f}")

# 保存准确率结果
with open('result4/测试/accuracy_report_ok4.txt', 'w') as f:         #----------------------------------------------------------
    plt.tight_layout()
    f.write("4分类模型性能报告\n")  # 更新报告标题
    f.write("="*50 + "\n")
    f.write(f"预测时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"使用模型: {model_path}\n")
    f.write(f"分类阈值: {DEPTH_THRESHOLDS}\n\n")
    f.write("类别定义:\n")  # 增加类别定义说明
    f.write(f"  0: 不可能积水区域 (建筑物/无效区域)\n")
    f.write(f"  1: 无水 (depth ≤ {DEPTH_THRESHOLDS[0]}m)\n")
    f.write(f"  2: 浅水 ({DEPTH_THRESHOLDS[0]}m < depth ≤ {DEPTH_THRESHOLDS[1]}m)\n")
    f.write(f"  3: 深水 (depth > {DEPTH_THRESHOLDS[1]}m)\n\n")
    f.write(f"总体准确率: {overall_acc:.4f}\n\n")
    f.write("各类别准确率:\n")
    for cls, acc in class_acc.items():
        f.write(f"  类别 {cls} ({CLASS_NAMES[cls]}): {acc:.4f}\n")
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(valid_class.flatten(), pred_class_whole.flatten(), labels=range(NUM_CLASSES))
    f.write("\n混淆矩阵 (4×4):\n")  # 注明矩阵维度
    f.write("\t" + "\t".join([f"预测{c}" for c in range(NUM_CLASSES)]) + "\n")
    for i in range(NUM_CLASSES):
        f.write(f"真实{i}\t" + "\t".join([str(x) for x in cm[i]]) + "\n")
    
    # 计算每个类别的百分比
    total_pixels = np.prod(valid_class.shape)
    f.write("\n类别分布:\n")
    for i in range(NUM_CLASSES):
        true_count = np.sum(valid_class == i)
        pred_count = np.sum(pred_class_whole == i)
        f.write(f"  类别 {i} ({CLASS_NAMES[i]}):\n")
        f.write(f"    真实占比: {true_count/total_pixels:.4f} ({true_count} 像素)\n")
        f.write(f"    预测占比: {pred_count/total_pixels:.4f} ({pred_count} 像素)\n")
    
    # 新增：计算并记录Kappa系数
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(valid_class.flatten(), pred_class_whole.flatten())
    f.write(f"\nKappa系数: {kappa:.4f}\n")
    
    # 新增：计算F1-score
    from sklearn.metrics import f1_score
    f1 = f1_score(valid_class.flatten(), pred_class_whole.flatten(), average='weighted')
    f.write(f"加权F1-score: {f1:.4f}\n")
    
    from sklearn.metrics import f1_score

    # 计算各类别 F1-score
    f1_per_class = f1_score(valid_class.flatten(), pred_class_whole.flatten(), average=None, labels=range(NUM_CLASSES))

    # 输出到文件
    for cls, f1_val in enumerate(f1_per_class):
        f.write(f"类别 {cls} ({CLASS_NAMES[cls]}) F1-score: {f1_val:.4f}\n")

    
    CLASS_NAMES2 = ["NoFlood","Low", "Medium","High"]

    # ---------- 百分比色标 ----------
    cm = confusion_matrix(valid_class.flatten(),
                          pred_class_whole.flatten(),
                          labels=range(NUM_CLASSES))
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 行归一化

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.4)
    sns.heatmap(cm_pct,
                annot=True, fmt='.2%',        # 显示百分比
                cmap='magma_r',              # 反向色带，低值浅，高值深
                vmin=0, vmax=1,
                xticklabels=CLASS_NAMES2,
                yticklabels=CLASS_NAMES2,
                cbar_kws={'label': 'Proportion'})
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix (TEST4)")                    #----------------------------------------------------------
    plt.tight_layout()      
    plt.savefig('result4/测试/confusion_matrix_pct4.png', dpi=300, bbox_inches='tight')      #----------------------------------------------------------
    plt.tight_layout()
    plt.close()

print("4分类结果已保存至 result3 目录") 