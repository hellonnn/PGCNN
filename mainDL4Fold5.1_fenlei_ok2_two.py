import tensorflow as tf
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from dataProcess import process_depth_data, process_dem_data, raindata_process, normalization, data_process, data_augmentation_depth, data_augmentation_main
import pylab as plt
from def_iin2_two import classify_depth, build_pgnn_classification_model, build_classification_model
import matplotlib
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import pandas as pd

matplotlib.use('Agg')  # 不依赖图形界面，只保存文件

# ==================== 水深分类配置 ====================
NUM_CLASSES = 2   # ✅ 二分类：0=无水，1=有水


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 数据路径配置
    dem_path = "大区域数据/dem10.asc"
    rain_path = "大区域数据/rain2.0.asc"
    junction_path = "大区域数据/junctn10.asc"
    slope_path = "大区域数据/slope10.asc"
    aspect_path = "大区域数据/aspect10.asc"
    curvatu_path = "大区域数据/curvatu10.asc"
    building_path = "大区域数据/building10.asc"
    pipe_path = "大区域数据/pipe10.asc"
    depth_folder = "rain_max2.0"
    
    # 区域配置
    rowNum, colNum = 656, 650
    scnNum, rain_period = 36, 12
    patch_row_col_Num = 64
    inpFea_num = 7
    
    # 初始化数据矩阵
    dem = np.empty([rowNum,colNum], dtype=float)
    junction = np.empty([rowNum,colNum], dtype=float)
    slope = np.empty([rowNum,colNum], dtype=float)
    depth = np.empty([scnNum,rowNum,colNum], dtype=float)
    raw_rain_data = np.empty([scnNum,rain_period], dtype=float)
    aspect = np.empty([rowNum,colNum], dtype=float)
    curvature = np.empty([rowNum,colNum], dtype=float)
    building = np.empty([rowNum,colNum], dtype=float)
    pipe = np.empty([rowNum,colNum], dtype=float)
    
    # 加载数据
    dem = process_dem_data(dem_path)
    raw_rain_data = raindata_process(rain_path)
    junctn = data_process(junction_path)
    slope = data_process(slope_path)
    aspect = data_process(aspect_path)
    curvature = data_process(curvatu_path)
    building = data_process(building_path)
    pipe = data_process(pipe_path)
    depth = process_depth_data(depth_folder)
    
    impossible_mask = (building == 1) | (dem == -9999)

    # ✅ 二分类版 classify_depth
    def classify_depth(depth_data, impossible_mask):
        classified = np.zeros_like(depth_data, dtype=np.uint8)  # 默认 0=无水
        valid = ~impossible_mask
        classified[valid & (depth_data > 0)] = 1                # 大于0的就是 1=有水
        return classified

    # 数据预处理
    depth_class = classify_depth(depth, impossible_mask)  # 水深二分类
    np.place(dem, dem==-9999, np.max(dem))
    np.place(depth, depth==-9999, 0)
    np.place(slope, slope==-9999, 0)
    np.place(junction, junction==0, 1)      
    np.place(junction, junction==-9999, 0)  
    np.place(aspect, aspect==-9999, 0)
    np.place(curvature, curvature==-9999, 0)
    np.place(building, building==-9999, 0)
    np.place(pipe, pipe==-9999, 0)
    
    # 数据标准化
    dem = normalization(dem)
    rain_data = normalization(raw_rain_data)
    slope = normalization(slope)
    aspect = normalization(aspect)
    curvature = normalization(curvature)
    pipe = normalization(pipe)

    # ---------------- 提前计算所有有效坐标 ----------------
    def build_coords(depth_vol, patch_size=64, stride=32):
        coords = []
        for j in range(depth_vol.shape[0]):
            d = depth_vol[j]
            for x in range(0, d.shape[0] - patch_size + 1, stride):
                for y in range(0, d.shape[1] - patch_size + 1, stride):
                    if np.any(d[x:x+patch_size, y:y+patch_size] != 0):
                        coords.append((j, x, y))
        return np.array(coords, dtype=np.int32)

    # ---------------- 构建 tf.data ----------------
    def make_dataset(j_list, x_list, y_list, depth_vol, covariates, rain_seq,
                    patch_size=64, batch_size=64,
                    shuffle_buffer=2000, shuffle=True, augment=True):
        depth_vol = tf.constant(depth_vol, tf.uint8)
        covariates = tf.constant(covariates, tf.float32)
        rain_seq = tf.constant(rain_seq, tf.float32)

        ds = tf.data.Dataset.from_tensor_slices((j_list, x_list, y_list))

        if shuffle:
            ds = ds.shuffle(shuffle_buffer)

        def _crop(j, x, y):
            x = tf.cast(x, tf.int32)
            y = tf.cast(y, tf.int32)

            depth_patch = depth_vol[j, x:x+patch_size, y:y+patch_size]
            cov_patch = covariates[x:x+patch_size, y:y+patch_size, :]
            rain_vec = rain_seq[j]

            depth_patch = tf.expand_dims(depth_patch, -1)

            if augment:
                if tf.random.uniform(()) > 0.5:
                    depth_patch = tf.image.flip_left_right(depth_patch)
                    cov_patch = tf.image.flip_left_right(cov_patch)
                if tf.random.uniform(()) > 0.5:
                    depth_patch = tf.image.flip_up_down(depth_patch)
                    cov_patch = tf.image.flip_up_down(cov_patch)
                k = tf.random.uniform((), 0, 4, dtype=tf.int32)
                depth_patch = tf.image.rot90(depth_patch, k)
                cov_patch = tf.image.rot90(cov_patch, k)

            return (cov_patch, rain_vec), depth_patch

        ds = ds.map(_crop, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # ---------------- 主程序 ----------------
    # ✅ 拼协变量
    covariates = np.stack([
        dem, junction, slope, aspect,
        curvature, building, pipe
    ], axis=-1).astype(np.float32)

    coords = build_coords(depth_class, patch_size=64, stride=32)
    coords_val = build_coords(depth_class, patch_size=64, stride=48)

    train_ds = make_dataset(coords[:,0], coords[:,1], coords[:,2],
                            depth_class, covariates, rain_data,
                            batch_size=32, shuffle=True, augment=True)

    val_ds = make_dataset(coords_val[:,0], coords_val[:,1], coords_val[:,2],
                        depth_class, covariates, rain_data,
                        batch_size=32, shuffle=False, augment=False)
    
    # 自定义回调：每10个epoch保存一次评估指标到CSV
    class MetricsLogger(tf.keras.callbacks.Callback):
        def __init__(self, val_data, save_dir, num_classes, save_interval=10):
            super().__init__()
            self.val_data = val_data
            self.save_dir = save_dir
            self.num_classes = num_classes
            self.save_interval = save_interval
            self.history_metrics = []
            os.makedirs(self.save_dir, exist_ok=True)

        def on_epoch_end(self, epoch, logs=None):
            y_true_all, y_pred_all = [], []
            for (cov_patch, rain_vec), y_true in self.val_data:
                preds = self.model.predict((cov_patch, rain_vec), verbose=0)
                preds = np.argmax(preds, axis=-1)
                y_true = y_true.numpy()[..., 0]
                y_true_all.append(y_true.flatten())
                y_pred_all.append(preds.flatten())

            y_true_all = np.concatenate(y_true_all)
            y_pred_all = np.concatenate(y_pred_all)

            acc = np.mean(y_true_all == y_pred_all)
            precision = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
            recall = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
            f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
            kappa = cohen_kappa_score(y_true_all, y_pred_all)

            metrics_row = {
                "epoch": epoch + 1,
                "val_acc": acc,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "val_kappa": kappa
            }
            self.history_metrics.append(metrics_row)

            print(f"\n[Epoch {epoch+1}] val_acc={acc:.4f} | precision={precision:.4f} | recall={recall:.4f} | f1={f1:.4f} | kappa={kappa:.4f}")

            if (epoch + 1) % self.save_interval == 0:
                df = pd.DataFrame(self.history_metrics)
                csv_path = os.path.join(self.save_dir, "metrics_log.csv")
                df.to_csv(csv_path, index=False)
                print(f"✅ Metrics saved to {csv_path}")

    # =============================
    # 构建 & 训练模型
    # =============================
    model = build_classification_model(
        input_shape=(patch_row_col_Num, patch_row_col_Num, inpFea_num),
        rain_period=rain_period
    )
    model.summary()

    metrics_logger = MetricsLogger(
        val_data=val_ds,
        save_dir='result4/测试',
        num_classes=NUM_CLASSES,
        save_interval=10
    )

    history = model.fit(
        train_ds,
        epochs=100,
        validation_data=val_ds,
        callbacks=[metrics_logger]
    )

    model.save('result4/测试/modelSaver_pgnn_classification_ok.h5')

    # 绘制训练曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'],  label='Training Accuracy')
    plt.plot(history.history['val_acc'],  label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('result4/测试/pgnn_classification_performance.png')
    plt.show()
