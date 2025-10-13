from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, Dense, BatchNormalization, AveragePooling2D,
                                   Activation, Multiply, Add, Concatenate, UpSampling2D, 
                                   GlobalAveragePooling2D, Reshape, Dropout, Lambda,Subtract, ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal
import tensorflow as tf
import numpy as np
# ==================== 水深分类配置 ====================
DEPTH_THRESHOLDS = [0.15, 0.5]  # 分类阈值(米)
NUM_CLASSES = len(DEPTH_THRESHOLDS) + 2
# 统计训练集像素类别分布后，手动或自动设置
# 例：0:1、1:2、2:10、3:20
CLASS_WEIGHTS = tf.constant([1.0, 5.0, 10.0, 20.0], dtype=tf.float32)

def weighted_sparse_categorical_crossentropy(y_true, y_pred):
    """
    y_true : (B, H, W, 1)  uint8/int32  取值 0~3
    y_pred : (B, H, W, 4)  softmax 概率
    """
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1)      # (B, H, W)

    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False)    # (B, H, W)

    # 按类别取权重
    weight_map = tf.gather(CLASS_WEIGHTS, y_true)  # (B, H, W)
    weighted_ce = ce * weight_map
    return tf.reduce_mean(weighted_ce)

def classify_depth(depth_data, impossible_mask):
    """
    depth_data      : 原始水深 (row, col) 或 (scn, row, col)
    impossible_mask : 与 depth_data 同形状的 0/1 掩膜，1 表示该像素不可能积水
    返回             : 分类索引数组，同形状
    """
    depth_data = np.asarray(depth_data)
    impossible_mask = np.asarray(impossible_mask, dtype=bool)
    
    if depth_data.ndim == 2:
        depth_data = depth_data[np.newaxis, ...]   # 变成 (1, row, col)
        impossible_mask = impossible_mask[np.newaxis, ...]
    
    classified = np.full(depth_data.shape, 0, dtype=np.uint8)  # 先默认 0：不可能积水
    
    # 只有在“可能积水”区域才继续细分
    valid = ~impossible_mask
    
    # 0.15 以下
    mask1 = valid & (depth_data > 0) & (depth_data <= 0.15)
    classified[mask1] = 1
    
    # 0.15-0.5
    mask2 = valid & (depth_data > 0.15) & (depth_data <= 0.5)
    classified[mask2] = 2
    
    # >0.5
    mask3 = valid & (depth_data > 0.5)
    classified[mask3] = 3
    
    return classified.squeeze()   # 把多余的第一维去掉


# ==================== PGNN模型组件 ====================
def channel_attention(input_feature, ratio=4):
    """通道注意力机制"""
    channel = input_feature.shape[-1]
    shared_layer_one = Dense(channel//ratio, activation='relu', kernel_initializer=he_normal())
    shared_layer_two = Dense(channel, kernel_initializer=he_normal())
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return Multiply()([input_feature, cbam_feature])

def enhanced_separable_block(x, filters, kernel_size, strides=1, physical_guidance=None):
    """增强的深度可分离卷积块"""
    shortcut = x
    # 深度可分离卷积
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, 
                       padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 点卷积
    x = Conv2D(filters, (1,1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # 通道注意力
    x = channel_attention(x)
    # 物理引导
    if physical_guidance is not None:
        phys_gate = Conv2D(filters, (1,1), activation='sigmoid')(physical_guidance)
        x = Multiply()([x, phys_gate])
    # 残差连接
    if strides == 1 and shortcut.shape[-1] == filters:
        x = Add()([x, shortcut])
    return Activation('relu')(x)
#-------------------原始PGNN------------------------------------------------
def physical_constraints_module(dem, slope, aspect, curvature, building, pipe, junction):
    """物理约束处理模块"""
    # 标准化输入
    dem = BatchNormalization()(dem)
    slope = BatchNormalization()(slope)
    aspect = BatchNormalization()(aspect)
    curvature = BatchNormalization()(curvature)
    building = BatchNormalization()(building)
    pipe = BatchNormalization()(pipe)
    junction = BatchNormalization()(junction)
    
    # 计算物理特征
    #汇流累积量，计算每个像素“上游有多少像素最终会流到我这儿”
    flow_accumulation = Conv2D(16, (3,3), padding='same', activation='relu')(dem)
    flow_accumulation = Conv2D(16, (3,3), padding='same', activation='relu')(flow_accumulation)
    #3×3 的卷积对这一片3x3的区域进行分析，来找出水会往哪里流
    #然后第二层卷积再扫一遍，来确定水的流向，把他合成一条水沟
    
    # 交汇点和管网的交互作用
    # junction_effect = Multiply()([junction, pipe])  
    # junction_effect = Conv2D(16, (3,3), padding='same', activation='relu')(junction_effect)
    #同时研究某一区域的检查井和管道直径，去找既重要又有大管的地方
    #然后使用3×3卷积进行操作，因为卷积是滑动的，那么他向外一抹就能模拟“排水能力向外扩散”的效果。
    #交汇点密度 × 管径大小 = 排水潜力
    
    # 加入 aspect & curvature
    aspect_feat   = Conv2D(8, (3,3), padding='same', activation='relu')(aspect)
    curv_feat     = Conv2D(8, (3,3), padding='same', activation='relu')(curvature)
    terrain_inflow = Concatenate(axis=-1)([flow_accumulation, aspect_feat, curv_feat])  # (B,H,W,32)
    terrain_inflow = Conv2D(16, (1,1), padding='same', activation='relu')(terrain_inflow)
    #aspect_feat：坡向决定水流汇聚方向，某些坡向更容易把水“导向”井口。
    #curv_feat：曲率表征地形凹凸，凹面更易积水。
    
    # 建筑物阻挡水流
    terrain_building = Multiply()([dem, building]) 
    terrain_building = Conv2D(16, (3,3), padding='same', activation='relu')(terrain_building)
    #dem和有建筑物的地方相乘，相当于“房子把自然地形削平，水在这儿流不动”。
    #用3×3卷积进行操作，让“被房子挡住”的影响向外扩散一点，模拟“水绕房走”的效果。
    
    # 管网在坡度大的区域排水更强
    pipe_capacity = Multiply()([pipe, slope])
    #坡度越陡、管子越粗，排水能力越强   
    # pipe_capacity = Add()([pipe_capacity, junction_effect])  
    # 加入junction影响，“井盖多”的地方再额外加分。
    pipe_capacity = Conv2D(16, (3,3), padding='same', activation='relu')(pipe_capacity)
    #让“强排水能力”的区域向外扩散，形成连续的“排水主干道”
    
    # junction 为井口掩膜，乘上汇水潜力 → 井口附近汇水量
    junction_inflow = Multiply()([flow_accumulation, junction])   # (B,H,W,16)
    # 汇水量 − 排水能力，负值截断为 0
    net_risk = Subtract()([junction_inflow, pipe_capacity])
    net_risk = ReLU()(net_risk)                                 # (B,H,W,16)
    junction_effect = net_risk    
    
    # 合并特征
    physical_features = Concatenate(axis=-1)([
        flow_accumulation,terrain_inflow,terrain_building,pipe_capacity,junction_effect
    ])
    physical_features = Conv2D(64, (3,3), padding='same', activation='relu')(physical_features)
    return BatchNormalization()(physical_features)



# 以下所有网络结构、build_pgnn_classification_model 均无需改动
# 只要保证训练脚本 import 的是上面这份带开关的 physical_constraints_module 即可

def build_pgnn_classification_model(input_shape=(64,64,7), rain_period=12):
    """构建PGNN分类模型（改进版）"""
    # 多物理输入
    main_input = Input(shape=input_shape, name='main_input')
    dem = Lambda(lambda x: x[:,:,:,0:1])(main_input)  # 提取DEM
    junction = Lambda(lambda x: x[:,:,:,1:2])(main_input)  # 提取junction
    slope = Lambda(lambda x: x[:,:,:,2:3])(main_input)  # 提取坡度
    aspect = Lambda(lambda x: x[:,:,:,3:4])(main_input)  # 提取坡向
    curvature = Lambda(lambda x: x[:,:,:,4:5])(main_input)  # 提取曲率
    building = Lambda(lambda x: x[:,:,:,5:6])(main_input)  # 提取建筑物
    pipe = Lambda(lambda x: x[:,:,:,6:7])(main_input)  # 提取管网
    rain_input = Input(shape=(rain_period,), name='rain_input')
    
    # 降雨数据处理
    rainfall = Dense(64, activation='relu')(rain_input)
    rainfall = Dropout(0.3)(rainfall)
    rainfall = Reshape([1,1,64])(rainfall)
    rainfall = UpSampling2D(size=input_shape[:2])(rainfall)

    # 物理特征和降雨特征融合
    # x = physical_constraints_module(dem, slope, aspect, curvature, building, pipe, junction)   # 64 通道
    # ========== 1️⃣ 第一次 PGNN（原分辨率 64×64） ==========
    phy_64 = physical_constraints_module(dem, slope, aspect, curvature, building, pipe, junction)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(main_input)
    x = Concatenate(axis=-1)([x, phy_64])              # 融合第一次 PGNN
    x = Conv2D(64, (1,1), padding='same')(x)           # 统一通道

    # ========== 2️⃣ 下采样到 32×32，第二次 PGNN ==========
    x = enhanced_separable_block(x, 64, 3, strides=2)  # 32×32
    dem_32 = AveragePooling2D((2,2))(dem)
    slope_32 = AveragePooling2D((2,2))(slope)
    aspect_32 = AveragePooling2D((2,2))(aspect)
    curvature_32 = AveragePooling2D((2,2))(curvature)
    building_32 = AveragePooling2D((2,2))(building)
    pipe_32 = AveragePooling2D((2,2))(pipe)
    junction_32 = AveragePooling2D((2,2))(junction)
    phy_32 = physical_constraints_module(
        dem_32, slope_32, aspect_32, curvature_32, building_32, pipe_32, junction_32
    )
    x = Concatenate(axis=-1)([x, phy_32])
    x = Conv2D(128, (1,1), padding='same')(x)

    # ========== 3️⃣ 下采样到 16×16，第三次 PGNN ==========
    x = enhanced_separable_block(x, 128, 3, strides=2)  # 16×16
    dem_16 = AveragePooling2D((4,4))(dem)
    slope_16 = AveragePooling2D((4,4))(slope)
    aspect_16 = AveragePooling2D((4,4))(aspect)
    curvature_16 = AveragePooling2D((4,4))(curvature)
    building_16 = AveragePooling2D((4,4))(building)
    pipe_16 = AveragePooling2D((4,4))(pipe)
    junction_16 = AveragePooling2D((4,4))(junction)
    phy_16 = physical_constraints_module(
        dem_16, slope_16, aspect_16, curvature_16, building_16, pipe_16, junction_16
    )
    x = Concatenate(axis=-1)([x, phy_16])
    x = Conv2D(256, (1,1), padding='same')(x)

    # ========== 4️⃣ 上采样回 64×64，第四次 PGNN ==========
    x = UpSampling2D((2,2))(x)   # 32×32
    x = Concatenate()([x, phy_32])
    x = UpSampling2D((2,2))(x)   # 64×64
    x = Concatenate()([x, phy_64])
    x = Conv2D(256, (3,3), padding='same', activation='relu')(x)   # ← 新增
    phy_final = physical_constraints_module(dem, slope, aspect, curvature, building, pipe, junction)
    # x = Concatenate(axis=-1)([x, phy_final])              
    x = Concatenate(axis=-1)([x, phy_final, rainfall])      #这里要和rainfall拼接

    # ========== 注意力（可选） ==========
    attention = Conv2D(1, (1,1), activation='sigmoid')(phy_final)
    attended_features = Multiply()([x, attention])
    x = attended_features

    # 编码器-解码器结构
    x1 = enhanced_separable_block(attended_features, 64, 3)
    x = enhanced_separable_block(x1, 64, 3, strides=2)
    x = enhanced_separable_block(x, 128, 3)
    x = enhanced_separable_block(x, 64, 3)
    x = UpSampling2D(size=(2,2))(x)
    x = Concatenate()([x, x1])
    
    # 输出层
    x = enhanced_separable_block(x, 32, 3)
    x = Conv2D(16, (3,3), padding='same', activation='relu')(x)
    output = Conv2D(NUM_CLASSES, (1,1), activation='softmax')(x)
    
    # 构建模型
    model = Model(
        inputs=[main_input, rain_input],
        outputs=output,
        name='PGNN_Classification_Model'
    )
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5),
        loss=weighted_sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
    )
    
    return model


#-------------------------------无PGNN----------------------------------------------------
def channel_attention0(input_feature, ratio=4):
    channel = input_feature.shape[-1]
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer=he_normal())
    shared_layer_two = Dense(channel, kernel_initializer=he_normal())

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return Multiply()([input_feature, cbam_feature])

# ---------- 增强深度可分离块（去掉 PGNN 相关） ----------
def enhanced_separable_block0(x, filters, kernel_size, strides=1):
    shortcut = x
    # 深度可分离卷积
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                        padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 点卷积
    x = Conv2D(filters, (1, 1), padding='same', kernel_initializer=he_normal())(x)
    x = BatchNormalization()(x)
    # 通道注意力
    x = channel_attention0(x)
    # 残差连接
    if strides == 1 and shortcut.shape[-1] == filters:
        x = Add()([x, shortcut])
    return Activation('relu')(x)

# ---------- 模型 ----------
def build_classification_model(input_shape=(64, 64, 7), rain_period=12):
    main_input = Input(shape=input_shape, name='main_input')
    rain_input = Input(shape=(rain_period,), name='rain_input')

    # 1. 降雨特征
    rainfall = Dense(64, activation='relu')(rain_input)
    rainfall = Dropout(0.3)(rainfall)
    rainfall = Dense(128, activation='relu')(rainfall)
    rainfall = Dropout(0.3)(rainfall)
    rainfall = Dense(256, activation='relu')(rainfall)
    rainfall = Reshape([1, 1, 256])(rainfall)
    rainfall = UpSampling2D(size=input_shape[:2])(rainfall)

    # 2. 主网络：直接以 main_input 作为起始
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(main_input)
    x = Concatenate(axis=-1)([x, rainfall])  # 融合降雨
    x = Conv2D(64, (1, 1), padding='same')(x)

    # encoder-decoder 结构，仅用普通卷积块
    x = enhanced_separable_block0(x, 64, 3)
    x = enhanced_separable_block0(x, 64, 3, strides=2)  # 32×32
    x = enhanced_separable_block0(x, 128, 3)
    x = enhanced_separable_block0(x, 128, 3, strides=2)  # 16×16
    x = enhanced_separable_block0(x, 256, 3)

    # decoder
    x = UpSampling2D((2, 2))(x)  # 32×32
    x = enhanced_separable_block(x, 128, 3)
    x = UpSampling2D((2, 2))(x)  # 64×64
    x = enhanced_separable_block0(x, 64, 3)

    # 输出
    x = enhanced_separable_block0(x, 32, 3)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    output = Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(x)

    # 构建模型
    model = Model(
        inputs=[main_input, rain_input],
        outputs=output,
        name='PGNN_Classification_Model'
    )
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5),
        loss=weighted_sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
    )
    return model

