import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Conv1D, Dense, BatchNormalization, 
                                     Activation, Multiply, Add, Concatenate, UpSampling2D, 
                                     MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, 
                                     Reshape, Lambda, Subtract, ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal

# 物理约束与特征提取
def compute_dem_gradients(tensor):
    # 计算高程的空间物理梯度 
    dy, dx = tf.image.image_gradients(tensor)
    return tf.concat([dx, dy], axis=-1)

def physics_interaction_module(dem, junction, slope, aspect, curvature, pipe):
    dem_grad = Lambda(compute_dem_gradients, name="dem_gradient")(dem)
    # 汇水势能
    gather_inputs = Concatenate()([slope, curvature, dem_grad, aspect])
    gather_potential = Conv2D(32, (3,3), padding='same', activation='relu', 
                              kernel_initializer=he_normal(), name="gather_potential")(gather_inputs)
    
    # 排水
    drain_inputs = Concatenate()([junction, pipe])
    drain_resistance = Conv2D(32, (3,3), padding='same', activation='relu', 
                              kernel_initializer=he_normal(), name="drain_resistance")(drain_inputs)
    
    # 净汇水风险 (汇水 - 排水)
    net_risk = Subtract(name="mass_balance_sub")([gather_potential, drain_resistance])
    net_risk = ReLU(name="mass_balance_relu")(net_risk)
    
    phys_feat = Concatenate()([dem, net_risk, gather_potential, drain_resistance])
    phys_feat = Conv2D(64, (3,3), padding='same', activation='relu')(phys_feat)
    
    return phys_feat

def rainfall_temporal_extractor(rain_input):
    # 降雨时序演进提取器 (1D Casual CNN)
    x = Reshape((rain_input.shape[1], 1))(rain_input)
    
    x = Conv1D(16, kernel_size=3, padding='same', activation='relu', kernel_initializer=he_normal())(x)
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer=he_normal())(x)
    seq_feat = GlobalAveragePooling1D()(x)
    
    accumulation = Lambda(lambda t: tf.reduce_sum(t, axis=1, keepdims=True), name="rain_accumulation")(rain_input)
    
    combined = Concatenate()([seq_feat, accumulation])
    rain_state = Dense(64, activation='relu', kernel_initializer=he_normal())(combined)
    return rain_state

# 多尺度主干网络组件
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

# 主模型构建函数 双分支架构
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
    
    # 双分支输出
    # 分支 A：概率掩膜分支 (Classification Branch)
    # 负责判断“这个像素点是否会积水？” 
    prob_mask = Conv2D(1, (1, 1), activation='sigmoid', name="prob_mask")(dec1)
    
    # 分支 B：水深回归分支 (Regression Branch)
    # 负责回答“如果这里积水了，水有多深？”
    raw_depth = Conv2D(1, (1, 1), activation='softplus', name="raw_depth")(dec1)
    
    # 物理门控融合 (Gated Fusion)
    # 最终水深 = 积水概率 × 预测水深
    gated_depth = Multiply(name="gated_depth")([prob_mask, raw_depth])
    # 物理边界截断
    # 建筑物内水深归零
    inv_building_mask = Lambda(lambda x: tf.cast(1.0, x.dtype) - x, name="inv_building_mask")(building)
    final_output = Multiply(name="depth_output")([gated_depth, inv_building_mask])
    
    # 构建模型
    model = Model(inputs=[spatial_input, rain_input], outputs=final_output)
    return model

#对比无物理引导的基础深度学习模型
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Conv1D, Dense, BatchNormalization, 
                                     Activation, Concatenate, UpSampling2D, 
                                     MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, 
                                     Reshape, Lambda, ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal

# 数据驱动型时序特征提取
def standard_temporal_extractor(rain_input):
    x = Reshape((rain_input.shape[1], 1))(rain_input)
    x = Conv1D(16, kernel_size=3, padding='same', activation='relu', kernel_initializer=he_normal())(x)
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer=he_normal())(x)
    seq_feat = GlobalAveragePooling1D()(x)
    rain_state = Dense(64, activation='relu', kernel_initializer=he_normal())(seq_feat)
    return rain_state

# 多尺度主干网络组件 (保持一致)
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

# 纯数据驱动主模型 (Baseline Pure-Deep-Learning Model)
def build_pure_datadriven_model(input_shape=(64, 64, 7), rain_period=12):
    
    spatial_input = Input(shape=input_shape, name="spatial_input")
    rain_input = Input(shape=(rain_period,), name="rain_input")
    
    # 移除了变量解耦与物理算子 (DEM梯度、产汇流)
    # 直接对 7 通道的原始图像矩阵进行常规卷积降维编码
    spatial_feat = Conv2D(64, (3,3), padding='same', activation='relu', 
                          kernel_initializer=he_normal(), name="spatial_embed")(spatial_input)
    
    # 提取时序特征 (纯 CNN)
    rain_feat_vec = standard_temporal_extractor(rain_input)
    rain_broadcast = Lambda(
        lambda tensors: tf.tile(tf.reshape(tensors[0], [-1, 1, 1, 64]), 
                                [1, tf.shape(tensors[1])[1], tf.shape(tensors[1])[2], 1]),
        name="spatio_temporal_broadcast"
    )([rain_feat_vec, spatial_feat])
    
    fused = Concatenate()([spatial_feat, rain_broadcast])
    x = Conv2D(64, (3,3), padding='same', activation='relu')(fused)
    
    # U-Net 多尺度网络 (完全等价的参数容量)
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
    
    # 核心修改：移除零膨胀和物理截断，直接回归
    # 采用 ReLU 激活函数直接回归出水深（ReLU 保证水深非负，这是回归深度的基本设定）
    # 没有概率掩膜的乘法运算，也没有建筑物的强制清零。完全指望网络自己从 Loss 中学到。
    final_output = Conv2D(1, (1, 1), activation='relu', name="depth_output")(dec1)
    
    # 构建模型
    model = Model(inputs=[spatial_input, rain_input], outputs=final_output)
    
    return model