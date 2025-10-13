# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:05:15 2021
该模块包含的主要是数据处理函数，用于模拟前中后处理
@author: dell
"""
import numpy as np
import pandas 
import os

#标准化函数
def normalization(data):
    rangeV=np.max(data)-np.min(data)
    return (data-np.min(data))/rangeV

def normalization_ab(data):
    x_mean = data - np.mean(data)  
    return (data-x_mean) / (np.max(data)-np.min(data)) 

#反标准化函数
def normalization_reverse(data_pre_norl,data_predict):
    rangeV=np.max(data_pre_norl)-np.min(data_pre_norl)
    return data_predict*rangeV+ np.min(data_pre_norl)

#数据处理函数
def depth_data_process(table_path):
    
    #读取数据
    dataframe = pandas.read_table(table_path,header=None,sep=' ')
    dataset = dataframe.values
    dataset = np.nan_to_num(dataset)
    
    #存储dem数据
    n_dem_col=650
    n_dem_row=656
    dem=np.empty([n_dem_row,n_dem_col], dtype = float) 
    for k in range(n_dem_col):
        for j in range(n_dem_row):
            dem[j,k]=dataset[j,k].astype('float')
    
    #读取并存储水深数据
    n_sample_col=n_dem_col #每个淹没分布图/地形图的列数
    n_sample_row=n_dem_row #每个淹没分布图/地形图的行数
    n_sample=15 #总共有18张淹没分布图
    depth_sample=np.empty([n_sample,n_sample_row,n_sample_col],dtype=float)
    for i in range(n_sample):
        depth_sample[ i, :, :]=dataset[(i+1)*656:(i+2)*656,:].astype('float')
    
    return dem, depth_sample
    
#处理水深
def process_dem_data(table_path):
    # 读取数据
    dataframe = pandas.read_table(table_path, header=None, sep=' ')
    dataset = dataframe.values
    dataset = np.nan_to_num(dataset)
    
    # 存储DEM数据
    n_dem_col=650
    n_dem_row=656
    dem=np.empty([n_dem_row,n_dem_col], dtype = float) 
    for k in range(n_dem_col):
        for j in range(n_dem_row):
            dem[j,k]=dataset[j,k].astype('float')
    
    return dem


#降雨数据处理函数
def raindata_process(table_path):
    dataframe = pandas.read_table(table_path,header=None,sep=' ')
    dataset = dataframe.values
    dataset = np.nan_to_num(dataset)
    n_dem_col=12
    n_dem_row=6
    rain=np.empty([n_dem_row,n_dem_col], dtype = float) 
    for k in range(n_dem_col):
        for j in range(n_dem_row):
            rain[j,k]=dataset[j,k].astype('float')
    return rain

#处理积水数据
def process_depth_data(depth_folder):
    # 定义行范围
    start_row = 8    # 从第9行开始（跳过前8行）
    nrows = 656      # 要读取的行数，共185行


    # 获取所有水深文件
    depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.asc')]
    n_sample = len(depth_files)  # 水深图的数量
    n_dem_col = 650  # 假设列数为197，与高程数据相同
    depth_sample = np.empty([n_sample, nrows, n_dem_col], dtype=float)  # 存储水深数据的数组

    # 逐个读取水深数据文件
    for i, depth_file in enumerate(depth_files):
        depth_path = os.path.join(depth_folder, depth_file)
        
        # 只读取指定行范围的数据
        dataframe_depth = pandas.read_csv(
            depth_path,
            header=None,
            delim_whitespace=True,
            skiprows=start_row,
            nrows=nrows
        )
        
        # 确保读取的数据列数一致
        if dataframe_depth.shape[1] < n_dem_col:
            raise ValueError(f"File {depth_file} has fewer than {n_dem_col} columns")
        
        # 将水深数据存储到 depth_sample 数组中
        depth_sample[i, :, :] = dataframe_depth.iloc[:, :n_dem_col].values

    return depth_sample

#处理除dem外的地形特征数据
def data_process(table_path):
    dataframe = pandas.read_table(table_path,header=None,sep=' ')
    dataset = dataframe.values
    dataset = np.nan_to_num(dataset)
    n_dem_col=650
    n_dem_row=656
    dataTensor=np.empty([n_dem_row,n_dem_col], dtype = float) 
    for k in range(n_dem_col):
        for j in range(n_dem_row):
            dataTensor[j,k]=dataset[j,k].astype('float')
    return dataTensor
#处理预测用的水深数据
def data_process_pro(table_path):
    dataframe = pandas.read_table(table_path,header=None,sep=' ',skiprows=8,nrows=656)
    dataset = dataframe.values
    dataset = np.nan_to_num(dataset)
    n_dem_col=650
    n_dem_row=656
    dataTensor=np.empty([n_dem_row,n_dem_col], dtype = float) 
    for k in range(n_dem_col):
        for j in range(n_dem_row):
            dataTensor[j,k]=dataset[j,k].astype('float')
    return dataTensor

#建立样本后，采用旋转和翻转扩充depth数据
def data_augmentation_depth(scnNum, sample_per_scn, patch_row_col_Num, depth_input, interpolation='nearest'):
    temp=np.empty([patch_row_col_Num,patch_row_col_Num], dtype = float) 
    for k in range(scnNum*sample_per_scn):
        temp[:,:]=depth_input[k, :,:, 0]
        depth_input[scnNum*sample_per_scn+k, :,:, 0]=np.rot90(temp,k=1,axes=(1,0))#axes=(1,0)顺时针，axes=(0,1)逆时针
        depth_input[2*scnNum*sample_per_scn+k, :,:, 0]=np.rot90(temp,k=2,axes=(1,0))
        depth_input[3*scnNum*sample_per_scn+k, :,:, 0]=np.rot90(temp,k=3,axes=(1,0))
        depth_input[4*scnNum*sample_per_scn+k, :,:, 0]=np.flip(temp,axis=0)
        depth_input[5*scnNum*sample_per_scn+k, :,:, 0]=np.rot90(np.flip(temp,axis=0),k=1,axes=(1,0)) #axis=0沿x轴翻转，axis=1沿y轴翻转
        depth_input[6*scnNum*sample_per_scn+k, :,:, 0]=np.rot90(np.flip(temp,axis=0),k=2,axes=(1,0))
        depth_input[7*scnNum*sample_per_scn+k, :,:, 0]=np.rot90(np.flip(temp,axis=0),k=3,axes=(1,0))
    return depth_input

#建立样本后，采用旋转和翻转扩充地形特征数据
def data_augmentation_main(scnNum,sample_per_scn,patch_row_col_Num,main_input):
    temp=np.empty([patch_row_col_Num,patch_row_col_Num], dtype = float) 
    for j in range(7):    #7代表输入参数个数，修改输入参数个数时，记得修改这里
        for k in range(scnNum*sample_per_scn):
            temp[:,:]=main_input[k, :,:, j]
            main_input[scnNum*sample_per_scn+k, :,:, j]=np.rot90(temp,k=1,axes=(1,0))
            main_input[2*scnNum*sample_per_scn+k, :,:, j]=np.rot90(temp,k=2,axes=(1,0))
            main_input[3*scnNum*sample_per_scn+k, :,:, j]=np.rot90(temp,k=3,axes=(1,0))
            main_input[4*scnNum*sample_per_scn+k, :,:, j]=np.flip(temp,axis=0)
            main_input[5*scnNum*sample_per_scn+k, :,:, j]=np.rot90(np.flip(temp,axis=0),k=1,axes=(1,0))
            main_input[6*scnNum*sample_per_scn+k, :,:, j]=np.rot90(np.flip(temp,axis=0),k=2,axes=(1,0))
            main_input[7*scnNum*sample_per_scn+k, :,:, j]=np.rot90(np.flip(temp,axis=0),k=3,axes=(1,0))
    return main_input
    
    
#将整个研究区域处理成多个样本，作为训练好的模型的输入
#def predict_dataInput(rowNum,colNum,patch_row_col_Num,patch_num_row,patch_num_col,inpFea_num,dem,mask,building):
def predict_dataInput(rowNum,colNum,patch_row_col_Num,patch_num_row,patch_num_col,inpFea_num,dem,junctn,slope,aspect,curvatu,building,pipe):
    pred_dem_whole=np.empty([(patch_num_row+1)*(patch_num_col+1),patch_row_col_Num,patch_row_col_Num,inpFea_num],dtype=float)
    for j in range(patch_num_row):#行
        for k in range(patch_num_col):#列
            pred_dem_whole[j*patch_num_col+k,:,:,0]=dem[j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
            pred_dem_whole[j*patch_num_col+k,:,:,1]=junctn[j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
            pred_dem_whole[j*patch_num_col+k,:,:,2]=slope[j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
            pred_dem_whole[j*patch_num_col+k,:,:,3]=aspect[j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
            pred_dem_whole[j*patch_num_col+k,:,:,4]=curvatu[j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
            #pred_dem_whole[j*patch_num_col+k,:,:,5]=mask[j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
            pred_dem_whole[j*patch_num_col+k,:,:,5]=building[j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
            pred_dem_whole[j*patch_num_col+k,:,:,6]=pipe[j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
    
    #补全右边和下边
    for j in range(patch_num_row):#行,python数组左闭右开
        pred_dem_whole[patch_num_row*patch_num_col+j,:,:,0]=dem[j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-patch_row_col_Num:colNum].astype('float')
        pred_dem_whole[patch_num_row*patch_num_col+j,:,:,1]=junctn[j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-patch_row_col_Num:colNum].astype('float')
        pred_dem_whole[patch_num_row*patch_num_col+j,:,:,2]=slope[j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-patch_row_col_Num:colNum].astype('float')
        pred_dem_whole[patch_num_row*patch_num_col+j,:,:,3]=aspect[j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-patch_row_col_Num:colNum].astype('float')
        pred_dem_whole[patch_num_row*patch_num_col+j,:,:,4]=curvatu[j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-patch_row_col_Num:colNum].astype('float')
        #pred_dem_whole[patch_num_row*patch_num_col+j,:,:,5]=mask[j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-patch_row_col_Num:colNum].astype('float')
        pred_dem_whole[patch_num_row*patch_num_col+j,:,:,5]=building[j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-patch_row_col_Num:colNum].astype('float')
        pred_dem_whole[patch_num_row*patch_num_col+j,:,:,6]=pipe[j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-patch_row_col_Num:colNum].astype('float')
    
    for k in range(patch_num_col):#列
        pred_dem_whole[patch_num_row*(patch_num_col+1)+k,:,:,0]=dem[rowNum-patch_row_col_Num:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
        pred_dem_whole[patch_num_row*(patch_num_col+1)+k,:,:,1]=junctn[rowNum-patch_row_col_Num:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
        pred_dem_whole[patch_num_row*(patch_num_col+1)+k,:,:,2]=slope[rowNum-patch_row_col_Num:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
        pred_dem_whole[patch_num_row*(patch_num_col+1)+k,:,:,3]=aspect[rowNum-patch_row_col_Num:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
        pred_dem_whole[patch_num_row*(patch_num_col+1)+k,:,:,4]=curvatu[rowNum-patch_row_col_Num:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
        #pred_dem_whole[patch_num_row*(patch_num_col+1)+k,:,:,5]=mask[rowNum-patch_row_col_Num:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
        pred_dem_whole[patch_num_row*(patch_num_col+1)+k,:,:,5]=building[rowNum-patch_row_col_Num:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
        pred_dem_whole[patch_num_row*(patch_num_col+1)+k,:,:,6]=pipe[rowNum-patch_row_col_Num:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num].astype('float')
        
    pred_dem_whole[(patch_num_row+1)*(patch_num_col+1)-1,:,:,0]=dem[rowNum-patch_row_col_Num:rowNum,colNum-patch_row_col_Num:colNum].astype('float')
    pred_dem_whole[(patch_num_row+1)*(patch_num_col+1)-1,:,:,1]=junctn[rowNum-patch_row_col_Num:rowNum,colNum-patch_row_col_Num:colNum].astype('float')
    pred_dem_whole[(patch_num_row+1)*(patch_num_col+1)-1,:,:,2]=slope[rowNum-patch_row_col_Num:rowNum,colNum-patch_row_col_Num:colNum].astype('float')
    pred_dem_whole[(patch_num_row+1)*(patch_num_col+1)-1,:,:,3]=aspect[rowNum-patch_row_col_Num:rowNum,colNum-patch_row_col_Num:colNum].astype('float')
    pred_dem_whole[(patch_num_row+1)*(patch_num_col+1)-1,:,:,4]=curvatu[rowNum-patch_row_col_Num:rowNum,colNum-patch_row_col_Num:colNum].astype('float')
    #pred_dem_whole[(patch_num_row+1)*(patch_num_col+1)-1,:,:,5]=mask[rowNum-patch_row_col_Num:rowNum,colNum-patch_row_col_Num:colNum].astype('float')
    pred_dem_whole[(patch_num_row+1)*(patch_num_col+1)-1,:,:,5]=building[rowNum-patch_row_col_Num:rowNum,colNum-patch_row_col_Num:colNum].astype('float')
    pred_dem_whole[(patch_num_row+1)*(patch_num_col+1)-1,:,:,6]=pipe[rowNum-patch_row_col_Num:rowNum,colNum-patch_row_col_Num:colNum].astype('float')
    return pred_dem_whole

#预测结果出来后，将多个patch合成整个研究区的形状
def predict_resultCompos(rowNum,colNum,patch_row_col_Num,patch_num_row,patch_num_col,pred):
    pred_depth_whole=np.empty([1,rowNum,colNum,1], dtype = float)
    remainder_row=rowNum%patch_row_col_Num#列方向余数，距下方
    remainder_col=colNum%patch_row_col_Num#行方向余数数，距右方
    for j in range(patch_num_row):#行
        for k in range(patch_num_col):#列
            pred_depth_whole[0,j*patch_row_col_Num:(j+1)*patch_row_col_Num,k*patch_row_col_Num:(k+1)*patch_row_col_Num,0]=pred[j*patch_num_col+k,:,:,0]
    
    for j in range(patch_num_row):#行
        pred_depth_whole[0,j*patch_row_col_Num:(j+1)*patch_row_col_Num,colNum-remainder_col:colNum,0]=pred[patch_num_row*patch_num_col+j,:,patch_row_col_Num-remainder_col:patch_row_col_Num,0]
    
    for k in range(patch_num_col):#列
        pred_depth_whole[0,rowNum-remainder_row:rowNum,k*patch_row_col_Num:(k+1)*patch_row_col_Num,0]=pred[patch_num_row*(patch_num_col+1)+j,patch_row_col_Num-remainder_row:patch_row_col_Num,:,0]
    
    pred_depth_whole[0,rowNum-remainder_row:rowNum,colNum-remainder_col:colNum,0]=pred[(patch_num_row+1)*(patch_num_col+1)-1,patch_row_col_Num-remainder_row:patch_row_col_Num,patch_row_col_Num-remainder_col:patch_row_col_Num,0]
    return pred_depth_whole

from scipy.ndimage import gaussian_filter

# # ==================== 物理增强核心函数 ====================
# def physics_augmentation(main_patch, depth_patch, rain_data=None):
#     """
#     基于物理规律的增强方法：
#     1. 降雨强度变化模拟
#     2. 管网状态扰动
#     3. 地形微小形变
#     参数：
#         main_patch: 地形特征数据 [H,W,7]
#         depth_patch: 对应水深标签 [H,W,1]
#         rain_data: 降雨数据 (可选)
#     返回：
#         增强后的main_patch和depth_patch
#     """
#     np.random.seed(int.from_bytes(os.urandom(4), 'big'))  # 确保可复现性
    
#     # 1. 降雨强度变化 (影响所有相关特征)
#     if np.random.rand() > 0.5:  # 50%概率应用
#         rain_scale = np.random.uniform(0.7, 1.5)  # 降雨强度变化范围
#         # 假设第1通道是junction(汇水点)，第6通道是pipe(管网)
#         main_patch[:,:,1] = np.clip(main_patch[:,:,1] * rain_scale, 0, 1)
#         main_patch[:,:,6] = np.clip(main_patch[:,:,6] * (1 + (rain_scale-1)*0.3, 0, 1))  # 管网影响较小
        
#         # 同步调整水深标签 (降雨越多积水越深)
#         if rain_scale > 1:
#             depth_patch = np.where(depth_patch>0, 
#                                  depth_patch * (1 + (rain_scale-1)*0.5), 
#                                  depth_patch)
    
#     # 2. 管网状态扰动 (模拟堵塞/畅通)
#     if np.random.rand() > 0.7:  # 30%概率应用
#         pipe_noise = gaussian_filter(
#             np.random.normal(0, 0.1, main_patch[:,:,6].shape), 
#             sigma=1.5
#         )
#         main_patch[:,:,6] = np.clip(main_patch[:,:,6] + pipe_noise, 0, 1)
    
#     # 3. 地形微小形变 (DEM、坡度等)
#     if np.random.rand() > 0.5:  # 50%概率应用
#         alpha = np.random.uniform(5, 15)  # 形变强度
#         sigma = np.random.uniform(2, 4)   # 平滑系数
        
#         # 对DEM和衍生特征进行一致形变
#         dx = gaussian_filter((np.random.rand(*main_patch.shape[:2]) * 2 - 1), sigma) * alpha
#         dy = gaussian_filter((np.random.rand(*main_patch.shape[:2]) * 2 - 1), sigma) * alpha
        
#         # 应用相同的形变场到所有空间相关特征
#         for c in [0,2,3,4]:  # DEM(0), slope(2), aspect(3), curvature(4)
#             main_patch[:,:,c] = elastic_deformation_single(main_patch[:,:,c], dx, dy)
        
#         # 同步形变水深标签
#         depth_patch[:,:,0] = elastic_deformation_single(depth_patch[:,:,0], dx, dy)
    
#     # 确保建筑物区域不受影响 (建筑物区域应保持无积水)
#     building_mask = (main_patch[:,:,5] > 0.5)
#     depth_patch[building_mask] = 0
    
#     return main_patch, depth_patch

# def elastic_deformation_single(channel, dx, dy):
#     """对单通道应用弹性形变"""
#     from scipy.ndimage import map_coordinates
#     shape = channel.shape
#     x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
#     indices = np.reshape(y+dy, (-1,1)), np.reshape(x+dx, (-1,1))
#     return map_coordinates(channel, indices, order=1).reshape(shape)

# # ==================== 改进的采样函数 ====================
# def guided_sampling(dem, slope, building_mask, sample_per_scn=200, patch_size=64):
#     """
#     基于地形特征的引导式采样
#     优先选择易涝区域(低洼+低坡度)
#     """
#     # 计算易涝权重图
#     flood_weight = np.exp(-dem / np.std(dem)) * (1 - slope / np.max(slope))
#     flood_weight[building_mask == 1] = 0  # 排除建筑物区域
    
#     samples = []
#     rows, cols = dem.shape
#     valid_area = flood_weight[patch_size//2 : rows-patch_size//2, 
#                              patch_size//2 : cols-patch_size//2]
    
#     if np.sum(valid_area) == 0:
#         raise ValueError("无有效采样区域，请检查输入数据")
    
#     for _ in range(sample_per_scn):
#         # 按权重分布采样
#         flat_weights = valid_area.flatten()
#         center_idx = np.random.choice(
#             len(flat_weights), 
#             p=flat_weights/flat_weights.sum()
#         )
#         y, x = np.unravel_index(center_idx, valid_area.shape)
#         y += patch_size//2  # 补偿裁剪偏移
#         x += patch_size//2
#         samples.append((x, y))
    
#     return samples

# # ==================== 改进的数据增强函数 ====================
# def enhanced_data_augmentation(scnNum, sample_per_scn, patch_size, main_input, depth_input, rain_input=None):
#     """
#     增强版数据增强流程：
#     1. 基础几何增强 (旋转/翻转)
#     2. 物理约束增强
#     3. 可选降雨数据增强
#     """
#     # 初始几何增强 (8倍)
#     main_input = data_augmentation_main(scnNum, sample_per_scn, patch_size, main_input)
#     depth_input = data_augmentation_depth(scnNum, sample_per_scn, patch_size, depth_input)
    
#     # 物理增强 (额外增加4倍)
#     total_samples = 8 * scnNum * sample_per_scn
#     augmented_main = np.empty_like(main_input, shape=[total_samples*2, *main_input.shape[1:]])
#     augmented_depth = np.empty_like(depth_input, shape=[total_samples*2, *depth_input.shape[1:]])
    
#     # 复制原始增强数据
#     augmented_main[:total_samples] = main_input
#     augmented_depth[:total_samples] = depth_input
    
#     # 应用物理增强
#     for i in range(total_samples):
#         idx = total_samples + i
#         augmented_main[idx], augmented_depth[idx] = physics_augmentation(
#             main_input[i].copy(), 
#             depth_input[i].copy()
#         )
    
#     # 降雨数据增强 (如果存在)
#     if rain_input is not None:
#         augmented_rain = np.repeat(rain_input, 2, axis=0)  # 物理增强样本复用原始降雨数据
#         return augmented_main, augmented_depth, augmented_rain
    
#     return augmented_main, augmented_depth