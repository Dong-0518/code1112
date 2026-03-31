import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

from models import create_model
from data_loader import get_transforms

# ==========================================
# ⚙️ 参数设置区 (适配 GitHub 原版代码)
# ==========================================
# 请改成你实际训练用的模型，比如 'resnet50' 或 'vit_b16'
MODEL_TYPE = 'resnet50'  
FEATURE_DIM = 512      
IMAGE_SIZE = 224       
ANCHOR_WEIGHT = 11.0   
# ==========================================

def get_single_feature(model, image_path, device):
    """提取单张图片特征 (适配单流架构)"""
    # 直接调用你 data_loader.py 里的旧版 transform
    transform = get_transforms(mode='test', image_size=IMAGE_SIZE, model_type=MODEL_TYPE)
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 旧版模型只需要传一个 img_tensor 即可
        feature = model.feature_extractor(img_tensor)
        
    return feature.cpu().numpy().flatten()

def main():
    print(f"🚀 启动毛被特征干预程序 (适配 {MODEL_TYPE} 版)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📦 加载 {MODEL_TYPE} 模型权重...")
    model = create_model(model_type=MODEL_TYPE, feature_dim=FEATURE_DIM, pretrained=False, use_triplet=True)
    
    # 自动去你的 outputs 文件夹里找对应模型的 best.pth
    weights_path = f'outputs/models/{MODEL_TYPE}_best.pth'
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        # 兼容处理：检查是存了整个字典还是只存了 state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✅ 成功加载了微调模型权重: {weights_path}")
    else:
        print(f"⚠️ 警告：未找到 {weights_path}，使用未训练的基础权重！")
        
    model.to(device)
    model.eval()

    print("🔬 正在提取 3 张锚点照片的‘真理特征’...")
    feat_s = get_single_feature(model, 'anchors/stellate.jpg', device)
    feat_p = get_single_feature(model, 'anchors/peltate.jpg', device)
    feat_m = get_single_feature(model, 'anchors/mixed.jpg', device)

    # 读取特征文件 (根据模型名字自动拼出文件名)
    feat_file = f'outputs/features/specimen_{MODEL_TYPE}_features_1.npz'
    print(f"📊 正在读取原始特征文件: {feat_file}...")
    
    if not os.path.exists(feat_file):
        print(f"❌ 找不到特征文件！请先运行: python main.py --mode extract --model_type {MODEL_TYPE}")
        return
        
    data = np.load(feat_file, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    
    print("🧲 正在进行特征干预...")
    new_features = []
    
    for feat in features:
        # 【切片操作】：永远只拿前 512 维去和锚点比对，防止报错
        feat_raw = feat[:FEATURE_DIM].reshape(1, -1)
        
        sim_s = cosine_similarity(feat_raw, feat_s.reshape(1, -1))[0][0]
        sim_p = cosine_similarity(feat_raw, feat_p.reshape(1, -1))[0][0]
        sim_m = cosine_similarity(feat_raw, feat_m.reshape(1, -1))[0][0]
        
        # 放大得分
        signals = np.array([sim_s, sim_p, sim_m]) * ANCHOR_WEIGHT
        
        # 重新拼接：512维原始 + 3维干预信号 = 515维
        feat_anchored = np.concatenate([feat[:FEATURE_DIM], signals])
        new_features.append(feat_anchored)
        
    # 保存为一个新的文件，带上 species_names 防止后续建树报错
    save_path = 'outputs/features/specimen_resnet50_features.npz'
    
    save_kwargs = {'features': np.array(new_features), 'labels': labels}
    if 'species_names' in data.files:
        save_kwargs['species_names'] = data['species_names']
        
    np.savez(save_path, **save_kwargs)
    print(f"🎉 干预完成！515维新特征已保存至: {save_path}")

if __name__ == '__main__':
    main()