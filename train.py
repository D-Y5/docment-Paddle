import os
import yaml
import paddle
import argparse
import numpy as np
from tqdm import tqdm
from paddle.io import Dataset, DataLoader
from model import DocDetector

class DocDataset(Dataset):
    """文档检测数据集"""
    
    def __init__(self, data_root, image_size=(640, 640), transform=None):
        self.data_root = data_root
        self.image_size = image_size
        self.transform = transform
        
        # 加载所有图像路径
        self.image_paths = []
        self.mask_paths = []
        
        image_dir = os.path.join(data_root, "images")
        mask_dir = os.path.join(data_root, "masks")
        
        for image_name in os.listdir(image_dir):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(image_dir, image_name)
                mask_name = image_name.replace(".jpg", ".png")
                mask_path = os.path.join(mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        import cv2
        
        # 读取图像
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        
        # 读取分割掩码
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        
        mask = cv2.resize(mask, self.image_size)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # 添加通道维度
        
        return image, mask

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader)):
        # 转换为paddle张量
        images = paddle.to_tensor(images)
        masks = paddle.to_tensor(masks)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        
        # 反向传播
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        
        # 打印日志
        if (batch_idx + 1) % config["train"]["log_interval"] == 0:
            print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, dataloader, criterion, config):
    """验证模型"""
    model.eval()
    total_loss = 0
    
    with paddle.no_grad():
        for images, masks in tqdm(dataloader):
            # 转换为paddle张量
            images = paddle.to_tensor(images)
            masks = paddle.to_tensor(masks)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def dice_loss(pred, target):
    """Dice Loss"""
    smooth = 1e-5
    pred = paddle.nn.functional.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train document boundary detection model")
    parser.add_argument("--config", type=str, default="train.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use (gpu or cpu)")
    
    args = parser.parse_args()
    
    # 设置设备
    paddle.set_device(args.device)
    print(f"Using device: {args.device}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建模型保存目录
    os.makedirs(config["train"]["save_dir"], exist_ok=True)
    
    # 加载数据集
    train_dataset = DocDataset(
        config["dataset"]["train_root"],
        image_size=config["dataset"]["image_size"]
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=config["dataset"]["shuffle"],
        num_workers=config["dataset"]["num_workers"]
    )
    
    # 初始化模型
    model = DocDetector(
        backbone=config["model"]["backbone"],
        input_size=config["model"]["input_size"]
    )
    
    # 初始化优化器
    if config["optimizer"]["type"] == "Adam":
        optimizer = paddle.optimizer.Adam(
            parameters=model.parameters(),
            learning_rate=config["train"]["learning_rate"],
            weight_decay=config["train"]["weight_decay"],
            beta1=config["optimizer"]["beta1"],
            beta2=config["optimizer"]["beta2"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']['type']}")
    
    # 初始化损失函数
    if config["loss"]["type"] == "BCEWithLogitsLoss":
        criterion = paddle.nn.BCEWithLogitsLoss(reduction=config["loss"]["reduction"])
    elif config["loss"]["type"] == "DiceLoss":
        criterion = dice_loss
    elif config["loss"]["type"] == "MSELoss":
        criterion = paddle.nn.MSELoss(reduction=config["loss"]["reduction"])
    else:
        raise ValueError(f"Unsupported loss: {config['loss']['type']}")
    
    # 加载验证集
    val_dataset = DocDataset(
        config["dataset"]["val_root"],
        image_size=config["dataset"]["image_size"]
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"]
    )
    
    # 开始训练
    best_val_loss = float('inf')
    
    for epoch in range(config["train"]["epochs"]):
        # 训练
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, epoch, config)
        
        # 验证
        if (epoch + 1) % config["eval"]["eval_interval"] == 0:
            val_loss = validate(model, val_dataloader, criterion, config)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(config["train"]["save_dir"], "best_model.pdparams")
                paddle.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path} with val_loss: {val_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % config["train"]["save_interval"] == 0:
            model_path = os.path.join(config["train"]["save_dir"], f"model_epoch_{epoch+1}.pdparams")
            paddle.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
