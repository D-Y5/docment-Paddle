import os
import paddle
import argparse
from paddle.io import DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from models.docaligner import DocAligner, DocAlignerLoss
from models.dataset import create_dataloaders
from models.loss import compute_iou, compute_nme

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/smartdoc")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--save_interval", type=int, default=5)
    return parser.parse_args()

def setup_environment(output_dir):
    paddle.device.set_device("gpu")
    output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    metrics = {"iou": [], "nme": []}

    with tqdm(dataloader, desc=f"Epoch {epoch} Training") as pbar:
        for batch in pbar:
            images = batch["image"]
            corners = batch["corners"]
            
            outputs = model(images)
            loss = criterion(outputs, {"corners": corners})
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            # 计算指标
            with paddle.no_grad():
                pred_corners = outputs["corners"]
                metrics["iou"].append(compute_iou(pred_corners, corners).mean().item())
                metrics["nme"].append(compute_nme(pred_corners, corners).mean().item())
            
            # 更新进度条
            avg_iou = sum(metrics["iou"]) / len(metrics["iou"])
            avg_nme = sum(metrics["nme"]) / len(metrics["nme"])
            pbar.set_postfix(loss=loss.item(), iou=avg_iou, nme=avg_nme)

def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    val_iou = []
    val_nme = []
    
    with paddle.no_grad(), tqdm(dataloader, desc="Validating") as pbar:
        for batch in pbar:
            images = batch["image"]
            corners = batch["corners"]
            
            outputs = model(images)
            loss = criterion(outputs, {"corners": corners}).item()
            
            # 计算指标
            pred_corners = outputs["corners"]
            val_iou.append(compute_iou(pred_corners, corners).mean().item())
            val_nme.append(compute_nme(pred_corners, corners).mean().item())
            
            pbar.set_postfix(loss=loss, iou=val_iou[-1], nme=val_nme[-1])
            
    return {
        "loss": sum(val_loss) / len(val_loss),
        "iou": sum(val_iou) / len(val_iou),
        "nme": sum(val_nme) / len(val_nme)
    }

def main():
    args = parse_args()
    output_dir = setup_environment(args.output_dir)
    
    # 数据和模型
    train_loader, val_loader = create_dataloaders(
        args.data_root, 
        batch_size=args.batch_size
    )
    model = DocAligner()
    criterion = DocAlignerLoss(alpha=0.1)
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=args.lr
    )
    
    # 训练循环
    best_iou = 0
    best_model_path = output_dir / "best_model.pdparams"
    
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_metrics = validate(model, val_loader, criterion)
        
        print(f"\nEpoch {epoch} Validation:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  IOU: {val_metrics['iou']:.4f}")
        print(f"  NME: {val_metrics['nme']:.4f}\n")
        
        # 保存最佳模型
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            paddle.save(model.state_dict(), best_model_path)
            print(f"  New best model saved with IoU: {best_iou:.4f}")
            
        # 定期保存
        if epoch % args.save_interval == 0:
            save_path = output_dir / f"epoch_{epoch}.pdparams"
            paddle.save(model.state_dict(), save_path)
            
if __name__ == "__main__":
    main()
