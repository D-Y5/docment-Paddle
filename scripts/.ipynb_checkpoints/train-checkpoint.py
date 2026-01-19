import sys

sys.path.append("/home/aistudio/paddle_project")
import os

import paddle
from paddle_utils import *
import paddle.nn.functional as F

"""
DocAligner训练脚本
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import paddle.nn.functional as F

import numpy as np
import yaml
from tqdm import tqdm


from visualdl import LogWriter #解决torch.utils.tensorboard.SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))
from models.dataset import create_dataloaders
from models.docaligner import create_docaligner_model
from models.loss import DocAlignerLoss, compute_iou, compute_nme
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="paddle.nn.layer.norm")


class Trainer:
    """DocAligner训练器"""

    def __init__(self, config):
        self.config = config
        # self.device = paddle.device(
        #    # "cuda" if paddle.cuda.is_available() and config["use_cuda"] else "cpu"
        #     "gpu" if paddle.device.is_compiled_with_cuda() and config["use_cuda"] else "cpu"
        # )
        if config["use_cuda"] and paddle.is_compiled_with_cuda():
            self.device = paddle.CUDAPlace(0)  # 使用第一个GPU
            paddle.device.set_device("gpu:0")
        else:
            self.device = paddle.CPUPlace()
            paddle.device.set_device("cpu")
        #paddle.device.set_device(self.device)
        print(f"Using device: {self.device}")
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        self.model = create_docaligner_model(
            model_type=config["model_type"],
            num_corners=4,
            pretrained=config["pretrained"],
        )
        self.model = self.model.to(self.device)
        self.criterion = DocAlignerLoss(
            heatmap_weight=config["heatmap_weight"],
            offset_weight=config["offset_weight"],
            use_focal=config["use_focal"],
        )
        self.optimizer = paddle.optimizer.AdamW(
            parameters=self.model.parameters(),
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(
            T_max=config["epochs"],
            eta_min=config["learning_rate"] * 0.01,
            learning_rate=self.optimizer.get_lr(),
        )
        self.optimizer.set_lr_scheduler(tmp_lr)
        self.scheduler = tmp_lr
        self.train_loader, self.val_loader = create_dataloaders(
            root_dir=config["data_root"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            input_size=tuple(config["input_size"]),
            target_size=config.get("target_size", config["input_size"]),  # 从配置读取
            use_heatmap=True,
        )
        #self.writer = torch.utils.tensorboard.SummaryWriter(self.output_dir / "logs")
        # 注意：路径需要转为字符串
        self.writer = LogWriter(str(self.output_dir / "logs"))
        self.current_epoch = 0
        self.best_iou = 0.0
        if config.get("checkpoint"):
            self.load_checkpoint(config["checkpoint"])

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_heatmap_loss = 0
        total_offset_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            target_heatmap = batch["heatmap"].to(self.device)
            target_corners = batch["corners"].to(self.device)
            self.optimizer.clear_grad()
            output = self.model(images)
            target = {"heatmap": target_heatmap}
            if "offset" in output:
                target_offset = self._compute_target_offset(
                    target_corners, target_heatmap.shape
                ).to(self.device)
                target["offset"] = target_offset
            #print(f"Model output shape: {output['heatmap'].shape}")
            #print(f"Target shape: {target['heatmap'].shape}")
            losses = self.criterion(output, target)
            losses["total"].backward()
            if self.config["grad_clip"] > 0:
                paddle.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=self.config["grad_clip"],
                )
            self.optimizer.step()
            total_loss += losses["total"].item()
            total_heatmap_loss += losses["heatmap"].item()
            if "offset" in losses:
                total_offset_loss += losses["offset"].item()
            pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "lr": f"{self.optimizer.get_lr():.6f}",
                }
            )
            if batch_idx % self.config["log_interval"] == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar(
                    "train/loss", losses["total"].item(), global_step
                )
                self.writer.add_scalar(
                    "train/heatmap_loss", losses["heatmap"].item(), global_step
                )
                if "offset" in losses:
                    self.writer.add_scalar(
                        "train/offset_loss", losses["offset"].item(), global_step
                    )
                self.writer.add_scalar(
                    "train/lr", self.optimizer.get_lr(), global_step
                )
        avg_loss = total_loss / len(self.train_loader)
        avg_heatmap_loss = total_heatmap_loss / len(self.train_loader)
        avg_offset_loss = total_offset_loss / len(self.train_loader)
        return {
            "total": avg_loss,
            "heatmap": avg_heatmap_loss,
            "offset": avg_offset_loss,
        }

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_nme = 0
        total_samples = 0
        with paddle.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                
                images = batch["image"].to(self.device)
                target_heatmap = batch["heatmap"].to(self.device)
                target_corners = batch["corners"].to(self.device)
                output = self.model(images)
                target = {"heatmap": target_heatmap}
                if "offset" in output:
                    target_offset = self._compute_target_offset(
                        target_corners, target_heatmap.shape
                    ).to(self.device)
                    target["offset"] = target_offset
                losses = self.criterion(output, target)
                #pred_corners = self._extract_corners_from_heatmap(output["heatmap"])

                pred_corners = self._extract_corners_from_heatmap(F.sigmoid(output["heatmap"]))
                pred_corners = pred_corners.to(self.device)
                iou = compute_iou(pred_corners, target_corners)
                nme = compute_nme(pred_corners, target_corners)
                total_loss += losses["total"].item()
                total_iou += iou.sum().item()
                total_nme += nme.sum().item()
                total_samples += iou.shape[0]
        avg_loss = total_loss / len(self.val_loader)
        avg_iou = total_iou / total_samples
        avg_nme = total_nme / total_samples
        return {"loss": avg_loss, "iou": avg_iou, "nme": avg_nme}

    def _compute_target_offset(self, corners, heatmap_shape):
        """计算目标偏移量"""
        batch_size = heatmap_shape[0]
        num_corners = 4
        height = heatmap_shape[2]
        width = heatmap_shape[3]
        offset = paddle.zeros(batch_size, num_corners * 2, height, width)
        for b in range(batch_size):
            for i in range(4):
                x, y = corners[b, i]
                x_heatmap = int(x * (width - 1))
                y_heatmap = int(y * (height - 1))
                offset_x = x * (width - 1) - x_heatmap
                offset_y = y * (height - 1) - y_heatmap
                offset[b, i * 2, y_heatmap, x_heatmap] = offset_x
                offset[b, i * 2 + 1, y_heatmap, x_heatmap] = offset_y
        return offset

    # def _extract_corners_from_heatmap(self, heatmap):
    #     """从热力图中提取角点坐标"""
    #     batch_size = heatmap.shape[0]
    #     num_corners = 4
    #     height = heatmap.shape[2]
    #     width = heatmap.shape[3]
    #     corners = paddle.zeros(batch_size, num_corners, 2)
    #     for b in range(batch_size):
    #         for i in range(num_corners):
    #             h = heatmap[b, i]
    #             max_val = paddle.max(h)
    #             if max_val > 0.1:
    #                 y, x = paddle.where(h == max_val)
    #                 y, x = y[0].item(), x[0].item()
    #                 corners[b, i, 0] = x / (width - 1)
    #                 corners[b, i, 1] = y / (height - 1)
    #             else:
    #                 corners[b, i] = paddle.tensor([0.5, 0.5])
    #     return corners

    def _extract_corners_from_heatmap(self, heatmap):
        """从热力图中提取角点坐标 - 最终修复版"""
        batch_size = heatmap.shape[0]
        num_corners = 4
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        corners = paddle.zeros([batch_size, num_corners, 2])
        
        for b in range(batch_size):
            for i in range(num_corners):
                h = heatmap[b, i]  # [H, W]
                
                # 找到最大值索引
                flat_idx = paddle.argmax(h.reshape([-1]))
                y = flat_idx // width
                x = flat_idx % width
                
                # 转换为归一化坐标
                corners[b, i, 0] = x.astype('float32') / (width - 1)
                corners[b, i, 1] = y.astype('float32') / (height - 1)
        
        return corners


    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_iou": self.best_iou,
            "config": self.config,
        }
        # checkpoint_path = self.output_dir / "checkpoint_latest.pth"
        # paddle.save(obj=checkpoint, path=checkpoint_path)
        checkpoint_path = str(self.output_dir / "checkpoint_latest.pth")
        paddle.save(obj=checkpoint, path=checkpoint_path)
        if is_best:
            #best_path = self.output_dir / "checkpoint_best.pth"
            best_path = str(self.output_dir / "checkpoint_best.pth")
            paddle.save(obj=checkpoint, path=best_path)
            print(f"Best model saved with IoU: {self.best_iou:.4f}")
        if self.current_epoch % self.config["save_interval"] == 0:
            #epoch_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
            epoch_path = str(self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pth")
            paddle.save(obj=checkpoint, path=epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = paddle.load(path=str(checkpoint_path))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_iou = checkpoint.get("best_iou", 0.0)
        print(f"Checkpoint loaded from epoch {self.current_epoch}")

    def train(self):
        """训练循环"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        for epoch in range(self.current_epoch, self.config["epochs"]):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()
            self.writer.add_scalar("epoch/train_loss", train_metrics["total"], epoch)
            self.writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("epoch/val_iou", val_metrics["iou"], epoch)
            self.writer.add_scalar("epoch/val_nme", val_metrics["nme"], epoch)
            print(f"\nEpoch {epoch}")
            print(f"  Train Loss: {train_metrics['total']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Val NME: {val_metrics['nme']:.4f}")
            print(f"  Learning Rate: {self.optimizer.get_lr():.6f}")
            is_best = val_metrics["iou"] > self.best_iou
            if is_best:
                self.best_iou = val_metrics["iou"]
            self.save_checkpoint(is_best)
        print(f"\nTraining complete!")
        print(f"Best IoU: {self.best_iou:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train DocAligner")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume"
    )
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
