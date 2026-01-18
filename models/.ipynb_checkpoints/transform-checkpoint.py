import paddle

"""
透视变换矫正工具
"""
from typing import Optional, Tuple

import cv2
import numpy as np


class PerspectiveTransformer:
    """透视变换矫正器"""

    def __init__(self, output_width: int = 800, output_height: int = 1000):
        """
        Args:
            output_width: 输出宽度
            output_height: 输出高度
        """
        self.output_width = output_width
        self.output_height = output_height

    def sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        将四个角点按顺序排列：左上、右上、右下、左下

        Args:
            corners: 四个角点 (4, 2)

        Returns:
            sorted_corners: 排序后的角点
        """
        center = np.mean(corners, axis=0)

        def get_corner_type(point):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            if dx < 0 and dy < 0:
                return 0
            elif dx >= 0 and dy < 0:
                return 1
            elif dx >= 0 and dy >= 0:
                return 2
            else:
                return 3

        sorted_corners = sorted(corners, key=get_corner_type)
        return np.array(sorted_corners, dtype=np.float32)

    def compute_perspective_transform(
        self,
        corners: np.ndarray,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算透视变换矩阵

        Args:
            corners: 四个角点 (4, 2)，按顺序：左上、右上、右下、左下
            output_width: 输出宽度，如果为None则使用默认值
            output_height: 输出高度，如果为None则使用默认值

        Returns:
            M: 透视变换矩阵 (3, 3)
            output_size: 输出尺寸 (width, height)
        """
        if output_width is None:
            output_width = self.output_width
        if output_height is None:
            output_height = self.output_height
        dst_corners = np.array(
            [
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(corners, dst_corners)
        return M, (output_width, output_height)

    def estimate_document_size(self, corners: np.ndarray) -> Tuple[int, int]:
        """
        估计文档尺寸（保持宽高比）

        Args:
            corners: 四个角点 (4, 2)

        Returns:
            width: 估计的宽度
            height: 估计的高度
        """
        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])
        avg_height = (left + right) / 2
        avg_width = (top + bottom) / 2
        aspect_ratio = avg_width / avg_height
        max_size = 1200
        if aspect_ratio > 1:
            width = max_size
            height = int(max_size / aspect_ratio)
        else:
            height = max_size
            width = int(max_size * aspect_ratio)
        return width, height

    def transform(
        self, image: np.ndarray, corners: np.ndarray, maintain_aspect: bool = True
    ) -> np.ndarray:
        """
        执行透视变换

        Args:
            image: 输入图像
            corners: 四个角点 (4, 2)
            maintain_aspect: 是否保持宽高比

        Returns:
            corrected_image: 矫正后的图像
        """
        sorted_corners = self.sort_corners(corners)
        if maintain_aspect:
            output_width, output_height = self.estimate_document_size(sorted_corners)
        else:
            output_width = self.output_width
            output_height = self.output_height
        M, output_size = self.compute_perspective_transform(
            sorted_corners, output_width, output_height
        )
        corrected_image = cv2.warpPerspective(
            image,
            M,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return corrected_image

    def transform_batch(
        self, images: paddle.Tensor, corners: paddle.Tensor
    ) -> paddle.Tensor:
        """
        批量透视变换

        Args:
            images: 图像批次 (B, C, H, W)
            corners: 角点批次 (B, 4, 2)，归一化到[0, 1]

        Returns:
            corrected_images: 矫正后的图像批次
        """
        batch_size = images.shape[0]
        corrected_images = []
        for i in range(batch_size):
            image = images[i].permute(1, 2, 0).cpu().numpy()
            corner = corners[i].cpu().numpy()
            h, w = image.shape[:2]
            corner[:, 0] *= w
            corner[:, 1] *= h
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            corrected = self.transform(image, corner)
            corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
            corrected_tensor = paddle.from_numpy(corrected).permute(2, 0, 1)
            corrected_images.append(corrected_tensor)
        corrected_batch = paddle.stack(corrected_images)
        return corrected_batch


def crop_document_from_corners(
    image: np.ndarray, corners: np.ndarray, padding: int = 20
) -> np.ndarray:
    """
    从角点裁剪文档区域

    Args:
        image: 输入图像
        corners: 四个角点 (4, 2)
        padding: 裁剪边距

    Returns:
        cropped_image: 裁剪后的图像
    """
    x_min = int(np.min(corners[:, 0])) - padding
    y_min = int(np.min(corners[:, 1])) - padding
    x_max = int(np.max(corners[:, 0])) + padding
    y_max = int(np.max(corners[:, 1])) + padding
    h, w = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    cropped = image[y_min:y_max, x_min:x_max]
    return cropped


def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    labels: Optional[list] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制角点

    Args:
        image: 输入图像
        corners: 角点 (4, 2)
        labels: 角点标签
        color: 颜色 (B, G, R)
        thickness: 线条粗细

    Returns:
        result: 绘制后的图像
    """
    result = image.copy()
    corners = corners.astype(np.int32)
    cv2.polylines(result, [corners], True, color, thickness)
    for i, corner in enumerate(corners):
        cv2.circle(result, tuple(corner), 5, color, -1)
        if labels:
            cv2.putText(
                result,
                str(labels[i]),
                tuple(corner + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
    return result


if __name__ == "__main__":
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.putText(
        image, "Test Document", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3
    )
    corners = np.array([[100, 50], [700, 100], [650, 550], [50, 500]], dtype=np.float32)
    image_with_corners = draw_corners(image, corners, labels=["TL", "TR", "BR", "BL"])
    transformer = PerspectiveTransformer(output_width=600, output_height=800)
    corrected = transformer.transform(image, corners)
    cv2.imwrite(
        "test_original.jpg", cv2.cvtColor(image_with_corners, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite("test_corrected.jpg", cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
    print("Test images saved!")
