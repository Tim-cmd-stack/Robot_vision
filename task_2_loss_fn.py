import torch
import torch.nn as nn
import torch.nn.functional as F


class StereoDepthLoss(nn.Module):
    """Улучшенная функция потерь для стерео глубины"""

    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(StereoDepthLoss, self).__init__()
        self.alpha = alpha  # Вес L1 потерь
        self.beta = beta  # Вес градиентных потерь
        self.gamma = gamma  # Вес SSIM потерь

    def ssim_loss(self, pred, target, valid, window_size=11):
        """SSIM loss для сохранения структуры"""
        # Убедимся, что тензоры 4D [B, 1, H, W]
        if len(pred.shape) == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
            valid = valid.unsqueeze(1)

        # Применяем маску
        pred_masked = pred * valid
        target_masked = target * valid

        # Вычисляем SSIM
        ssim_val = self._ssim(pred_masked, target_masked, window_size)
        # SSIM ближе к 1 = лучше, поэтому loss = 1 - SSIM
        return (1 - ssim_val) * valid.mean()

    def _ssim(self, img1, img2, window_size=11):
        """Вычисление SSIM"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def gradient_loss(self, pred, target, valid):
        """Потери по градиентам для сохранения краев"""

        # Определяем размерность тензоров
        if len(pred.shape) == 3:  # [batch, height, width]
            # Вычисляем градиенты: обрезаем тензоры одинаково
            # Горизонтальные градиенты (по оси X) - обрезаем справа
            pred_grad_x = torch.abs(pred[:, :, :-1] - pred[:, :, 1:])
            target_grad_x = torch.abs(target[:, :, :-1] - target[:, :, 1:])
            valid_x = valid[:, :, :-1]

            # Вертикальные градиенты (по оси Y) - обрезаем снизу
            pred_grad_y = torch.abs(pred[:, :-1, :] - pred[:, 1:, :])
            target_grad_y = torch.abs(target[:, :-1, :] - target[:, 1:, :])
            valid_y = valid[:, :-1, :]

        else:  # [batch, channel, height, width]
            # Горизонтальные градиенты - обрезаем справа
            pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
            target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
            valid_x = valid[:, :, :, :-1]

            # Вертикальные градиенты - обрезаем снизу
            pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
            target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
            valid_y = valid[:, :, :-1, :]

        # Вычисляем потери по градиентам
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)

        # Применяем маску и вычисляем средние значения
        if valid_x.sum() > 0:
            grad_loss_x = (grad_diff_x * valid_x).sum() / valid_x.sum()
        else:
            grad_loss_x = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if valid_y.sum() > 0:
            grad_loss_y = (grad_diff_y * valid_y).sum() / valid_y.sum()
        else:
            grad_loss_y = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        return grad_loss_x + grad_loss_y

    def forward(self, pred, target, valid):
        # Обработка кортежа выходов
        if isinstance(pred, tuple):
            pred = pred[0]  # Берем основной выход

        # Убедимся, что размеры совпадают
        if pred.shape != target.shape:
            # Интерполяция pred к размеру target
            if len(pred.shape) == 3:
                pred = pred.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            if len(target.shape) == 3:
                target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

            pred = torch.nn.functional.interpolate(
                pred, size=target.shape[2:], mode='bilinear', align_corners=False
            )

            # Интерполяция valid если нужно
            if len(valid.shape) == 3:
                valid = valid.unsqueeze(1)
            valid = torch.nn.functional.interpolate(
                valid.float(), size=target.shape[2:], mode='nearest'
            )

            # Вернем к 3D если нужно
            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
                target = target.squeeze(1)
                valid = valid.squeeze(1)

        # L1 loss
        if valid.sum() > 0:
            l1_loss = (torch.abs(pred - target) * valid).sum() / valid.sum()
        else:
            l1_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Gradient loss
        grad_loss = self.gradient_loss(pred, target, valid)

        # SSIM loss
        ssim_loss = self.ssim_loss(pred, target, valid)

        total_loss = self.alpha * l1_loss + self.beta * grad_loss + self.gamma * ssim_loss
        return total_loss