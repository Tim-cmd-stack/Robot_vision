import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
bgnet_path = os.path.join(project_root, 'BGNet')

sys.path.insert(0, bgnet_path)

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import numpy as np
from task_2_dataloader import get_dataloaders, get_test_dataloader
from task_2_loss_fn import StereoDepthLoss
from BGNet.models.bgnet_plus import BGNet_Plus
import cv2
from pathlib import Path


def train_stereo_depth():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Конфигурация
    batch_size = 2
    epochs = 40
    lr = 1e-4
    log_dir = 'results/task_2_logs_stereo'
    save_dir = 'results/task_2_checkpoints_stereo'

    os.makedirs(save_dir, exist_ok=True)

    # Создание датасета
    try:
        train_loader, val_loader = get_dataloaders(
            stereo_root=r'data\instereo2k_sample',
            teacher_depth_dir=r'results\1st_task_disparity_instereo2k_sample',
            batch_size=batch_size,
            train_val_split=0.8,
            num_workers=4,
            image_size=(256, 256)
        )

        if train_loader is None or val_loader is None:
            print("Не удалось создать датасет. Завершение.")
            return

    except Exception as e:
        print(f"Ошибка создания датасета: {e}")
        return

    # МОДЕЛЬ - Загрузка предобученных весов
    model = BGNet_Plus().to(device)

    # Путь к предобученным весам (без использования весов)
    pretrained_path = r''

    try:
        # Загрузка предобученных весов
        checkpoint = torch.load(pretrained_path, map_location=device)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"Успешно загружены предобученные веса из {pretrained_path}")

    except Exception as e:
        print(f"Ошибка загрузки предобученных весов: {e}")
        print("Обучение начнется с нуля...")

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr}
    ], weight_decay=1e-5)

    criterion = StereoDepthLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    writer = SummaryWriter(log_dir)

    # Переменные для отслеживания прогресса
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_steps = 0

        for i, (left_imgs, right_imgs, targets) in enumerate(train_loader):
            left_imgs = left_imgs.to(device)
            right_imgs = right_imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(left_imgs, right_imgs)

            # Обработка выхода модели
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Маска для валидных значений
            valid = (targets > 0.1).float()
            loss = criterion(outputs, targets, valid)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            if i % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('Loss/train_step', loss.item(), epoch * len(train_loader) + i)

        # Validation phase
        model.eval()
        val_loss = 0
        val_rmse = 0
        val_absrel = 0
        val_steps = 0

        with torch.no_grad():
            for left_imgs, right_imgs, targets in val_loader:
                left_imgs = left_imgs.to(device)
                right_imgs = right_imgs.to(device)
                targets = targets.to(device)

                outputs = model(left_imgs, right_imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                valid = (targets > 0.1).float()
                loss = criterion(outputs, targets, valid)

                val_loss += loss.item()

                # Метрики
                rmse = torch.sqrt(torch.mean((outputs - targets) ** 2 * valid))

                mask = targets > 0.1
                if mask.sum() > 0:
                    outputs_flat = outputs.flatten()
                    targets_flat = targets.flatten()
                    mask_flat = mask.flatten()

                    absrel = torch.mean(
                        torch.abs(outputs_flat[mask_flat] - targets_flat[mask_flat]) /
                        (targets_flat[mask_flat] + 1e-6)
                    )
                else:
                    absrel = torch.tensor(0.0, device=outputs.device)

                val_rmse += rmse.item()
                val_absrel += absrel.item() if isinstance(absrel, torch.Tensor) else absrel
                val_steps += 1

        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps
        avg_val_rmse = val_rmse / val_steps
        avg_val_absrel = val_absrel / val_steps

        scheduler.step()

        # Логирование
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Val_RMSE', avg_val_rmse, epoch)
        writer.add_scalar('Metrics/Val_AbsRel', avg_val_absrel, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch [{epoch + 1}/{epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val RMSE: {avg_val_rmse:.4f}')
        print(f'  Val AbsRel: {avg_val_absrel:.4f}')

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_epoch': best_epoch
            }, os.path.join(save_dir, f'bgnet_model_best.pth'))

            print(f"Сохранена лучшая модель (эпоха {best_epoch})")

        # Сохранение checkpoint
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(save_dir, f'bgnet_model_epoch_{epoch + 1}.pth'))

        # Визуализация
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                left_sample, right_sample, target_sample = next(iter(val_loader))
                left_sample = left_sample.to(device)
                right_sample = right_sample.to(device)
                pred_sample = model(left_sample, right_sample)

                if isinstance(pred_sample, tuple):
                    pred_sample = pred_sample[0]

                # Денормализация изображений
                mean = torch.tensor([0.5]).view(1, 1, 1).to(device)
                std = torch.tensor([0.5]).view(1, 1, 1).to(device)
                left_denorm = left_sample * std + mean

                # Min-max нормализация для диспарантности
                pred_norm = (pred_sample - pred_sample.min()) / (pred_sample.max() - pred_sample.min() + 1e-6)
                target_norm = (target_sample - target_sample.min()) / (target_sample.max() - target_sample.min() + 1e-6)

                grid_left = vutils.make_grid(left_denorm[:4], normalize=True, nrow=2)
                grid_pred = vutils.make_grid(pred_norm[:4], normalize=True, nrow=2)
                grid_target = vutils.make_grid(target_norm[:4], normalize=True, nrow=2)

                writer.add_image('Val/Left_Input', grid_left, epoch)
                writer.add_image('Val/Prediction', grid_pred, epoch)
                writer.add_image('Val/Target', grid_target, epoch)

    writer.close()
    print(f"Лучшая модель на эпохе {best_epoch}")


def test_stereo_model(model_path, test_data_path,
                      teacher_depth_dir=r'results\1st_task_disparity_instereo2k_sample',
                      output_dir='results/task_2_test_predictions'):
    """Тестирование обученной стерео модели с сохранением результатов и target картинок"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка модели
    model = BGNet_Plus()
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Ошибка загрузки модели из {model_path}: {e}")
        return

    model.to(device)
    model.eval()

    print(f"Загружена модель с эпохи {checkpoint['epoch'] + 1}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"Val Loss: {checkpoint['val_loss']:.4f}")

    # Создание тестового датасета
    test_loader, _ = get_test_dataloader(
        stereo_root=test_data_path,
        batch_size=1,
        num_workers=2,
        image_size=(256, 256)
    )

    if test_loader is None:
        print("Не удалось создать тестовый датасет. Завершение.")
        return

    # Создание папки для результатов
    os.makedirs(output_dir, exist_ok=True)

    # Функция для загрузки disparity из PNG
    def load_disparity_from_png(path):
        """Загрузка disparity из PNG файла"""
        img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Не удалось загрузить {path}")
        disp = img.astype(np.float32)
        if len(disp.shape) == 3:
            disp = disp[..., 0]
        return disp

    # Улучшенная функция нормализации
    def robust_normalize_for_visualization(disp, clip_percentile=95):
        """
        Улучшенная нормализация для визуализации
        """
        # Убираем выбросы
        valid_pixels = disp[disp > 0]  # Только валидные пиксели
        if len(valid_pixels) > 0:
            max_val = np.percentile(valid_pixels, clip_percentile)
        else:
            max_val = disp.max()

        disp_clipped = np.clip(disp, 0, max_val)

        # Min-max нормализация
        min_val = disp_clipped.min()
        max_val = disp_clipped.max()

        if max_val - min_val > 1e-6:
            disp_normalized = ((disp_clipped - min_val) / (max_val - min_val + 1e-6) * 255).astype(np.uint8)
        else:
            disp_normalized = np.zeros_like(disp_clipped, dtype=np.uint8)

        return disp_normalized

    # Сохранение target картинок
    print("Сохранение target картинок для сравнения...")
    teacher_path = Path(teacher_depth_dir)
    if teacher_path.exists():
        with torch.no_grad():
            for i, (left_img, right_img, extra) in enumerate(test_loader):
                folder_name = extra[0]
                # Поиск соответствующей target маски
                pattern = f"{folder_name}_*.png"
                matching_files = sorted(teacher_path.glob(pattern))

                if matching_files:
                    target_file = matching_files[0]  # Берем первый найденный файл
                    try:
                        # Загружаем и восстанавливаем target disparity
                        target_disparity = load_disparity_from_png(target_file)
                        # Resize до (256, 256) для соответствия test_loader
                        target_disparity = cv2.resize(target_disparity, (256, 256), interpolation=cv2.INTER_NEAREST)

                        # Улучшенная нормализация
                        target_normalized = robust_normalize_for_visualization(target_disparity)

                        # Сохраняем в test_predictions
                        target_output_path = os.path.join(output_dir, f'target_{folder_name}.png')
                        cv2.imwrite(target_output_path, target_normalized)
                        print(f"  Сохранен target: {target_output_path}")

                    except Exception as e:
                        print(f"  Ошибка при сохранении target {target_file.name}: {e}")
                else:
                    print(f"  Не найдена target маска для {folder_name}")

                if i >= 10:  # Ограничиваем количество
                    break
    else:
        print(f"  Папка с target данными не найдена: {teacher_depth_dir}")

    # Тестирование
    with torch.no_grad():
        for i, (left_img, right_img, extra) in enumerate(test_loader):
            left_img = left_img.to(device)
            right_img = right_img.to(device)

            pred = model(left_img, right_img)

            # BGNet_Plus возвращает кортеж (out1, out2)
            if isinstance(pred, tuple):
                pred = pred[0]  # Берем основное предсказание (out1)

            # Получаем numpy массив
            pred_np = pred.cpu().numpy()[0]  # [H, W]

            # Улучшенная нормализация для визуализации
            pred_img = robust_normalize_for_visualization(pred_np)

            pred_path = os.path.join(output_dir, f'pred_{extra[0]}.png')
            cv2.imwrite(pred_path, pred_img)

            print(f"Сохранено предсказание: {pred_path}")
            if i >= 10:
                break

    print(f"Тестирование завершено! Результаты сохранены в {output_dir}")


if __name__ == "__main__":
    #train_stereo_depth()
    test_stereo_model(
        model_path=r'results\task_2_checkpoints_stereo\bgnet_model_epoch_40.pth',
        test_data_path=r'dl-cv-home-test-master\data\instereo2k_sample'
    )



