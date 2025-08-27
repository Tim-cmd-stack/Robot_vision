import torch
import numpy as np
import cv2
import os
from pathlib import Path
import random
import sys
import argparse

# Настройки путей
PROJECT_ROOT = Path(__file__).parent
#print(PROJECT_ROOT)
RAFT_DIR = PROJECT_ROOT / "RAFT-Stereo"
core_path = RAFT_DIR / "core"
# Добавляем пути
sys.path.insert(0, str(core_path))
sys.path.insert(0, str(RAFT_DIR))

# Импорты
from raft_stereo import RAFTStereo
from utils.utils import InputPadder

# Импорт классов из клонированного репозитория
from stereo_datasets import StereoDataset


# ================== Новый датасет для InStereo2K Sample ==================
class InStereo2KSample(StereoDataset):
    def __init__(self, root, is_test=True):
        super(InStereo2KSample, self).__init__()
        self.root = root
        self.is_test = True  # Всегда в режиме теста для инференса

        # Найдем все папки с данными
        folders = sorted([f for f in Path(root).iterdir() if f.is_dir()])

        self.image_list = []
        self.disparity_list = []

        for folder in folders:
            left_path = folder / "left.png"
            right_path = folder / "right.png"
            disp_path = folder / "left_disp.png"  # или "right_disp.png"

            if left_path.exists() and right_path.exists():
                self.image_list.append([str(left_path), str(right_path)])
                self.disparity_list.append(str(disp_path))
                self.extra_info.append(str(folder.name))

        print(f"Найдено {len(self.image_list)} стерео-пар")


# ================== Настройки ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = r"dl-cv-home-test-master\data\instereo2k_sample"
OUT_DIR = "results/1st_task_disparity_instereo2k_sample"
MODEL_PATH = "RAFT-Stereo/models/raftstereo-middlebury.pth"

os.makedirs(OUT_DIR, exist_ok=True)


# ================== Создание аргументов ==================
class Args:
    def __init__(self):
        self.hidden_dims = [128] * 3
        self.corr_implementation = "reg"
        self.shared_backbone = False
        self.corr_levels = 4
        self.corr_radius = 4
        self.n_downsample = 2
        self.context_norm = "batch"
        self.slow_fast_gru = False
        self.n_gru_layers = 3
        self.mixed_precision = False


# ================== Загрузка модели ==================
def load_model():
    args = Args()
    model = RAFTStereo(args)

    if os.path.exists(MODEL_PATH):
        print(f"Загружаю модель из: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # Обработка DataParallel
        if 'module.' in list(checkpoint.keys())[0]:
            new_checkpoint = {}
            for key, value in checkpoint.items():
                new_key = key.replace('module.', '')
                new_checkpoint[new_key] = value
            checkpoint = new_checkpoint

        model.load_state_dict(checkpoint, strict=False)
        print("✅ Модель загружена")
    else:
        print(f"❌ Файл модели не найден: {MODEL_PATH}")
        return None

    model = model.to(DEVICE)
    model.eval()
    return model


# ================== Инференс ==================
@torch.no_grad()
def run_inference(model, left_img, right_img, iters=32):
    # Преобразуем в тензоры
    left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).float()[None].to(DEVICE)
    right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).float()[None].to(DEVICE)

    # Padding
    padder = InputPadder(left_tensor.shape, divis_by=32)
    left_pad, right_pad = padder.pad(left_tensor, right_tensor)

    # Инференс
    use_mixed_precision = True
    with torch.cuda.amp.autocast(enabled=use_mixed_precision):
        _, flow_pr = model(left_pad, right_pad, iters=iters, test_mode=True)

    # Убираем padding
    flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

    # Извлекаем диспаритет (первый канал)
    disparity = flow_pr[0].numpy()
    return disparity


# ================== Визуализация и сохранение результатов при необходимости==================
'''def visualize_disparity_colormap(disparity_map, save_name):
    """Визуализация карты диспаритета"""
    print(f"Визуализация диспаритета для: {save_name}")
    print(f"Диапазон: {disparity_map.min():.4f} - {disparity_map.max():.4f}")
    print(f"Среднее: {disparity_map.mean():.4f}")
    print(f"Нулевых значений: {(disparity_map == 0).sum()} из {disparity_map.size}")

    # Создаем фигуру с тремя подграфиками
    plt.figure(figsize=(18, 5))

    # 1. Исходная карта диспаритета с цветовой схемой 'plasma'
    plt.subplot(1, 2, 1)
    plt.imshow(disparity_map, cmap='plasma')
    plt.colorbar(label='Disparity (raw)')
    plt.title('Raw Disparity Map')

    # 2. Гистограмма значений
    plt.subplot(1, 2, 2)
    plt.hist(disparity_map.flatten(), bins=100, alpha=0.7)
    plt.xlabel('Disparity Value')
    plt.ylabel('Frequency')
    plt.title('Disparity Distribution')
    plt.yscale('log')

    plt.tight_layout()

    # Сохраняем
    #save_path = os.path.join(OUT_DIR, f"{save_name}_analysis.png")
    #plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #plt.close()
    #print(f"📊 Визуализация сохранена: {save_path}")'''


def save_disparity(disparity, save_name):
    """Сохранение disparity map для использования как метки"""

    # Создаём папку, если её нет
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Сохраняем как .png с реальными значениями (KITTI-style)
    disparity_uint16 = (disparity * 256).astype(np.uint16)
    cv2.imwrite(os.path.join(OUT_DIR, f"{save_name}.png"), disparity_uint16)

    #disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    #disp_vis = disp_vis.astype(np.uint8)
    #cv2.imwrite(os.path.join(OUT_DIR, f"{save_name}_vis.png"), disp_vis)

    print(f"✅ Сохранено: {save_name} (.png)")

# ================== Основная функция ==================
def main():
    print("=== Инференс RAFT-Stereo на InStereo2K Sample ===")

    # ================== Аргументы командной строки ==================
    parser = argparse.ArgumentParser(description='Инференс RAFT-Stereo на датасете')
    parser.add_argument('--mode', choices=['random', 'all', 'first'],
                        default='random', help='Режим обработки')
    parser.add_argument('--count', type=int, default=10,
                        help='Количество сэмплов (для random/first)')
    args = parser.parse_args()

    # Загружаем модель
    model = load_model()
    if model is None:
        return

    # Создаем датасет
    dataset = InStereo2KSample(DATA_DIR, is_test=True)
    total_samples = len(dataset)
    print(f"Всего пар в датасете: {total_samples}")

    if total_samples == 0:
        print("❌ Нет данных для обработки")
        return

    # ================== Выбор индексов в зависимости от режима ==================
    if args.mode == 'all':
        indices = list(range(total_samples))
        print(f"Обработка всего датасета: {len(indices)} сэмплов")
    elif args.mode == 'random':
        count = min(args.count, total_samples)
        indices = random.sample(range(total_samples), count)
        print(f"Обработка {count} случайных сэмплов")
    elif args.mode == 'first':
        count = min(args.count, total_samples)
        indices = list(range(count))
        print(f'Обработка первых {count} сэмплов')

    # ================== Обработка ==================
    print(f"Начинаем обработку {len(indices)} сэмплов...")

    successful = 0
    failed = 0

    for i, idx in enumerate(indices):
        print(f"\n[{i + 1}/{len(indices)}] Обработка индекса {idx}...")

        try:
            # Получаем данные
            left_tensor, right_tensor, folder_name = dataset[idx]

            # Конвертируем тензоры в numpy
            left_img = left_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            right_img = right_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

            # Инференс
            disparity = run_inference(model, left_img, right_img)
            save_disparity(disparity, f"{folder_name}_{idx}")
            successful += 1

        except Exception as e:
            print(f"❌ Ошибка при обработке индекса {idx}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

    print(f"\n✅ Инференс завершён!")
    print(f"   Успешно: {successful}")
    print(f"   Ошибок:  {failed}")
    print(f"   Результаты сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    main()