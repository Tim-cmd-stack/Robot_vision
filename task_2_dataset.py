import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import cv2


def disparity_reader(path, image_size=(256, 256), teacher_depth_dir=None):
    """Функция для чтения диспарантности"""
    if path is None:
        return np.zeros(image_size, dtype=np.float32)

    try:
        # Загружаем как 16-bit (маски учителя сохранены с * 256)
        img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Не удалось загрузить {path}")

        disp = img.astype(np.float32)
        if len(disp.shape) == 3:
            disp = disp[..., 0]

        scale = 256.0
        disp /= scale

        # Resize если нужно
        if disp.shape[:2] != image_size:
            disp = cv2.resize(disp, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)

        return disp

    except Exception as e:
        print(f"Ошибка загрузки диспарантности {path}: {e}")
        return np.zeros(image_size, dtype=np.float32)


class InStereo2KSample:
    def __init__(self, root, teacher_depth_dir=None, is_test=False,
                 transform=None, image_size=(256, 256)):
        self.image_list = []
        self.disparity_list = []
        self.extra_info = []

        self.root = root
        self.teacher_depth_dir = teacher_depth_dir
        self.is_test = is_test

        # Нормализация для одноканальных изображений
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Для grayscale: mean=[0.5], std=[0.5]
        ])
        self.image_size = image_size

        # Проверяем существование пути
        if not Path(root).exists():
            print(f"ОШИБКА: Путь {root} не существует!")
            return

        # Заполняем списки
        folders = sorted([f for f in Path(root).iterdir() if f.is_dir()])
        print(f"Найдено {len(folders)} папок в {root}")

        for folder in folders:
            left_path = folder / "left.png"
            right_path = folder / "right.png"
            disp_path = None

            # Поиск карты диспарантности от учителя
            if teacher_depth_dir and not is_test:
                teacher_path = Path(teacher_depth_dir)
                if not teacher_path.exists():
                    print(f"ОШИБКА: Путь к teacher_depth_dir {teacher_path} не существует!")
                    continue
                # Ищем файлы вида {folder_name}_*.png
                pattern = f"{folder.name}_*.png"
                matching_files = sorted(teacher_path.glob(pattern))
                if matching_files:
                    disp_path = str(matching_files[0])  # Берем первый найденный файл
                else:
                    print(f"Не найдена disparity map для {folder.name} в {teacher_path}")

            # Проверяем существование входных изображений
            if left_path.exists() and right_path.exists():
                self.image_list.append([str(left_path), str(right_path)])
                self.disparity_list.append(disp_path)
                self.extra_info.append(str(folder.name))
            else:
                missing = []
                if not left_path.exists():
                    missing.append("left.png")
                if not right_path.exists():
                    missing.append("right.png")
                print(f"Пропущена папка {folder.name}: не найдены {', '.join(missing)}")

        print(f"Итого загружено {len(self.image_list)} стерео-пар")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # Загружаем изображения как градационные (как было изначально)
        left_path, right_path = self.image_list[index]
        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)  # Оставляем как было
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)  # Оставляем как было

        if left_img is None or right_img is None:
            raise ValueError(f"Не удалось загрузить изображения: {left_path}, {right_path}")

        # Resize
        if self.image_size:
            h, w = self.image_size
            left_img = cv2.resize(left_img, (w, h), interpolation=cv2.INTER_LINEAR)
            right_img = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # Загружаем диспарантность, если не в тестовом режиме
        disp = None
        if not self.is_test:
            disp_path = self.disparity_list[index]
            disp = disparity_reader(disp_path, self.image_size, self.teacher_depth_dir)
            if disp is None or np.all(disp == 0):
                print(f"ВНИМАНИЕ: Используется нулевая disp map для {disp_path}")
                disp = np.zeros(self.image_size, dtype=np.float32)

        # Преобразуем в тензоры
        left_img = Image.fromarray(left_img)
        right_img = Image.fromarray(right_img)
        left_img = self.transform(left_img)  # Shape: [1, H, W] для grayscale
        right_img = self.transform(right_img)  # Shape: [1, H, W] для grayscale

        if self.is_test:
            return left_img, right_img, self.extra_info[index]
        else:
            disp = torch.from_numpy(disp).float().unsqueeze(0)  # [1, H, W]
            return left_img, right_img, disp