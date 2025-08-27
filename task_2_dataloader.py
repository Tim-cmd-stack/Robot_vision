import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from task_2_dataset import InStereo2KSample

def get_dataloaders(stereo_root, teacher_depth_dir=None, batch_size=4,
                    train_val_split=0.8, num_workers=4, image_size=(512, 512)):
    """Создание DataLoader'ов на основе InStereo2KSample"""
    # Трансформации для тренировки
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Создаем датасет
    try:
        dataset = InStereo2KSample(
            root=stereo_root,
            teacher_depth_dir=teacher_depth_dir,
            is_test=False,
            transform=transform,
            image_size=image_size
        )
    except Exception as e:
        print(f"Ошибка создания датасета: {e}")
        return None, None

    # Проверяем размер датасета
    dataset_size = len(dataset)
    print(f"Размер датасета после создания: {dataset_size}")
    if dataset_size == 0:
        print(f"ВНИМАНИЕ: Датасет пуст! Проверьте путь: {stereo_root}")
        if teacher_depth_dir:
            print(f"Также проверьте путь к teacher_depth_dir: {teacher_depth_dir}")
        return None, None

    print(f"Размер датасета: {dataset_size}")

    # Разделяем на train/val
    indices = list(range(dataset_size))
    split = int(np.floor(train_val_split * dataset_size))

    np.random.seed(42)  # Для воспроизводимости
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # DataLoader'ы
    try:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"Успешно созданы DataLoader'ы: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    except Exception as e:
        print(f"Ошибка создания DataLoader'ов: {e}")
        return None, None

    return train_loader, val_loader

def get_test_dataloader(stereo_root, batch_size=1, num_workers=2, image_size=(512, 512)):
    """Создание тестового DataLoader'а"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    try:
        test_dataset = InStereo2KSample(
            root=stereo_root,
            teacher_depth_dir=None,  # Для тестирования нет ground truth
            is_test=True,
            transform=transform,
            image_size=image_size
        )
    except Exception as e:
        print(f"Ошибка создания тестового датасета: {e}")
        return None, None

    # Проверяем размер датасета
    dataset_size = len(test_dataset)
    print(f"Размер тестового датасета после создания: {dataset_size}")
    if dataset_size == 0:
        print(f"ВНИМАНИЕ: Тестовый датасет пуст! Проверьте путь: {stereo_root}")
        return None, None

    try:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"Успешно создан тестовый DataLoader: test_batches={len(test_loader)}")
    except Exception as e:
        print(f"Ошибка создания тестового DataLoader: {e}")
        return None, None

    return test_loader, None