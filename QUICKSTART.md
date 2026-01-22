# 🚀 Быстрый старт - Сегментация полей

## 📥 Получение данных

### EuroSAT Dataset

```bash
# Вариант 1: Прямое скачивание
wget http://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip EuroSAT.zip -d data/raw/

# Вариант 2: Kaggle
kaggle datasets download -d apollo2506/eurosat-dataset
unzip eurosat-dataset.zip -d data/raw/EuroSAT/
```

## ⚡ Быстрое обучение

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение U-Net (50 эпох)
python src/train.py --epochs 50 --batch_size 16

# Оценка с визуализацией
python src/evaluate.py --visualize --num_samples 10

# Сегментация нового изображения
python src/predict.py --image_path satellite.jpg --visualize
```

## 🎯 Параметры обучения

```bash
# С другим энкодером
python src/train.py --encoder resnet50 --epochs 50

# Меньший batch для слабых GPU
python src/train.py --batch_size 8 --epochs 50

# Настройка learning rate
python src/train.py --learning_rate 0.0005 --epochs 50
```

## 📊 Ожидаемые результаты

- **Mean IoU**: 0.80-0.85
- **Mean Dice**: 0.88-0.92
- **Pixel Accuracy**: 90-93%
- **Время обучения**: 
  - GPU: 60-90 минут
  - CPU: 5-8 часов

## 🗺️ Классы сегментации EuroSAT

1. **AnnualCrop** - Однолетние культуры
2. **Forest** - Лес
3. **HerbaceousVegetation** - Травянистая растительность
4. **Highway** - Дороги/шоссе
5. **Industrial** - Промышленная зона
6. **Pasture** - Пастбище
7. **PermanentCrop** - Многолетние культуры
8. **Residential** - Жилая зона
9. **River** - Река/водоём
10. **SeaLake** - Море/озеро

## 💡 Использование модели

### Python API

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Загрузка модели
model = tf.keras.models.load_model('models/best_model.h5')

# Загрузка изображения
img = Image.open('satellite.jpg').resize((256, 256))
img_array = np.array(img) / 255.0
img_batch = np.expand_dims(img_array, 0)

# Предсказание
prediction = model.predict(img_batch)[0]
segmentation_map = np.argmax(prediction, axis=-1)

# Получение статистики
unique, counts = np.unique(segmentation_map, return_counts=True)
for class_id, count in zip(unique, counts):
    percentage = (count / segmentation_map.size) * 100
    print(f"Class {class_id}: {percentage:.1f}%")
```

## 📐 Метрики сегментации

**IoU (Intersection over Union)**
- Мера перекрытия между предсказанием и ground truth
- Диапазон: 0-1 (1 = идеальное совпадение)

**Dice Coefficient**
- Похож на IoU, но более чувствителен к размеру объекта
- Формула: 2 * |A ∩ B| / (|A| + |B|)

**Pixel Accuracy**
- Процент правильно классифицированных пикселей
- Простая, но может быть misleading для несбалансированных классов

## 🔍 Анализ результатов

Результаты сохраняются в `reports/`:
- `segmentation_results.png` - примеры сегментации
- `training_history.json` - история обучения
- `metrics_plot.png` - графики метрик

## 🐛 Решение проблем

**Ошибка памяти (OOM)**
```bash
python src/train.py --batch_size 8  # или 4
```

**Низкий IoU**
- Увеличьте epochs
- Попробуйте другой encoder
- Проверьте качество масок

**Долгое обучение**
- Уменьшите размер изображения до 128x128
- Используйте более легкий encoder (mobilenetv2)

## 🌟 Продвинутые возможности

### 1. Экспорт масок в GeoTIFF

```python
import rasterio
from rasterio.transform import from_bounds

# Сохранение с геопривязкой
with rasterio.open(
    'output.tif', 'w',
    driver='GTiff',
    height=mask.shape[0],
    width=mask.shape[1],
    count=1,
    dtype=mask.dtype,
    crs='+proj=latlong',
    transform=from_bounds(west, south, east, north, width, height)
) as dst:
    dst.write(mask, 1)
```

### 2. Постобработка CRF

```python
import pydensecrf.densecrf as dcrf

# Применение CRF для сглаживания
d = dcrf.DenseCRF2D(w, h, n_classes)
d.setUnaryEnergy(unary)
d.addPairwiseGaussian(sxy=3, compat=3)
Q = d.inference(5)
refined_mask = np.argmax(Q, axis=0).reshape((h, w))
```

### 3. Пакетная обработка

```bash
# Сегментация всех изображений в папке
for img in data/test/*.jpg; do
    python src/predict.py --image_path "$img" --visualize
done
```

## 📚 Полезные ресурсы

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Segmentation Models Documentation](https://segmentation-models.readthedocs.io/)
- [EuroSAT Paper](https://arxiv.org/abs/1709.00029)
- [Sentinel-2 Data](https://sentinel.esa.int/)

## 💬 Поддержка

При возникновении проблем проверьте:
1. Версии библиотек: `pip list`
2. Наличие данных: `ls data/raw/EuroSAT/`
3. Доступную память: `nvidia-smi` (для GPU)
