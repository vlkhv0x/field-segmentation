"""
Предсказание сегментации для новых изображений
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('src')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='models/best_model.h5')
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()


def predict_segmentation(model, image_path, img_size=(256, 256)):
    """Предсказание сегментации"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, 0)
    
    prediction = model.predict(img_batch, verbose=0)[0]
    pred_mask = np.argmax(prediction, axis=-1)
    
    return img_array, pred_mask, prediction


def visualize_prediction(image, mask, class_names, save_path=None):
    """Визуализация предсказания"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    im = ax2.imshow(mask, cmap='tab10', vmin=0, vmax=len(class_names)-1)
    ax2.set_title('Segmentation Map', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Легенда
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046)
    cbar.set_label('Class', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Визуализация сохранена: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🔮 ПРЕДСКАЗАНИЕ СЕГМЕНТАЦИИ")
    print("=" * 80)
    
    if not os.path.exists(args.image_path):
        print(f"❌ Изображение не найдено: {args.image_path}")
        return
    
    # Загрузка модели
    import segmentation_models as sm
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'dice_loss_plus_1categorical_focal_loss': sm.losses.DiceLoss() + sm.losses.CategoricalFocalLoss(),
            'iou_score': sm.metrics.IOUScore(),
            'f1-score': sm.metrics.FScore()
        }
    )
    
    # Загрузка конфигурации
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    
    # Предсказание
    print(f"\n🔮 Сегментация изображения...")
    image, mask, predictions = predict_segmentation(model, args.image_path)
    
    # Статистика
    unique, counts = np.unique(mask, return_counts=True)
    
    print("\n📊 Результаты сегментации:")
    for class_id, count in zip(unique, counts):
        class_name = config['class_names'][class_id]
        percentage = (count / mask.size) * 100
        print(f"   {class_name}: {percentage:.1f}%")
    
    # Визуализация
    if args.visualize:
        save_path = f"reports/prediction_{os.path.basename(args.image_path)}.png"
        visualize_prediction(image, mask, config['class_names'], save_path)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
