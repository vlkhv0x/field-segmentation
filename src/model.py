"""
Архитектура U-Net для сегментации
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import segmentation_models as sm


class UNetSegmentation:
    """U-Net модель для сегментации полей"""
    
    def __init__(self, num_classes, img_size=(256, 256), encoder='resnet34'):
        self.num_classes = num_classes
        self.img_size = img_size
        self.encoder = encoder
        self.model = None
    
    def build_model(self):
        """Построение U-Net с предобученным энкодером"""
        
        # Используем segmentation_models библиотеку
        model = sm.Unet(
            self.encoder,
            input_shape=(*self.img_size, 3),
            classes=self.num_classes,
            activation='softmax',
            encoder_weights='imagenet'
        )
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.0001):
        """Компиляция модели"""
        
        # Комбинированная loss-функция
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        
        # Метрики
        metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
            'accuracy'
        ]
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=total_loss,
            metrics=metrics
        )
        
        print("✅ Модель скомпилирована")
    
    def get_callbacks(self, checkpoint_path='models/best_model.h5'):
        """Callbacks для обучения"""
        return [
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True, monitor='val_iou_score', mode='max'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7
            )
        ]
    
    def train(self, train_gen, val_gen, epochs=50, callbacks=None):
        """Обучение"""
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        print(f"\n🚀 Начало обучения U-Net ({self.encoder})")
        
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Обучение завершено!")
        return history
    
    def save_model(self, filepath='models/final_model.h5'):
        """Сохранение модели"""
        self.model.save(filepath)
        print(f"✅ Модель сохранена: {filepath}")


# Генератор данных для сегментации
class SegmentationDataGenerator(keras.utils.Sequence):
    """Генератор для загрузки изображений и масок"""
    
    def __init__(self, df, batch_size, img_size, num_classes, augment=False):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
    
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        masks = []
        
        for _, row in batch_df.iterrows():
            # Загрузка
            from data_preprocessing import load_image_and_mask
            img, mask = load_image_and_mask(
                row['image_path'], row['mask_path'], self.img_size
            )
            
            images.append(img)
            
            # One-hot encoding маски
            mask_onehot = keras.utils.to_categorical(mask, self.num_classes)
            masks.append(mask_onehot)
        
        return np.array(images), np.array(masks)


if __name__ == "__main__":
    import numpy as np
    
    model_builder = UNetSegmentation(num_classes=10, encoder='resnet34')
    model = model_builder.build_model()
    model_builder.compile_model()
    
    print(f"\n📊 Параметров: {model.count_params():,}")
