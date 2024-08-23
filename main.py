import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

def load_data(train_dir, validation_dir, img_shape, batch_size):
    train_image_generator = ImageDataGenerator(rescale=1./255)
    validation_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=img_shape,
        class_mode='categorical'
    )

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=validation_dir,
        shuffle=False,
        target_size=img_shape,
        class_mode='categorical'
    )
    return train_data_gen, val_data_gen

def build_model(img_shape, num_classes):
    base_model = ResNet50(input_shape=img_shape + (3,), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_data_gen, val_data_gen, epochs, batch_size):
    history = model.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(train_data_gen.samples / float(batch_size))),
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(val_data_gen.samples / float(batch_size))),
        verbose=1
    )
    return history

def plot_history(history):
    epochs_range = range(len(history.history['accuracy']))
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', [0] * len(acc))
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', [0] * len(loss))

    # Ajustar o tamanho de acc, val_acc, loss e val_loss para ter o mesmo comprimento que epochs_range
    if len(acc) < len(epochs_range):
        acc.extend([0] * (len(epochs_range) - len(acc)))

    if len(val_acc) < len(epochs_range):
        val_acc.extend([0] * (len(epochs_range) - len(val_acc)))

    if len(loss) < len(epochs_range):
        loss.extend([0] * (len(epochs_range) - len(loss)))

    if len(val_loss) < len(epochs_range):
        val_loss.extend([0] * (len(epochs_range) - len(val_loss)))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./foo.png')
    plt.show()

def load_and_preprocess_image(img_path, img_shape):
    img = image.load_img(img_path, target_size=img_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizar a imagem da mesma forma que foi feito no treinamento
    return img_array

def classify_image(model, img_shape, class_indices):
    # Cria a janela de diálogo para selecionar o arquivo
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal do Tkinter
    img_path = filedialog.askopenfilename()  # Abre a janela de diálogo para selecionar o arquivo
    
    if img_path:  # Verifica se um arquivo foi selecionado
        img_array = load_and_preprocess_image(img_path, img_shape)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        class_labels = {v: k for k, v in class_indices.items()}  # Inverte o dicionário de índices para rótulos
        return img_path, class_labels[predicted_class[0]], prediction[0]
    else:
        return None, None, None

# Uso das funções
train_dir = 'dataset-resized/train'
validation_dir = 'dataset-resized/validation'
IMG_SHAPE = (512, 384)
BATCH_SIZE = 50
EPOCHS = 6
NUM_CLASSES = 6

train_data_gen, val_data_gen = load_data(train_dir, validation_dir, IMG_SHAPE, BATCH_SIZE)
model = build_model(IMG_SHAPE, NUM_CLASSES)
history = train_model(model, train_data_gen, val_data_gen, EPOCHS, BATCH_SIZE)
plot_history(history)

# Classificação de uma imagem selecionada pelo usuário
img_path, predicted_class, probabilities = classify_image(model, IMG_SHAPE, train_data_gen.class_indices)

if img_path:
    print(f"A imagem '{img_path}' foi classificada como: {predicted_class}")
    print(f"Probabilidades para cada classe: {probabilities}")
else:
    print("Nenhuma imagem foi selecionada.")
