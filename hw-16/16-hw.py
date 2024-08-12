import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Завантаження моделей
@st.cache_resource
def load_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


@st.cache_resource
def load_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


cnn_model = load_cnn_model()
vgg16_model = load_vgg16_model()

# Інтерфейс Streamlit
st.title('Класифікація зображень за допомогою CNN та VGG16')

model_option = st.selectbox('Виберіть модель для класифікації', ['CNN', 'VGG16'])

uploaded_file = st.file_uploader("Завантажте зображення для класифікації", type=["jpg", "png"])

if uploaded_file is not None:
    # Завантаження та передобробка зображення
    image = Image.open(uploaded_file).convert('L') if model_option == 'CNN' else Image.open(uploaded_file).convert(
        'RGB')
    st.image(image, caption='Завантажене зображення', use_column_width=True)

    image = image.resize((28, 28)) if model_option == 'CNN' else image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    if model_option == 'CNN':
        image = np.expand_dims(image, axis=-1)

    # Передбачення класу
    model = cnn_model if model_option == 'CNN' else vgg16_model
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    st.write(f"Передбачений клас: {predicted_class}")
    st.write("Ймовірності для кожного класу:")
    st.write(predictions)

# Графіки функції втрат та точності
st.subheader("Графіки функції втрат та точності")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

history = cnn_model.history if model_option == 'CNN' else vgg16_model.history

# Графік втрат
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Validation Loss')
ax[0].set_title('Функція втрат')
ax[0].legend()

# Графік точності
ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[1].set_title('Точність')
ax[1].legend()

st.pyplot(fig)
