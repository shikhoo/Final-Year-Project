# Final-Year-Project
 
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

img_width, img_height = 150, 150
batch_size = 32

datasets = {
    "flower": {
        "data_dir": "flower_dataset",
        "task_name": "Flower Species Classification"
    },
    "plant_disease": {
        "data_dir": "PlantVillage",
        "task_name": "Plant Disease Detection"
    }
}

task_models = {}

def load_data(task):
    data_dir = datasets[task]["data_dir"]
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model(task):
    if task in task_models:
        print(f"Using loaded model for {datasets[task]['task_name']}.")
        return task_models[task]

    model_filename = f"{task}_model.h5"
    train_generator, validation_generator = load_data(task)
    num_classes = len(train_generator.class_indices)

    if os.path.exists(model_filename):
        print(f"Loading pre-trained model for {datasets[task]['task_name']}...")
        model = load_model(model_filename)
    else:
        print(f"Training new model for {datasets[task]['task_name']}...")
        model = build_model(num_classes)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            train_generator,
            epochs=25,
            validation_data=validation_generator,
            callbacks=[early_stop]
        )
        model.save(model_filename)
        print(f"Model saved as {model_filename}.")

    task_models[task] = model  # Store the model in memory
    return model

def predict_images_in_loop(model, class_indices):
    while True:
        test_image_path = input("\nEnter the path of an image to predict (or type 'exit' to quit): ")
        if test_image_path.lower() == 'exit':
            print("Exiting prediction loop.")
            break
        try:
            img = Image.open(test_image_path).resize((img_width, img_height))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            prediction = model.predict(img_array)
            predicted_label = list(class_indices.keys())[np.argmax(prediction)]

            print(f"Predicted Label: {predicted_label}")
            plt.imshow(img)
            plt.title(f"Predicted: {predicted_label}")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error processing image: {e}")

def main():
    while True:
        print("\nSelect a task:")
        for i, key in enumerate(datasets.keys(), 1):
            print(f"{i}. {datasets[key]['task_name']}")
        print("3. Exit")

        task_choice = int(input("Enter the task number: ")) - 1
        if task_choice == 2:
            print("Exiting program.")
            break

        task_key = list(datasets.keys())[task_choice]
        model = get_model(task_key)
        train_generator, _ = load_data(task_key)
        predict_images_in_loop(model, train_generator.class_indices)

if __name__ == "__main__":
    main()
