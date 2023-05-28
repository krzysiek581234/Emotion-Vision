import cv2
import numpy as np
import os
from PIL import Image


def load_dataset(path):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    with open(path, 'r') as file:
        lines = file.readlines()

        for i, line in enumerate(lines):
            if i == 0:
                continue

            emotion, usage, pixels = line.split(",")
            pixels = np.array(pixels.split(), dtype='uint8')
            image = pixels.reshape((48, 48))

            if usage == 'Training':
                X_train.append(image)
                y_train.append(int(emotion))
            else:
                X_test.append(image)
                y_test.append(int(emotion))

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def save_image(image, label, prefix, index, save_path):
    class_folder = os.path.join(save_path, str(label))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    image_path = os.path.join(class_folder, f'{prefix}_{index}.png')
    cv2.imwrite(image_path, image)


def flip_image(image):
    flipped_image = np.fliplr(image)
    return flipped_image


def rotate_image(image, angle):
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image


def augment_dataset(X_train, y_train, save_path):
    augmented_X_train = []
    augmented_y_train = []

    for i in range(len(X_train)):
        image = X_train[i]
        label = y_train[i]

        # Original image
        augmented_X_train.append(image)
        augmented_y_train.append(label)
        save_image(image, label, 'original', i, save_path)

        # Flipped image
        flipped_image = flip_image(image)
        augmented_X_train.append(flipped_image)
        augmented_y_train.append(label)
        save_image(flipped_image, label, 'flipped', i, save_path)

        # Rotated images
        angles = [-10, 10]
        for angle in angles:
            rotated_image = rotate_image(image, angle)
            augmented_X_train.append(rotated_image)
            augmented_y_train.append(label)
            save_image(rotated_image, label, f'rotated{angle}', i, save_path)

    return np.array(augmented_X_train), np.array(augmented_y_train)

def save_test(X_test, y_test):
    for i in range(len(X_test)):
        image = X_test[i]
        label = y_test[i]
        save_image(image, label, label, i, 'test_images')

def save_basic_training(X_train, y_train):
    for i in range(len(X_train)):
        image = X_train[i]
        label = y_train[i]
        save_image(image, label, label, i, 'training_images')


# Load the FER2013 dataset
X_train, y_train, X_test, y_test = load_dataset('./data/icml_face_data.csv/icml_face_data.csv')

# Define the path to save the augmented images
save_path = 'augmented_images'

# Augment the training data and save the images to class subfolders
augmented_X_train, augmented_y_train = augment_dataset(X_train, y_train, save_path)

# Concatenate original and augmented training data
final_X_train = np.concatenate((X_train, augmented_X_train))
final_y_train = np.concatenate((y_train, augmented_y_train))

# Save test data images
save_test(X_test, y_test)

# Save basic train data images
save_basic_training(X_train, y_train)

# Print data augmentation info
print("Original training data shape:", X_train.shape)
print("Augmented training data shape:", augmented_X_train.shape)
print("Final training data shape:", final_X_train.shape)
