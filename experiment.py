"""
This module is designed for image classification of muffins and chihuahuas. 

Key Features:
- Uses Keras and Keras Tuner for neural network and hyperparameter optimization.
- Data augmentation using ImageDataGenerator.
- Stratified K-Fold cross-validation for evaluating models.
- Provides three different CNN architectures for experimentation: 
  a simple sequential model, a simple CNN model, and a more complex CNN model.
- Evaluates models based on Accuracy, AUC, and F1 Score and visualizes the training history.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers.legacy import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score


# Set constants for reproducibility and ease of use
SEED = 45
BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_FOLDS = 5


# Create a ImageDataGenerator instance with augmentation options.
datagen = ImageDataGenerator(
    rotation_range=20,  # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # Randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # Randomly shift images vertically (fraction of total height)
    shear_range=0.2,  # Set range for random shear
    zoom_range=0.2,  # Set range for random zoom
    horizontal_flip=True,  # Randomly flip inputs horizontally
    fill_mode='nearest'  # Points outside the boundaries of the input are filled according to the given mode
)

def plot_training_history(history, title='Training History'):
    """
    This function plots the training history.
    
    Args:
        history (History): History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs.
        title (str): Title for the plot.
    
    Returns:
        None
    """
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evaluate_model(y_true, y_pred_probability, threshold=0.5):
    """
    Evaluate the model using various metrics.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels.
        y_pred_probability (numpy.ndarray): Predicted probabilities.
        threshold (float): Threshold for binary classification.
    
    Returns:
        tuple: A tuple containing accuracy, AUC, and F1 score.
    """
    y_pred = (y_pred_probability > threshold).astype(int)

    accuracy_score = np.mean(y_true == y_pred)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    f1_scr = f1_score(y_true, y_pred)

    return accuracy_score, auc_score, f1_scr

def load_data(path):
    """
    This function loads all the images from the path provided,
    and categorizes them into muffins and chihuahuas.
    
    Args:
        path (str): The root directory where the images are stored.
    
    Returns:
        numpy.ndarray: All images in np.array format.
        numpy.ndarray: Labels for the images in np.array format.
    """
    images = []
    labels = []
    for first_folder_name in ['train', 'test']:
        first_folder_path = os.path.join(path, first_folder_name)
        for second_folder_name in ['muffin', 'chihuahua']:
            second_folder_path = os.path.join(first_folder_path, second_folder_name)
            for img_name in os.listdir(second_folder_path):
                img_path = os.path.join(second_folder_path, img_name)
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
                img_array = img_to_array(img) / 255.
                images.append(img_array)
                labels.append(1 if second_folder_name == 'muffin' else 0)
    return np.array(images), np.array(labels)

# Load all data
X, y = load_data('../project/archive')


def build_simple_sequential_model(hp):
    """
    This function returns a simple sequential model for hyperparameter tuning.

    Args:
        hp (keras_tuner.HyperParameters): Hyperparameters to tune.

    Returns:
        keras.Model: The compiled model.
    """
    model1 = Sequential()
    model1.add(Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model1.add(Dense(hp.Int('units', min_value=192, max_value=256, step=32), activation='relu'))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])), metrics=['accuracy'])
    return model1

def build_simple_cnn_model(hp):
    """
    This function returns a simple CNN model for hyperparameter tuning.

    Args:
        hp (keras_tuner.HyperParameters): Hyperparameters to tune.

    Returns:
        keras.Model: The compiled model.
    """
    model2 = Sequential()
    model2.add(Conv2D(hp.Int('input_units', min_value=192, max_value=256, step=32), (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Conv2D(hp.Int('hidden_units', min_value=192, max_value=256, step=32), (3, 3), activation='relu'))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dense(hp.Int('dense_units', min_value=128, max_value=192, step=32), activation='relu'))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4])), metrics=['accuracy'])
    return model2

def build_complex_cnn_model(hp):
    """
    This function returns a complex CNN model for hyperparameter tuning.

    Args:
        hp (keras_tuner.HyperParameters): Hyperparameters to tune.

    Returns:
        keras.Model: The compiled model.
    """
    model3 = Sequential()
    model3.add(Conv2D(hp.Int('input_units', min_value=128, max_value=256, step=32), (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
    model3.add(MaxPooling2D((2, 2)))
    model3.add(Conv2D(hp.Int('hidden_units_1', min_value=128, max_value=256, step=32), (3, 3), activation='relu'))
    model3.add(MaxPooling2D((2, 2)))
    model3.add(Conv2D(hp.Int('hidden_units_2', min_value=128, max_value=256, step=32), (3, 3), activation='relu'))
    model3.add(MaxPooling2D((2, 2)))
    model3.add(Conv2D(hp.Int('hidden_units_3', min_value=128, max_value=256, step=32), (3, 3), activation='relu'))
    model3.add(MaxPooling2D((2, 2)))
    model3.add(Flatten())
    model3.add(Dense(hp.Int('dense_units', min_value=128, max_value=256, step=32), activation='relu'))
    model3.add(Dense(1, activation='sigmoid'))
    model3.compile(loss='binary_crossentropy', optimizer=Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4])), metrics=['accuracy'])
    return model3


# List of models
models = [build_simple_sequential_model, build_simple_cnn_model, build_complex_cnn_model]

# Dictionary to store the scores, best models, and their training histories for each model
results = {model.__name__: {'scores': [], 'best_models': [], 'histories': []} for model in models}

# Define StratifiedKFold (Stratified is often a better choice for classification problems)
k_fold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for build_model in models:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Change the directory for RandomSearch to avoid conflicts
    dir_name = f'random_search_{build_model.__name__}_{int(time.time())}'

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        seed=SEED,
        directory=dir_name
    )

    tuner.search_space_summary()

    datagen.fit(X_train)

    # Fit the model
    tuner.search(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                 epochs=15,
                 validation_data=(X_test, y_test))

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)

    # For each fold, create a new model, perform hyperparameter tuning, then test the model
    for train_index, test_index in k_fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Compute quantities required for featurewise normalization
        datagen.fit(X_train)

        model_history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                  epochs=18,
                                  validation_data=(X_test, y_test))

        # Save the best model and its training history
        results[build_model.__name__]['best_models'].append(model)
        results[build_model.__name__]['histories'].append(model_history)

        # Save the best model to a file
        model.save(f'{build_model.__name__}_best_model_fold_{len(results[build_model.__name__]["best_models"])}.h5')

        # Compute AUC and F1 score along with accuracy
        y_pred_prob = model.predict(X_test).squeeze()
        accuracy, auc, f1 = evaluate_model(y_test, y_pred_prob)

        results[build_model.__name__]['scores'].append({'accuracy': accuracy, 'AUC': auc, 'F1': f1})


# Print the scores for each model and plot their training history
for model_name, model_info in results.items():
    acc_scores = [score['accuracy'] for score in model_info['scores']]
    auc_scores = [score['AUC'] for score in model_info['scores']]
    f1_scores = [score['F1'] for score in model_info['scores']]

    print(f"{model_name}:")
    print(f"Accuracy: {np.mean(acc_scores):.2f} (+/- {np.std(acc_scores):.2f})")
    print(f"AUC: {np.mean(auc_scores):.2f} (+/- {np.std(auc_scores):.2f})")
    print(f"F1 Score: {np.mean(f1_scores):.2f} (+/- {np.std(f1_scores):.2f})")

    for model_history in model_info['histories']:
        plot_training_history(model_history, title=f'Training History: {model_name}')
