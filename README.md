# Muffin vs Chihuahua Image Classification Experiment

This experiment aims to classify images into two categories: muffins and chihuahuas. It employs the Keras framework alongside Keras Tuner for hyperparameter optimization.

## Key Features:

- Utilizes Keras and Keras Tuner for neural network construction and hyperparameter optimization.
- Implements data augmentation using ImageDataGenerator.
- Uses Stratified K-Fold cross-validation for model evaluation.
- Provides three distinct CNN architectures for experimentation: 
  - A simple sequential model
  - A simple CNN model
  - A more intricate CNN model.
- Evaluates models based on Accuracy, AUC, and F1 Score and showcases the training history.

## Requirements:

- Keras
- Keras Tuner
- Scikit-learn
- Numpy
- Matplotlib

## Dataset:

You can obtain the dataset used for this experiment from [here](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification). 

After downloading, ensure that the dataset is extracted and placed in a folder named `archive` in the root directory of the project, or modify the `load_data` function to point to the correct directory.

## Running the Experiment:

1. **Setup the Environment:**

    First, ensure that you have all the necessary libraries installed:
    ```bash
    pip install keras keras-tuner scikit-learn numpy matplotlib
    ```

2. **Clone and Navigate:**

    Clone this repository and navigate into the project directory.
    ```bash
    git clone https://github.com/Kalkan-Emre/Muffins-vs.-Chihuahuas
    cd Muffins-vs.-Chihuahuas
    ```

3. **Run the Code:**

    Execute the provided Python script to initiate the experiment.
    ```bash
    python experiment.py
    ```

4. **Review Results:**

    After the experiment completes, review the printed scores for each model architecture. The training history plots will also be displayed.

## Experiment Results:

Each of the three models' results, including Accuracy, AUC, and F1 Score, will be printed to the console. Additionally, training history plots will visualize how each model performed over the epochs.

## Troubleshooting:

If you encounter any issues related to the dataset path, make sure you've properly downloaded and extracted the dataset, and the path in the `load_data` function correctly points to the dataset directory.


## Acknowledgments:

Thanks to Kaggle and the dataset creator for making the [Muffin vs Chihuahua dataset](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification) available for public use.
