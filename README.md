# Rock vs Mine Classification Project

This machine learning project uses a Logistic Regression model to classify sonar signals as either a **Rock (R)** or a **Mine (M)**.

## Project Overview
The objective of this project is to analyze sonar signal responses bounced off a metallic cylinder (mine) versus a roughly cylindrical rock, determining the classification model's effectiveness using Python and scikit-learn.

## Dataset & Features
- **Dataset Source:** Sonar data containing frequency and energy measurements.
- **Dataset Dimensions:** 
  - Total Samples: 208 rows
  - Total Features: 61 columns (60 numerical feature columns representing energy across different frequency bands, and 1 target label column)
- **Features:** 
  - Columns `0` through `59`: Numerical float values representing the energy of the sonar signals within specific frequency bands.
  - Column `60`: The target label (`R` for Rock, `M` for Mine).
- **Class Distribution:** 
  - Mine (M): 111 samples
  - Rock (R): 97 samples

## Train and Test Sets Dimensions
- **Training Set (x_train, y_train):** 187 samples, 60 features.
- **Testing Set (x_test, y_test):** 21 samples, 60 features.
- *Note:* The dataset was split using a 90% train and 10% test ratio, maintaining class distribution balance using stratification (`stratify=Y`).

## Libraries Used
- `numpy`: For numerical array manipulation and matrix operations.
- `pandas`: For data loading, manipulation, and DataFrame management.
- `scikit-learn`: 
  - `train_test_split`: For splitting the dataset into training and testing sets.
  - `LogisticRegression`: For building the classification model.
  - `accuracy_score`: For evaluating model prediction performance.

## Step-by-Step Workflow
1. **Import Libraries:** Import essential Python libraries (`numpy`, `pandas`, `train_test_split`, `LogisticRegression`).
2. **Load Data:** Read the sonar CSV dataset into a pandas DataFrame without a header (`header=None`).
3. **Exploratory Data Analysis (EDA):** 
   - Check the shape and summary statistics of the dataset.
   - Count the distribution of target classes (`M` vs `R`).
4. **Data Preprocessing:** Separate the features (`X` from columns 0 to 59) from the target label (`Y` at column 60).
5. **Data Splitting:** Divide the data into training and testing subsets using `train_test_split` with a test size of 10% and stratification.
6. **Model Initialization & Training:** Instantiate the `LogisticRegression` model and train (fit) it using the training data (`x_train`, `y_train`).
7. **Model Evaluation:** 
   - Predict outcomes for both training and testing datasets.
   - Calculate accuracy scores to check for model performance and potential overfitting.
8. **Prediction on New Data:** Test the trained model using custom input data to verify its capability to classify unseen sonar readings.
