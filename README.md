# Neural-Networks-ML
# Titanic Survival Prediction — Neural Network Analysis

A neural network study on the classic Titanic dataset that goes beyond a single accuracy number. The project trains a shallow and a deep multilayer perceptron to predict passenger survival, then compares them through error analysis, ROC and precision-recall curves, confusion matrices, probability calibration, and a systematic hyperparameter sweep. The aim is to understand not only how well the models perform, but where they fail, how stable they are under different configurations, and whether extra depth actually pays off on a small tabular problem.

## Dataset and Preprocessing

The input is `Titanic-Dataset.csv`, containing 891 passengers with features covering class, sex, age, fare, family relations, and port of embarkation, along with the binary `Survived` target. Irrelevant or high-cardinality columns (`PassengerId`, `Name`, `Ticket`, `Cabin`) are dropped, missing values in `Age`, `Fare`, and `Embarked` are filled with the median or mode, `Sex` is binary-encoded, and `Embarked` is one-hot encoded. Three engineered features are added to capture household context: `FamilySize`, `IsAlone`, and `FarePerPerson`. The data is split 80/20 with stratification on the target, and all features are standardized with `StandardScaler` fit only on the training set.

## Models

Two networks are trained with scikit-learn's `MLPClassifier`. The basic network has a single hidden layer of 64 units with light L2 regularization, early stopping, and the Adam optimizer — a deliberately minimal baseline. The deep network stacks five hidden layers (256, 128, 64, 32, 16) with stronger regularization and an adaptive learning rate schedule. Both use ReLU activations. Each model is evaluated first in its baseline configuration, then again in a tuned configuration whose hyperparameters come from the cross-validated sweeps described below.

## Hyperparameter Tuning

Four hyperparameters are swept independently using 5-fold stratified cross-validation scored on ROC-AUC: L2 regularization strength (`alpha`), initial learning rate, batch size, and maximum iterations. The epoch sweep is run with early stopping disabled so the raw effect of training length is visible. The best `alpha` and learning rate from the sweeps are then combined to train tuned versions of both models, which are compared directly against the baselines.

## Evaluation

Each model variant is assessed with accuracy, ROC-AUC, average precision, log loss, Brier score, and training accuracy (for overfitting detection). The analysis generates two figures. The first covers error analysis and AUC: confusion matrices for all four variants, overlaid ROC and precision-recall curves, predicted probability distributions split by true class, error rates broken down by passenger class and sex, age distributions per error type, and a full metrics summary table. The second figure shows the four hyperparameter sweep curves side by side with the best value marked for each model.

## How to Run

The project needs only `numpy`, `pandas`, `matplotlib`, and `scikit-learn` — no TensorFlow or PyTorch required, so it runs cleanly in lightweight environments including JupyterLite and Pyodide. Place `Titanic-Dataset.csv` in the same directory as the notebook and run the cells in order. The sweeps take a few minutes on a typical laptop. Two figures are saved to disk: `titanic_error_auc.png` and `titanic_hyperparams.png`.

## Repository Contents

The repository contains the analysis notebook, the Titanic dataset CSV, this README, and the two generated figures produced after running the full pipeline.
