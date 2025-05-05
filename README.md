# AllWISE-W3-W4-verifier
This repository contains a machine learning-based framework to assess the reliability of WISE W3 and W4 detections for young stellar objects (YSOs) using AllWISE catalogue parameters. We trained and evaluated multiple classifiers—including Random Forest, SVM, Logistic Regression, Naive Bayes, KNN, and Decision Tree—on labeled cutout data from the Orion region. The final model selection is based on cross-validated accuracy, with Random Forest consistently performing best.

For detailed methodology, data preparation, and scientific context, please refer to our paper: https://arxiv.org/pdf/2501.08486

# Usage

# Training
Train models separately for W3 and W4 bands:

# Train W3 model
python W3_main.py train

# Train W4 model
python W4_main.py train

This will automatically: Load training data
Evaluate multiple classifiers (Random Forest, SVM, k-NN, Naive Bayes, Decision Tree)
Save the best-performing model based on validation accuracy
_________________________________________________________________________________________
# Prediction
After training, run predictions on new unlabelled data:

# Predict W3 band sources
python W3_main.py predict

# Predict W4 band sources
python W4_main.py predict
_________________________________________________________________________________________

The output CSV will include: Source ID
Predicted class (real or fake)
Probability of the source being real
