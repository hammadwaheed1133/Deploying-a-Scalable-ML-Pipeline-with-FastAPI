# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details


This machine learning model is a binary classifier built using the Logistic Regression algorithm from scikit-learn. It predicts whether an individual's annual income exceeds \$50K based on features collected from the U.S. Census Adult dataset. The project follows a full MLOps pipeline, including training, evaluation, testing, deployment with FastAPI, and CI/CD integration using GitHub Actions.

## Intended Purpose

This model is developed as part of an educational project focused on machine learning engineering and deployment practices. Its purpose is to showcase end-to-end pipeline development. it is not intended for use in real-world production environments or decision-making involving sensitive information.

## Training Data Description

The dataset used for model training is sourced from the U.S. Census Bureau’s Adult dataset. It includes 14 key attributes such as:

- Demographics (e.g., age, sex, race)
- Education and work history
- Capital gain/loss
- Hours worked per week
- Country of origin

Categorical features were transformed using OneHotEncoding, and the target variable (income) was binarized using LabelBinarizer.

## Evaluation Method

The model was tested on a hold-out test set (20% of the original data) to evaluate general performance. Additional testing was performed on specific slices of the dataset based on features like **education**, **race**, and **sex** to assess fairness.

## Performance Metrics

The following metrics were used to evaluate the model:

Precision
Recall
F1 Score

Example scores from the evaluation:
Metric     Value 
Precision   0.72   
Recall      0.75   
F1 Score    0.73   

Slice-based performance metrics were also calculated to monitor fairness across different groups.

## Ethical Implications

Fairness Concernsm: Since this model is trained on real-world data, it may reflect social and economic biases present in the original dataset.
Responsible Use : This model is intended for learning only. Using it in production without auditing for bias and fairness may result in harmful outcomes.
Transparency : All code, logic, and artifacts related to this model are open for review to promote responsible machine learning practices.

## Known Limitations

Not suited for deployment in production environments without extensive validation and retraining. Bias mitigation strategies (e.g., reweighting or post-processing) were not applied and may be needed for sensitive applications. Performance may vary depending on data distribution changes over time.