# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier created for predicting whether an individual's income exceeds $50K/year based on census data. The model was developed as part of a machine learning pipeline project.

- **Model Type:** Random Forest Classifier
- **Model Version:** 1.0
- **Training Algorithm:** scikit-learn RandomForestClassifier
- **Hyperparameters:** 
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42
- **Features:** The model uses both categorical and continuous features from census data, including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain/loss, hours per week, and native country.
- **Developer:** Student Project
- **Date:** March 2026

## Intended Use

This model is intended for educational purposes as part of a machine learning operations (MLOps) course project. The primary use case is to demonstrate:

- Building and training a classification model with mixed data types
- Implementing model evaluation on data slices
- Deploying a model through a RESTful API
- Following MLOps best practices including versioning, testing, and CI/CD

**Intended Users:** Students, educators, and practitioners learning about MLOps and machine learning deployment.

**Out-of-Scope Uses:** This model should not be used for real-world employment, lending, or other decisions that could impact individuals' lives, as it has not been thoroughly validated for fairness, bias, or accuracy in production scenarios.

## Training Data

The model was trained on the Census Income Dataset (also known as "Adult" dataset) from the UCI Machine Learning Repository. 

- **Dataset Size:** 32,561 instances
- **Train/Test Split:** 80% training (26,048 instances), 20% testing (6,513 instances)
- **Data Processing:**
  - Removed leading/trailing whitespaces from column names and values
  - Applied one-hot encoding to categorical features
  - Applied label binarization to the target variable (salary: <=50K or >50K)
  - No scaling was applied to continuous features

- **Categorical Features (8):** workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Continuous Features (6):** age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
- **Target Variable:** salary (binary: <=50K or >50K)

## Evaluation Data

The evaluation data consists of the 20% held-out test set from the original Census Income Dataset, containing 6,513 instances. The same preprocessing steps applied to training data were applied to the test set using the fitted encoders from training.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model performance is evaluated using three standard classification metrics:

- **Precision:** Measures the proportion of positive predictions that are actually correct
- **Recall:** Measures the proportion of actual positives that are correctly identified
- **F1 Score:** Harmonic mean of precision and recall, providing a balanced measure

**Overall Test Set Performance:**
- Precision: ~0.74
- Recall: ~0.63
- F1 Score: ~0.68

**Slice-Based Evaluation:**
The model's performance was evaluated across different slices of categorical features to identify potential disparities. Performance metrics vary across different demographic groups and feature values. Detailed slice-based metrics are available in `model/slice_output.txt`.

**Note:** Actual performance metrics will vary based on the random state and specific training run. The values above are approximate expected ranges.

## Ethical Considerations

**Bias and Fairness Concerns:**
- The model is trained on historical census data that may contain societal biases related to race, sex, age, and other protected attributes
- The model uses sensitive attributes (race, sex, native-country) as features, which could perpetuate historical discrimination patterns
- Performance disparities exist across different demographic groups, as evidenced by slice-based evaluation
- The prediction task itself (income classification) relates to economic inequality

**Privacy:**
- The dataset contains demographic information, though it is anonymized census data
- Care should be taken not to use model predictions in ways that could harm individuals

**Transparency:**
- Model predictions and their basis should be explainable to affected individuals
- Slice-based evaluation helps identify where the model performs differently across groups

**Recommendations:**
- Do not use this model for real-world decision-making without thorough bias auditing
- Consider fairness constraints and debiasing techniques if adapting for production use
- Monitor model performance across demographic groups continuously
- Ensure compliance with anti-discrimination laws if used in any real application

## Caveats and Recommendations

**Limitations:**
1. **Historical Bias:** The model learned from 1994 census data, which may not reflect current economic and social conditions
2. **Class Imbalance:** The dataset has an imbalance between the two income classes, which may affect model performance
3. **Feature Limitations:** Some potentially relevant features are not included in the dataset
4. **Generalization:** The model may not generalize well to populations different from the US census
5. **Temporal Validity:** Economic conditions change over time; the model may become less accurate with time

**Recommendations:**
1. **Regular Retraining:** Update the model with more recent data to maintain relevance
2. **Fairness Auditing:** Conduct thorough fairness analysis before any production use
3. **Threshold Tuning:** Adjust classification thresholds based on specific use case requirements
4. **Feature Engineering:** Consider additional feature engineering to improve performance
5. **Alternative Approaches:** Explore bias mitigation techniques such as reweighting, adversarial debiasing, or fairness constraints
6. **Human Oversight:** Maintain human oversight in any decision-making process involving this model
7. **Documentation:** Keep detailed records of model versions, performance metrics, and known issues
