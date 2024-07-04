from sklearn.pipeline import Pipeline
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline = Pipeline(
    [
        ('Mean Imputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('Mode Imputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('Domain Processing', pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY,
                                                  variable_to_add=config.FEATURE_TO_ADD))
        ('Drop Features', pp.DropColumns(variables_to_drop=config.DROP_FEATURES))
        ('Label Encoding', pp.CustomLabelEncoder(variables=config.FEATURES_ENCODE))
        ('Log Transformation', pp.LogTransforms(variables=config.LOG_FEATURES))
        ('MinMaxSxale', MinMaxScaler()),
        ('Logistic Regression Classifier', LogisticRegression(random_state=42))
    ]
)