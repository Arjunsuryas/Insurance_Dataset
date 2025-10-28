import matplotlib.pyplot as plt
import pandas as pd

# Fit preprocessor and transform data
X_processed = preprocessor.fit_transform(X)

# Get the feature names correctly
def get_feature_names(preprocessor):
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == 'passthrough':
            feature_names.extend(cols)
        elif hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    return [str(f) for f in feature_names]

feature_names = get_feature_names(preprocessor)

# Ensure lengths match
print("Number of feature names:", len(feature_names))
print("Number of model features:", len(xgb_pipe.named_steps['model'].feature_importances_))

# Fit XGBoost
xgb_model = xgb_pipe.named_steps['model']
xgb_model.fit(X_processed, y)

# If lengths mismatch, only take the first n features
min_len = min(len(feature_names), len(xgb_model.feature_importances_))
feature_names = feature_names[:min_len]

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(feature_names, xgb_model.feature_importances_[:min_len])
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
