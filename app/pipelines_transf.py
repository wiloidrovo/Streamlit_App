import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rbst_scaler", RobustScaler()),
])

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        try:
            self._oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            self._oh = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self._columns = []

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=["object"]).copy()
        if X_cat.shape[1] == 0:
            self._columns = []
            self._oh.fit(pd.DataFrame(index=X.index))
            return self
        self._oh.fit(X_cat)
        self._columns = self._oh.get_feature_names_out(X_cat.columns).tolist()
        return self

    def transform(self, X, y=None):
        X_cat = X.select_dtypes(include=["object"]).copy()
        if X_cat.shape[1] == 0:
            return pd.DataFrame(index=X.index)
        X_cat_oh = self._oh.transform(X_cat)
        return pd.DataFrame(X_cat_oh, columns=self._columns, index=X.index)

class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._full_pipeline = None
        self._columns = None
        self.input_features_ = None

    def fit(self, X, y=None):
        X0 = X.copy()
        self.input_features_ = list(X0.columns)

        num_attribs = list(X0.select_dtypes(exclude=["object"]).columns)
        cat_attribs = list(X0.select_dtypes(include=["object"]).columns)

        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", CustomOneHotEncoder(), cat_attribs),
        ])

        Xt = self._full_pipeline.fit_transform(X0)
        out_cols = []
        out_cols.extend(num_attribs)
        cat_encoder = self._full_pipeline.named_transformers_["cat"]
        if hasattr(cat_encoder, "_columns") and cat_encoder._columns:
            out_cols.extend(list(cat_encoder._columns))
        self._columns = out_cols
        return self

    def transform(self, X, y=None):
        X0 = X.copy()
        return self._full_pipeline.transform(X0)


class ColumnFilter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = list(columns) if columns is not None else None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.columns is None:
            return X

        if isinstance(X, pd.DataFrame):
            keep = [c for c in self.columns if c in X.columns]
            return X[keep]

        elif isinstance(X, np.ndarray):
            if len(self.columns) > X.shape[1]:
                raise ValueError(
                    f"ColumnFilter: expected {len(self.columns)} columns, got {X.shape[1]}."
                )
            return X[:, :len(self.columns)]

        return X