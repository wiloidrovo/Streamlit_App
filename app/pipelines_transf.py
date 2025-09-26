import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('rbst_scaler', RobustScaler()),
])

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Convierte columnas de tipo fecha en tres nuevas: *_year (num), *_month_name (cat), *_day_name (cat).
    - Solo considera columnas string/objeto o datetime (NO numéricas).
    - Requiere que >= min_valid_ratio de los valores se conviertan correctamente a fecha.
    - Ignora columnas categóricas como 'arrival_date_month' (nombres de meses sin día/año).
    """
    def __init__(self, date_cols=None, min_valid_ratio=0.8):
        self.date_cols = date_cols
        self.min_valid_ratio = min_valid_ratio

    def _is_date_candidate(self, s: pd.Series) -> bool:
        # Aceptar solo object/string o datetime; NO numéricas
        if pd.api.types.is_numeric_dtype(s):
            return False
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s)):
            return False
        # Intentar parsear
        fechas = pd.to_datetime(s, errors="coerce")
        valid_ratio = fechas.notna().mean()
        return valid_ratio >= self.min_valid_ratio

    def fit(self, X, y=None):
        Xc = X.copy()
        if self.date_cols is None:
            # Candidatas: que contengan 'date' en el nombre y sean parseables
            candidatas = [c for c in Xc.columns if "date" in c.lower()]
            self.date_cols = [c for c in candidatas if self._is_date_candidate(Xc[c])]
        else:
            # Si nos pasan lista, filtrar por parseables
            self.date_cols = [c for c in self.date_cols if c in Xc.columns and self._is_date_candidate(Xc[c])]
        return self

    def transform(self, X, y=None):
        Xc = X.copy()
        for col in self.date_cols:
            fechas = pd.to_datetime(Xc[col], errors="coerce")
            Xc[f"{col}_year"] = fechas.dt.year
            Xc[f"{col}_month_name"] = fechas.dt.month_name()
            Xc[f"{col}_day_name"] = fechas.dt.day_name()
            Xc.drop(columns=[col], inplace=True)  # eliminar original
        return Xc


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        try:
            self._oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            self._oh = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self._columns = None

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object']).copy()
        # Si no hay categóricas, mantener lista vacía
        if X_cat.shape[1] == 0:
            self._columns = []
            self._oh.fit(pd.DataFrame(index=X.index))
            return self
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self

    def transform(self, X, y=None):
        X_cat = X.select_dtypes(include=['object']).copy()
        if X_cat.shape[1] == 0:
            return pd.DataFrame(index=X.index)  # no hay categóricas
        X_cat_oh = self._oh.transform(X_cat)
        return pd.DataFrame(X_cat_oh, columns=self._columns, index=X.index)


class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._full_pipeline = None
        self._columns = None
        self.input_features_ = None
        self._date_extractor = None  # persistimos el extractor

    def fit(self, X, y=None):
        X0 = X.copy()
        self.input_features_ = list(X0.columns)

        self._date_extractor = DateFeatureExtractor()
        X1 = self._date_extractor.fit_transform(X0)

        num_attribs = list(X1.select_dtypes(exclude=['object']).columns)
        cat_attribs = list(X1.select_dtypes(include=['object']).columns)

        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", CustomOneHotEncoder(), cat_attribs),
        ])
        self._full_pipeline.fit(X1)

        out_cols = []
        out_cols.extend(num_attribs)
        cat_encoder = self._full_pipeline.named_transformers_["cat"]
        if hasattr(cat_encoder, "_columns") and len(cat_encoder._columns) > 0:
            out_cols.extend(list(cat_encoder._columns))
        self._columns = out_cols
        return self

    def transform(self, X, y=None):
        X0 = X.copy()
        X1 = self._date_extractor.transform(X0)
        X_prep = self._full_pipeline.transform(X1)
        return pd.DataFrame(X_prep, columns=self._columns, index=X.index)