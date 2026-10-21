"""Đặc trưng kế thừa checkpoint giữa kỳ, tách thành module để lưu/nạp mô hình."""
import hashlib, html, json, re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
TARGET='fraudulent'
ID_COLUMN='job_id'
TEXT_COLUMNS=['title','company_profile','description','requirements','benefits']
BINARY_COLUMNS=['telecommuting','has_company_logo','has_questions']
CATEGORICAL_COLUMNS=['department','employment_type','required_experience','required_education','industry','function','country']
COUNT_COLUMNS=[f'{c}_word_count' for c in TEXT_COLUMNS]
NUMERIC_COLUMNS=[f'log1p_{c}' for c in COUNT_COLUMNS]
MISSING_FLAGS=[f'{c}_missing' for c in TEXT_COLUMNS+['salary_range']]
FLAG_COLUMNS=BINARY_COLUMNS+MISSING_FLAGS
def normalize_and_validate(data):
    """Sao chép dữ liệu, chuẩn hóa khoảng trắng và kiểm tra miền giá trị."""
    result = data.copy(deep=True)
    for col in result.select_dtypes(include=["object", "string"]).columns:
        result[col] = result[col].str.strip().replace(r"^\s*$", np.nan, regex=True)
    labels = pd.to_numeric(result[TARGET], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Nhãn fraudulent phải có đủ và chỉ nhận 0/1; cần rà soát nguồn.")
    result[TARGET] = labels.astype("int8")
    if result[ID_COLUMN].isna().any() or result[ID_COLUMN].duplicated().any():
        raise ValueError("job_id thiếu hoặc trùng: cần xác minh trước khi chia dữ liệu.")
    for col in BINARY_COLUMNS:
        values = pd.to_numeric(result[col], errors="coerce")
        invalid = result[col].notna() & (values.isna() | ~values.isin([0, 1]))
        if invalid.any():
            raise ValueError(f"{col} có {invalid.sum()} giá trị khác 0/1.")
        result[col] = values.astype(float)
    return result

def clean_text(value):
    """Làm sạch xác định theo từng ô: không học từ các dòng khác hoặc từ nhãn.

    Giữ từ phủ định và dấu câu; TF-IDF sẽ tách token. Không cần tải NLTK/spaCy.
    URL/email/số điện thoại không được xác minh; đây chỉ là biểu diễn văn bản.
    """
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def content_groups(frame):
    """Khóa nhóm chỉ từ X, tuyệt đối không dùng fraudulent hoặc ID làm tín hiệu."""
    cols = ["title", "description", "requirements"]
    normalized = frame[cols].apply(lambda col: col.map(clean_text))
    # JSON giữ ranh giới giữa các trường, SHA-256 tạo khóa gọn và ổn định.
    return normalized.apply(
        lambda row: hashlib.sha256(
            json.dumps(row.tolist(), ensure_ascii=False).encode("utf-8")
        ).hexdigest(), axis=1,
    ).rename("content_group")

def audit_duplicates(frame):
    """Tách trường hợp nhãn mâu thuẫn trước khi loại bản sao hoàn toàn."""
    predictors = [c for c in frame.columns if c not in [ID_COLUMN, TARGET]]
    # groupby với dropna=False coi các giá trị thiếu cùng vị trí là giống nhau.
    label_counts = frame.groupby(predictors, dropna=False)[TARGET].transform("nunique")
    conflict = label_counts.gt(1)
    quarantined = frame.loc[conflict, [ID_COLUMN, TARGET]].copy()
    consistent = frame.loc[~conflict].copy()
    duplicate = consistent.duplicated(subset=predictors, keep="first")
    removed = consistent.loc[duplicate, [ID_COLUMN, TARGET]].copy()
    cleaned = consistent.loc[~duplicate].copy().reset_index(drop=True)
    audit = pd.Series({
        "rows_before": len(frame), "conflicting_rows_quarantined": int(conflict.sum()),
        "exact_duplicate_rows_removed": int(duplicate.sum()), "rows_after": len(cleaned),
    }, name="value")
    return cleaned, audit, quarantined, removed

def make_features(frame):
    """Feature engineering theo từng dòng; cùng quy tắc cho train và tin mới."""
    result = pd.DataFrame(index=frame.index)
    cleaned_text = pd.DataFrame(index=frame.index)
    for col in TEXT_COLUMNS:
        series = frame[col].replace(r"^\s*$", np.nan, regex=True)
        cleaned = series.map(clean_text)
        cleaned_text[col] = cleaned
        result[f"{col}_missing"] = series.isna().astype(float)
        counts = cleaned.str.split().str.len().astype(float).mask(series.isna())
        result[f"{col}_word_count"] = counts
        result[f"log1p_{col}_word_count"] = np.log1p(counts)
    result["text_combined"] = cleaned_text.apply(
        lambda row: " ".join(part for part in row if part).strip(), axis=1
    )
    for col in BINARY_COLUMNS:
        values = pd.to_numeric(frame[col], errors="coerce")
        result[col] = values.where(values.isin([0, 1]), np.nan)
    result["salary_range_missing"] = (
        frame["salary_range"].replace(r"^\s*$", np.nan, regex=True).isna().astype(float)
    )
    for col in CATEGORICAL_COLUMNS:
        source = frame["location"].str.split(",").str[0] if col == "country" else frame[col]
        result[col] = source.str.strip().replace(r"^\s*$", np.nan, regex=True)
    return result

def make_numeric_pipeline():
    """Median bền vững hơn mean trước đuôi phân phối dài; fit chỉ trên train."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])

def make_categorical_pipeline():
    """Mode học từ train; OHE học danh mục/tần suất từ train, không từ test."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore", min_frequency=10, max_categories=60,
            sparse_output=True, dtype=np.float64,
        )),
    ])

def build_preprocessor(variant='combined'):
    """Fit riêng trong từng fold. Ablation chỉ thay nhánh chỉ định, giữ cấu hình còn lại."""
    if variant not in {'combined','text_only','metadata_only','no_logo','no_log'}:
        raise ValueError('Unknown feature variant')
    branches=[]
    if variant != 'text_only':
        flags=[c for c in FLAG_COLUMNS if not (variant=='no_logo' and c=='has_company_logo')]
        numeric=COUNT_COLUMNS if variant=='no_log' else NUMERIC_COLUMNS
        branches.extend([
            ('numeric',make_numeric_pipeline(),numeric),
            ('categorical',make_categorical_pipeline(),CATEGORICAL_COLUMNS),
            ('flags',SimpleImputer(strategy='most_frequent',keep_empty_features=True),flags)])
    if variant != 'metadata_only':
        branches.append(('text',TfidfVectorizer(max_features=15000,ngram_range=(1,2),min_df=3,
            sublinear_tf=True,strip_accents='unicode',dtype=np.float64),'text_combined'))
    return Pipeline([('features',FunctionTransformer(make_features,validate=False)),
        ('columns',ColumnTransformer(branches,remainder='drop',sparse_threshold=1.0))])
