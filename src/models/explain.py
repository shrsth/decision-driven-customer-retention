"""Per-customer churn explanations from the model itself, via SHAP.

Replaces hand-written if/else reasons with genuine attributions: SHAP tells us,
for each customer, how much each feature pushed their churn risk up or down.
For a logistic-regression pipeline these are exact and instant
(`shap.LinearExplainer` on the preprocessed features). We surface the top few
features that *raise* an ACT customer's risk, phrased in plain language.
"""

import numpy as np

_NUM_LABEL = {
    "tenure": "tenure",
    "MonthlyCharges": "monthly charges",
    "TotalCharges": "total charges",
}
# Features that are not actionable retention drivers — excluded from reasons.
_SKIP_COLS = {"gender"}

_FLAG_LABEL = {
    "PaperlessBilling": "paperless billing",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone service",
    "OnlineSecurity": "online security",
    "OnlineBackup": "online backup",
    "DeviceProtection": "device protection",
    "TechSupport": "tech support",
    "StreamingTV": "streaming TV",
    "StreamingMovies": "streaming movies",
    "MultipleLines": "multiple lines",
}


def _token(name, raw_value, median):
    """Turn a (possibly one-hot) feature name into a readable driver phrase."""
    if name in _NUM_LABEL:
        level = "high" if (median is not None and raw_value >= median) else "low"
        return f"{level} {_NUM_LABEL[name]}"

    col, _, val = name.partition("_")
    if col == "InternetService":
        return "no internet" if val == "No" else f"{val.lower()} internet"
    if col == "Contract":
        return f"{val.lower()} contract"
    if col == "PaymentMethod":
        return val.lower()
    if col == "SeniorCitizen":
        return "senior citizen"
    if col in _FLAG_LABEL:
        return ("no " if val == "No" else "") + _FLAG_LABEL[col]
    return val or name


def shap_reasons(pipeline, X, top_n=3):
    """Top churn-raising features per row of X, as readable phrases.

    Returns a list of strings aligned to X's rows.
    """
    import shap

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    names = [
        n.replace("num__", "").replace("cat__", "")
        for n in pre.get_feature_names_out()
    ]
    explainer = shap.LinearExplainer(model, Xt)
    sv = np.asarray(explainer.shap_values(Xt))

    medians = {c: float(X[c].median()) for c in _NUM_LABEL if c in X.columns}
    X_reset = X.reset_index(drop=True)

    reasons = []
    for i in range(sv.shape[0]):
        order = np.argsort(sv[i])[::-1]  # most churn-raising first
        toks, seen = [], set()
        for j in order:
            if sv[i, j] <= 0 or len(toks) >= top_n:
                break
            base = names[j]
            if base.partition("_")[0] in _SKIP_COLS:
                continue
            raw = X_reset[base].iloc[i] if base in X_reset.columns else None
            tok = _token(base, raw, medians.get(base))
            if tok not in seen:
                seen.add(tok)
                toks.append(tok)
        reasons.append(", ".join(toks) if toks else "low modeled risk")
    return reasons
