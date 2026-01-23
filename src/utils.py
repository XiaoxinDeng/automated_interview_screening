import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def _safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else np.nan

def _group_confusion_counts(y_true, y_pred):
    """
    Returns (TP, FP, TN, FN) for a binary classification task.
    """
    # confusion_matrix returns [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(tn), int(fn)

def _rates_from_counts(tp, fp, tn, fn):
    """
    Lecture 2 error-rate definitions (as probabilities):
      TPR = P(Ŷ=1 | Y=1) = TP/(TP+FN)
      FPR = P(Ŷ=1 | Y=0) = FP/(FP+TN)
      TNR = P(Ŷ=0 | Y=0) = TN/(FP+TN)
      FNR = P(Ŷ=0 | Y=1) = FN/(TP+FN)
    """
    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    tnr = _safe_div(tn, fp + tn)
    fnr = _safe_div(fn, tp + fn)
    return tpr, fpr, tnr, fnr

def _selection_rate(y_pred):
    """Selection rate = P(Ŷ=1). For DP compute per group: P(Ŷ=1 | A=a)."""
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_pred == 1))

def preprocess_data(train_df, val_df, test_df, classifier, sensitive_col='is_white', target='prior_hiring_decision'):
    # Create Binary 'is_white' column for Fairness Evaluation (1 = White, 0 = Non-White)
    train_df['is_white'] = (train_df['Race'] == 1).astype(int)
    val_df['is_white'] = (val_df['Race'] == 1).astype(int)
    test_df['is_white'] = (test_df['Race'] == 1).astype(int)


    # Features to DROP: Target + Sensitive Attributes + Proxies
    # Note: 'Race' is the raw sensitive column, 'is_white' is our derived one. We drop both from training features.
    drop_cols = [target, 'Race', 'is_white']

    # Define Predictors list by excluding the drop_cols
    features = [c for c in train_df.columns if c not in drop_cols]
    print("Using Features:", features)

    # Prepare X (features) and y (target)
    data={}
    data['X_train'] = train_df[features]
    data['y_train'] = train_df[target]

    data['X_val'] = val_df[features]
    data['y_val'] = val_df[target]
    # CRITICAL: We Keep 'is_white' (sensitive_attribute) separate for the Fairness Evaluation later.
    data['sensitive_val'] = val_df[sensitive_col]

    # Identify Numerical vs Categorical columns for different preprocessing
    # Explicitly define categorical columns since they are integer-encoded in ACS data
    numerical_cols = ['Age', 'Hours_Per_Week', 'interview_score', 'cv_assessment_score']
    categorical_cols = [c for c in features if c not in numerical_cols]

    print("Numerical:", numerical_cols)
    print("Categorical:", categorical_cols)

    # Create the ColumnTransformer
    # - Numerical: Standard Scaler (mean=0, var=1)
    # - Categorical: One-Hot Encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    # Full Pipeline 
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return data, clf

def compute_reweighing_weights(A, y):
    # A: sensitive attribute array
    # y: labels array

    weights = np.zeros(len(y))

    for a in np.unique(A):
        for label in np.unique(y):
            P_A = np.mean(A == a)
            P_Y = np.mean(y == label)
            P_A_Y = np.mean((A == a) & (y == label))

            w = (P_A * P_Y) / P_A_Y

            weights[(A == a) & (y == label)] = w

    return weights

def fairness_metrics(y_true, y_pred, sensitive_attr, *, min_group_size=1):
    """
    Compute DP, EO, and EOD for binary sensitive attribute ∈ {0,1}.

    Returns:
      results: dict with per-group metrics
      gaps:    dict with parity gaps (max-min); for binary A, this is abs(diff)
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    A = np.asarray(sensitive_attr).reshape(-1).astype(int)

    if not (y_true.shape == y_pred.shape == A.shape):
        raise ValueError("y_true, y_pred, and sensitive_attr must have the same shape.")
    if np.any((A != 0) & (A != 1)):
        raise ValueError("sensitive_attr must be binary (0/1).")
    if np.any((y_true != 0) & (y_true != 1)):
        raise ValueError("y_true must be binary (0/1).")
    if np.any((y_pred != 0) & (y_pred != 1)):
        raise ValueError("y_pred must be binary (0/1).")

    results = {}
    for a in [0, 1]:
        mask = (A == a)
        n = int(np.sum(mask))
        if n < min_group_size:
            results[a] = {
                "n": n,
                "selection_rate_P(yhat=1|A)": np.nan,
                "TPR_P(yhat=1|Y=1,A)": np.nan,
                "FPR_P(yhat=1|Y=0,A)": np.nan,
                "TNR": np.nan,
                "FNR": np.nan,
                "TP": 0, "FP": 0, "TN": 0, "FN": 0
            }
            continue

        yt, yp = y_true[mask], y_pred[mask]
        tp, fp, tn, fn = _group_confusion_counts(yt, yp)
        tpr, fpr, tnr, fnr = _rates_from_counts(tp, fp, tn, fn)

        results[a] = {
            "n": n,
            # Demographic Parity quantity (slide ~30)
            "selection_rate_P(yhat=1|A)": _selection_rate(yp),
            # Equalized Opportunity quantity (slide ~34)
            "TPR_P(yhat=1|Y=1,A)": tpr,
            # Equalized Odds additionally needs FPR parity (slide ~36)
            "FPR_P(yhat=1|Y=0,A)": fpr,
            "TNR": tnr,
            "FNR": fnr,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }
    # Parity gaps (for binary A: abs difference between group 1 and 0)
    sr0, sr1 = results[0]["selection_rate_P(yhat=1|A)"], results[1]["selection_rate_P(yhat=1|A)"]
    tpr0, tpr1 = results[0]["TPR_P(yhat=1|Y=1,A)"], results[1]["TPR_P(yhat=1|Y=1,A)"]
    fpr0, fpr1 = results[0]["FPR_P(yhat=1|Y=0,A)"], results[1]["FPR_P(yhat=1|Y=0,A)"]

    dp_gap = float(np.abs(sr1 - sr0)) if not (np.isnan(sr0) or np.isnan(sr1)) else np.nan
    eo_gap = float(np.abs(tpr1 - tpr0)) if not (np.isnan(tpr0) or np.isnan(tpr1)) else np.nan
    fpr_gap = float(np.abs(fpr1 - fpr0)) if not (np.isnan(fpr0) or np.isnan(fpr1)) else np.nan
    eod_gap = float(np.nanmax([eo_gap, fpr_gap]))  # common reporting: max(TPR gap, FPR gap)

    gaps = {
        "DP_gap_|P(yhat=1|A=1)-P(yhat=1|A=0)|": dp_gap,
        "EO_gap_|TPR(A=1)-TPR(A=0)|": eo_gap,
        "FPR_gap_|FPR(A=1)-FPR(A=0)|": fpr_gap,
        "EOD_gap_max(TPR_gap,FPR_gap)": eod_gap
    }
    return results, gaps


def plot_roc_curve(y_val, y_score, sensitive_attr=None, title="ROC Curve"):
    # fpr, tpr, _ = roc_curve(y_val, y_pred)
    # roc_auc = auc(fpr, tpr)
 
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(title)
    # plt.legend(loc="lower right")
    # plt.show()
    y_val = np.asarray(y_val).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if sensitive_attr is not None:
        sensitive_attr = np.asarray(sensitive_attr).reshape(-1)

    
    if sensitive_attr is not None:
        if not (y_val.shape == y_score.shape == sensitive_attr.shape):
            raise ValueError("y_val, y_score, and sensitive_attr must have the same shape.")
        if np.any((sensitive_attr != 0) & (sensitive_attr != 1)):
            raise ValueError("sensitive attribute must be binary (0/1).")
    else:
        if not (y_val.shape == y_score.shape):
            raise ValueError("y_val, y_score must have the same shape.")
    
    
    plt.figure(figsize=(8, 6))

    # Plot overall ROC
    fpr_all, tpr_all, _ = roc_curve(y_val, y_score)
    auc_all = auc(fpr_all, tpr_all)
    plt.plot(
        fpr_all,
        tpr_all,
        lw=3,
        label=f"Overall (AUC = {auc_all:.2f})"
    )
    # Plot each group
    if sensitive_attr is not None:
        for g, label in [(0, "A=0"), (1, "A=1")]:
            mask = (sensitive_attr == g)
            if np.sum(mask) < 2:
                # Not enough points to form an ROC curve
                continue

            fpr, tpr, _ = roc_curve(y_val[mask], y_score[mask])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")

    # Reference diagonal
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# def plot_groupwise_roc(y_true, y_score, is_white):
#     """
#     Plot ROC curves separately for A=0 and A=1 (is_white).
#     """
#     y_true = np.asarray(y_true).reshape(-1)
#     y_score = np.asarray(y_score).reshape(-1)
#     A = np.asarray(is_white).reshape(-1)

#     plt.figure(figsize=(6, 5))

#     for a, label in [(0, "Non-white (A=0)"), (1, "White (A=1)")]:
#         mask = (A == a)
#         if np.sum(mask) == 0:
#             continue

#         fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
#         roc_auc = auc(fpr, tpr)

#         plt.plot(fpr, tpr, lw=2, label=f"{label} | AUC={roc_auc:.3f}")
    
#     fpr, tpr, _ = roc_curve(y_true, y_score)
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=2, label=f"Overall | AUC={roc_auc:.3f}")

#     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Group-wise ROC Curves by Sensitive Attribute")
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

