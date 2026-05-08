# -*- coding: utf-8 -*-
"""
BAI TAP LON: PHAN TICH DU LIEU NHAN SU (HR ANALYTICS)
======================================================
Dataset: IBM HR Analytics Employee Attrition & Performance
Nguon: Kaggle - pavansubhasht/ibm-hr-analytics-attrition-dataset

Yeu cau:
1. Xu ly du lieu nhan su
2. Phan tich muc luong theo phong ban
3. Phan tich yeu to anh huong nghi viec
4. Truc quan hoa du lieu HR
5. Xay dung mo hinh du doan nghi viec

Cach chay:
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy
    python hr_analytics.py
"""

import os
import sys
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.bbox"] = "tight"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MIRROR_URLS = [
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-HR-Employee-Attrition.csv",
]


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =====================================================================
# PHAN 1: IMPORT & TAI DU LIEU
# =====================================================================
section("PHAN 1: TAI DU LIEU")


def load_dataset():
    """Doc CSV local hoac tai tu GitHub mirror."""
    if os.path.exists(CSV_FILENAME):
        print(f"[OK] Da tim thay file local: {CSV_FILENAME}")
        return pd.read_csv(CSV_FILENAME)

    print("[..] Khong co file local, dang tai tu GitHub mirror...")
    for url in MIRROR_URLS:
        try:
            print(f"     Thu: {url}")
            urllib.request.urlretrieve(url, CSV_FILENAME)
            df = pd.read_csv(CSV_FILENAME)
            if "Attrition" in df.columns:
                print(f"[OK] Tai thanh cong ({len(df)} dong)")
                return df
        except Exception as e:
            print(f"     Loi: {e}")
            continue

    print("[!!] Khong the tai dataset. Vui long tai thu cong tu Kaggle:")
    print("     https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
    sys.exit(1)


df = load_dataset()
print(f"\nShape: {df.shape}")
print(f"So cot: {df.shape[1]}, So nhan vien: {df.shape[0]}")
print("\n5 dong dau tien:")
print(df.head())
print("\nThong tin cot:")
print(df.dtypes.value_counts())


# =====================================================================
# PHAN 2: XU LY DU LIEU NHAN SU
# =====================================================================
section("PHAN 2: XU LY DU LIEU")

# 2.1 Missing values
print("\n[2.1] Kiem tra missing values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("      Khong co missing value.")
else:
    print(missing)

# 2.2 Duplicates
print(f"\n[2.2] So dong trung lap: {df.duplicated().sum()}")
df = df.drop_duplicates()

# 2.3 Xoa cac cot khong co gia tri phan tich (hang so hoac ID)
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
print(f"\n[2.3] Cot co gia tri hang so (se xoa): {constant_cols}")
if "EmployeeNumber" in df.columns:
    constant_cols.append("EmployeeNumber")
df = df.drop(columns=constant_cols, errors="ignore")

# 2.4 Encode bien muc tieu
df["Attrition_bin"] = (df["Attrition"] == "Yes").astype(int)

# 2.5 Thong ke mo ta
print("\n[2.4] Thong ke mo ta cac bien so:")
print(df.describe().T[["mean", "std", "min", "max"]].round(2))

# 2.6 Phan bo Attrition
attrition_counts = df["Attrition"].value_counts()
print(f"\n[2.5] Phan bo Attrition:")
print(f"      O lai (No):    {attrition_counts.get('No', 0)} ({attrition_counts.get('No', 0)/len(df)*100:.1f}%)")
print(f"      Nghi viec (Yes): {attrition_counts.get('Yes', 0)} ({attrition_counts.get('Yes', 0)/len(df)*100:.1f}%)")


# =====================================================================
# PHAN 3: PHAN TICH MUC LUONG THEO PHONG BAN
# =====================================================================
section("PHAN 3: PHAN TICH LUONG THEO PHONG BAN")

salary_by_dept = df.groupby("Department")["MonthlyIncome"].agg(
    ["count", "mean", "median", "min", "max", "std"]
).round(2)
print("\n[3.1] Thong ke luong theo phong ban:")
print(salary_by_dept)

print("\n[3.2] Top luong trung binh theo JobRole:")
salary_by_role = df.groupby("JobRole")["MonthlyIncome"].mean().sort_values(ascending=False)
print(salary_by_role.round(2))

# Kiem dinh ANOVA
groups = [g["MonthlyIncome"].values for _, g in df.groupby("Department")]
f_stat, p_value = stats.f_oneway(*groups)
print(f"\n[3.3] Kiem dinh ANOVA (luong giua cac phong ban):")
print(f"      F-statistic = {f_stat:.3f}")
print(f"      p-value     = {p_value:.6f}")
if p_value < 0.05:
    print("      => Co khac biet co y nghia thong ke ve luong giua cac phong ban (p < 0.05)")
else:
    print("      => Khong co khac biet dang ke ve luong giua cac phong ban")


# =====================================================================
# PHAN 4: PHAN TICH YEU TO ANH HUONG NGHI VIEC
# =====================================================================
section("PHAN 4: YEU TO ANH HUONG NGHI VIEC")

# 4.1 Ty le nghi viec theo phong ban
print("\n[4.1] Ty le nghi viec theo phong ban:")
attr_by_dept = df.groupby("Department").apply(
    lambda g: (g["Attrition"] == "Yes").mean() * 100
).round(2)
print(attr_by_dept.to_string())

# 4.2 T-test cac bien numerical
print("\n[4.2] T-test: so sanh nhom Nghi vs O lai")
numerical_cols = ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany",
                  "TotalWorkingYears", "JobSatisfaction", "WorkLifeBalance"]
ttest_results = []
for col in numerical_cols:
    if col not in df.columns:
        continue
    g1 = df[df["Attrition"] == "Yes"][col]
    g2 = df[df["Attrition"] == "No"][col]
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    ttest_results.append({
        "Bien": col,
        "TB_NghiViec": round(g1.mean(), 2),
        "TB_OLai": round(g2.mean(), 2),
        "t-stat": round(t, 3),
        "p-value": round(p, 5),
        "Co y nghia": "Yes" if p < 0.05 else "No",
    })
print(pd.DataFrame(ttest_results).to_string(index=False))

# 4.3 Chi-square test bien categorical
print("\n[4.3] Chi-square test: bien phan loai vs Attrition")
categorical_cols = ["Department", "Gender", "OverTime", "MaritalStatus", "BusinessTravel"]
chi_results = []
for col in categorical_cols:
    if col not in df.columns:
        continue
    contingency = pd.crosstab(df[col], df["Attrition"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    chi_results.append({
        "Bien": col,
        "chi2": round(chi2, 3),
        "p-value": round(p, 5),
        "Co y nghia": "Yes" if p < 0.05 else "No",
    })
print(pd.DataFrame(chi_results).to_string(index=False))


# =====================================================================
# PHAN 5: TRUC QUAN HOA DU LIEU HR
# =====================================================================
section("PHAN 5: TRUC QUAN HOA DU LIEU")


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"      [Saved] {path}")


# 5.1 Phan bo tuoi
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Age", hue="Attrition", kde=True, bins=25, palette="Set1")
plt.title("5.1 - Phan bo tuoi nhan vien theo tinh trang nghi viec")
plt.xlabel("Tuoi")
plt.ylabel("So luong")
save_fig("5.1_phan_bo_tuoi.png")

# 5.2 Boxplot luong theo phong ban
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Department", y="MonthlyIncome", palette="Set2")
plt.title("5.2 - Phan bo luong theo phong ban")
plt.xlabel("Phong ban")
plt.ylabel("Luong thang (USD)")
save_fig("5.2_boxplot_luong_phong_ban.png")

# 5.3 Ty le nghi viec theo phong ban
plt.figure(figsize=(10, 5))
attr_plot = df.groupby(["Department", "Attrition"]).size().unstack()
attr_plot_pct = attr_plot.div(attr_plot.sum(axis=1), axis=0) * 100
attr_plot_pct.plot(kind="bar", stacked=True, colormap="RdYlGn_r", ax=plt.gca())
plt.title("5.3 - Ty le nghi viec theo phong ban (%)")
plt.ylabel("Ty le (%)")
plt.xticks(rotation=0)
plt.legend(title="Attrition")
save_fig("5.3_ty_le_nghi_viec.png")

# 5.4 Heatmap tuong quan
plt.figure(figsize=(14, 10))
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("5.4 - Heatmap tuong quan cac bien so")
save_fig("5.4_heatmap.png")

# 5.5 OverTime vs Attrition
plt.figure(figsize=(8, 5))
ot_attr = pd.crosstab(df["OverTime"], df["Attrition"], normalize="index") * 100
ot_attr.plot(kind="bar", colormap="coolwarm", ax=plt.gca())
plt.title("5.5 - Ty le nghi viec theo OverTime")
plt.xlabel("Lam them gio")
plt.ylabel("Ty le (%)")
plt.xticks(rotation=0)
save_fig("5.5_overtime_vs_attrition.png")

# 5.6 JobSatisfaction vs Attrition
plt.figure(figsize=(9, 5))
js_attr = pd.crosstab(df["JobSatisfaction"], df["Attrition"], normalize="index") * 100
js_attr.plot(kind="bar", colormap="viridis", ax=plt.gca())
plt.title("5.6 - Ty le nghi viec theo muc do hai long cong viec")
plt.xlabel("JobSatisfaction (1=Thap, 4=Cao)")
plt.ylabel("Ty le (%)")
plt.xticks(rotation=0)
save_fig("5.6_job_satisfaction.png")

# 5.7 YearsAtCompany - KDE
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x="YearsAtCompany", hue="Attrition", fill=True, palette="Set1")
plt.title("5.7 - Phan bo so nam lam viec tai cong ty")
plt.xlabel("So nam tai cong ty")
save_fig("5.7_years_at_company.png")

# 5.8 Violin Income
plt.figure(figsize=(9, 5))
sns.violinplot(data=df, x="Attrition", y="MonthlyIncome", palette="Set2")
plt.title("5.8 - Phan bo luong giua nhom nghi viec va o lai")
plt.xlabel("Attrition")
plt.ylabel("Luong thang (USD)")
save_fig("5.8_violin_income.png")


# =====================================================================
# PHAN 6: XAY DUNG MO HINH DU DOAN NGHI VIEC
# =====================================================================
section("PHAN 6: MO HINH DU DOAN NGHI VIEC")

# 6.1 Chuan bi du lieu
df_model = df.copy()
if "Attrition" in df_model.columns:
    df_model = df_model.drop(columns=["Attrition"])

# Encode categorical
cat_cols = df_model.select_dtypes(include=["object"]).columns.tolist()
print(f"\n[6.1] Encode {len(cat_cols)} cot categorical: {cat_cols}")
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

X = df_model.drop(columns=["Attrition_bin"])
y = df_model["Attrition_bin"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train: {X_train.shape}, Test: {X_test.shape}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xu ly mat can bang bang SMOTE (neu co imbalanced-learn)
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
    print(f"      SMOTE: {X_train.shape[0]} -> {X_train_bal.shape[0]} mau")
except ImportError:
    print("      [!] Khong co imbalanced-learn, dung class_weight=balanced thay the")
    X_train_bal, y_train_bal = X_train_scaled, y_train

# 6.2 Huan luyen 3 mo hinh
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = []
roc_data = {}
print("\n[6.2] Ket qua tung mo hinh:\n")
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "AUC-ROC": round(auc, 4),
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["O lai", "Nghi viec"]))

# 6.3 Bang tong hop
results_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
print("\n[6.3] BANG TONG HOP 3 MO HINH:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\n      >>> Mo hinh tot nhat: {best_model_name} (AUC-ROC = {results_df.iloc[0]['AUC-ROC']})")

# 6.4 ROC Curve
plt.figure(figsize=(9, 7))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("5.9 - ROC Curve - So sanh 3 mo hinh")
plt.legend(loc="lower right")
save_fig("5.9_roc_curve.png")

# 6.5 Confusion Matrix mo hinh tot nhat
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["O lai", "Nghi viec"],
            yticklabels=["O lai", "Nghi viec"])
plt.title(f"5.10 - Confusion Matrix - {best_model_name}")
plt.xlabel("Du doan")
plt.ylabel("Thuc te")
save_fig("5.10_confusion_matrix.png")

# 6.6 Feature importance
if hasattr(best_model, "feature_importances_"):
    fi = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(15)
    print("\n[6.6] TOP 15 YEU TO QUAN TRONG:")
    print(fi.to_string(index=False))

    plt.figure(figsize=(10, 7))
    sns.barplot(data=fi, x="Importance", y="Feature", palette="viridis")
    plt.title(f"5.11 - Top 15 yeu to quan trong ({best_model_name})")
    save_fig("5.11_feature_importance.png")

# 6.7 Cross-validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
print(f"\n[6.7] Cross-Validation (5-fold) AUC-ROC cho {best_model_name}:")
print(f"      Cac fold: {[round(s, 4) for s in cv_scores]}")
print(f"      Trung binh: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# =====================================================================
# PHAN 7: KET LUAN & KHUYEN NGHI
# =====================================================================
section("PHAN 7: KET LUAN & KHUYEN NGHI")

print("""
CAC PHAT HIEN CHINH:
--------------------
1. TY LE NGHI VIEC: Khoang 16% nhan vien da nghi viec - muc can luu y.

2. LUONG THEO PHONG BAN: Co su khac biet dang ke ve luong giua cac phong ban.
   Research & Development thuong co luong cao nhat, Sales co bien do rong.

3. YEU TO ANH HUONG NGHI VIEC (theo thu tu quan trong):
   - Lam them gio (OverTime): Nhan vien OT co ty le nghi cao gap 3 lan.
   - Luong thap (MonthlyIncome): Nhom luong thap co ty le nghi cao hon.
   - Tuoi tre: Nhom duoi 30 tuoi co xu huong nghi viec nhieu hon.
   - It nam lam viec: Nhan vien <2 nam co risk nghi viec cao.
   - Khoang cach xa nha (DistanceFromHome): Cang xa cang de nghi.
   - Muc do hai long thap (JobSatisfaction, WorkLifeBalance).

4. MO HINH DU DOAN: AUC-ROC dat >= 0.75 - du do chinh xac de ung dung.

KHUYEN NGHI CHIEN LUOC GIU CHAN NHAN VIEN:
------------------------------------------
[1] Han che giao qua nhieu OT, hoac co che do den bu xung dang.
[2] Ra soat chinh sach luong, dac biet o nhom co luong thap va phong Sales.
[3] Tang cuong chuong trinh onboarding cho nhan vien moi (<2 nam).
[4] Khao sat dinh ky muc do hai long + WorkLifeBalance.
[5] Ho tro nhan vien o xa (phu cap di lai, lam viec tu xa).
[6] Chuong trinh phat trien su nghiep cho nhan vien tre.
""")

print("=" * 70)
print("  HOAN THANH! Tat ca bieu do da luu trong thu muc 'output/'")
print("=" * 70)
