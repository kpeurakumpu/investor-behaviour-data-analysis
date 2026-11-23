#####################################################################
## FINAL ASSIGNMENT / STUDY
## Advanced course in structural equation modeling: mixture modeling
## Kaapo Peurakumpu
#####################################################################


# Libraries
import pandas as pd
import numpy as np
import pyreadstat
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from stepmix.stepmix import StepMix
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Data
df, meta = pyreadstat.read_sav("Data/NFCS 2024 Investor Data SPSS.sav")
print(df.head())
print(df.shape)
print(df.columns.tolist())


#####################################################################
## 1. DATA PREPARATION
#####################################################################

# 1.1 Variables
risk_variables = ["B10", "B40"]
investment_variables = ["B30_2024", "B3"]
knowledge_variables = ["G4", "G5", "G6", "G7", "G21", "G8", "G44", "G22", "G11", "G12", "G13", "G23"]
study_data = df[risk_variables + investment_variables + knowledge_variables].copy()
print(study_data)

# 1.2 Missing values
missing_summary = pd.DataFrame({
    "Missing (NaN)": study_data.isna().sum(),
    "Don't know (98)": study_data.eq(98).sum(),
    "Prefer not to say (99)": study_data.eq(99).sum()
})
missing_summary["Total"] = missing_summary.sum(axis=1)
print(missing_summary)

# 1.3 New Variable (knowledge_score)
answer_key = {
    "G4": 1, "G5": 2, "G6": 2, "G7": 1, "G21": 2, "G8": 1, 
    "G44": 3, "G22": 2, "G11": 3, "G12": 0, "G13": 4, "G23": 2
}
correct = pd.DataFrame(index=study_data.index)
for col, answer in answer_key.items():
    correct[col] = (study_data[col].eq(answer)).astype(int)
study_data["knowledge_score"] = correct.sum(axis=1) / len(answer_key)
print(study_data["knowledge_score"].head(10))

# 1.4 Clean and reverse-code investment indicators
investment_indicators = ["B10", "B40", "B30_2024", "B3"]
study_data[investment_indicators] = study_data[investment_indicators].replace(
    [98, 99], np.nan
)
study_data["B10_reversed"] = 5 - study_data["B10"]
print(
    study_data[["B10", "B10_reversed"]]
    .drop_duplicates()
    .sort_values("B10")
)

# 1.5 Final LCA indicators
lca_variables = ["B10_reversed", "B40", "B30_2024", "B3", "knowledge_score"]
lca_data = study_data[lca_variables].copy()
print(lca_data.head())

# 1.6 Descriptive statistics
descriptives = lca_data.describe().T.rename(columns={
    "count": "N", "mean": "Mean", "std": "SD", "min": "Minimum", 
    "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Maximum"
})
print(descriptives.round(3))

# 1.7 Frequency distributions and plots
ordinal_variables = ["B10_reversed", "B40", "B30_2024", "B3"]
frequency_tables = {}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for v, ax in zip(ordinal_variables, axes.flatten()):
    counts = lca_data[v].value_counts(dropna=False).sort_index()
    frequency_tables[v] = pd.DataFrame({
        "Frequency": counts,
        "Percentage": counts / counts.sum() * 100
    })
    ax.bar(
        range(len(counts)),
        counts.values.astype(float)
    )
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str))
    ax.set(title=v, ylabel="Frequency", xlabel="Response category")
plt.tight_layout()
plt.show()

# 1.8 Knowledge score distribution
scores = study_data["knowledge_score"].dropna()
mean_score, median_score = scores.mean(), scores.median()
plt.figure(figsize=(9, 5))
plt.hist(scores, bins=np.linspace(0, 1, 13), edgecolor="black", alpha=0.8)
plt.axvline(mean_score, linestyle="--", linewidth=2,
            label=f"Mean = {mean_score:.2f}")
plt.axvline(median_score, linestyle=":", linewidth=2,
            label=f"Median = {median_score:.2f}")
plt.title("Distribution of Financial Knowledge Scores")
plt.xlabel("Proportion of Correct Answers")
plt.ylabel("Number of Respondents")
plt.xticks(np.linspace(0, 1, 11))
plt.xlim(0, 1)
plt.legend()
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()

# 1.9 Spearman correlations and heatmap
correlation_matrix = lca_data.corr(method="spearman")
correlation_n = lca_data.notna().astype(int).T.dot(
    lca_data.notna().astype(int)
)
print("\nSPEARMAN CORRELATIONS\n", correlation_matrix.round(3))
print("\nCORRELATION SAMPLE SIZES\n", correlation_n)
fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
ax.set(xticks=range(len(lca_variables)), yticks=range(len(lca_variables)),
       xticklabels=lca_variables, yticklabels=lca_variables,
       title="Spearman Correlation Matrix")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for i in range(len(lca_variables)):
    for j in range(len(lca_variables)):
        ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center", va="center")
fig.colorbar(image, ax=ax, label="Spearman correlation")
plt.tight_layout()
plt.show()

# 1.10 Prepare data for LCA
categorical_variables = ["B10_reversed", "B40", "B30_2024", "B3"]
continuous_variables = ["knowledge_score"]
lca_variables = categorical_variables + continuous_variables
lca_complete = lca_data[lca_variables].dropna().copy()
category_maps = {}
for v in categorical_variables:
    lca_complete[v] = lca_complete[v].astype("category")
    category_maps[v] = dict(enumerate(lca_complete[v].cat.categories))
    lca_complete[v] = lca_complete[v].cat.codes
X_lca = lca_complete.to_numpy(dtype=float)
n = len(X_lca)
print(lca_complete.head())
print("\nSample size:", n)
print("\nCategory mappings:", category_maps)

# 1.11 Additional summary tables
sample_overview = pd.DataFrame({
    "Measure": ["Total observations", "Complete observations",
                "Excluded observations", "Percentage excluded"],
    "Value": [len(lca_data), len(lca_complete),
              len(lca_data) - len(lca_complete),
              lca_data.isna().any(axis=1).mean() * 100]
})
knowledge_summary = study_data["knowledge_score"].describe().to_frame(
    "Knowledge score"
)
knowledge_frequency = (
    study_data["knowledge_score"].value_counts().sort_index()
    .to_frame("Frequency")
)
knowledge_frequency["Percentage"] = (
    knowledge_frequency["Frequency"] /
    knowledge_frequency["Frequency"].sum() * 100
)

# 1.12 Export data preparation results
with pd.ExcelWriter("Results/data_preparation_results.xlsx", engine="openpyxl") as writer:
    for name, data in {
        "Sample overview": sample_overview,
        "Missing values": missing_summary,
        "Descriptive statistics": descriptives,
        "Spearman correlations": correlation_matrix,
        "Knowledge summary": knowledge_summary,
        "Knowledge frequency": knowledge_frequency
    }.items():
        data.to_excel(writer, sheet_name=name)
    for v, table in frequency_tables.items():
        table.to_excel(writer, sheet_name=f"Freq_{v}")


#####################################################################
## 2. LATENT CLASS ANALYSIS
#####################################################################

# 2.1 Mixed measurement model
measurement_model = {
    "categorical": {"model": "categorical", "n_columns": 4},
    "continuous": {"model": "gaussian_diag", "n_columns": 1}
}

# 2.2 Models comparison (1-6 classes)
models, results = {}, []

for k in range(1, 7):
    print(f"Estimating {k} classes...", flush=True)

    m = StepMix(
        n_components=k,
        measurement=measurement_model,
        n_init=20,
        max_iter=500,
        random_state=123,
        abs_tol=1e-6,
        rel_tol=1e-6,
        progress_bar=0
    ).fit(X_lca)

    models[k] = m

    # Posterior probabilities and class sizes
    posterior = m.predict_proba(X_lca)
    assignments = posterior.argmax(axis=1)
    class_counts = np.bincount(assignments, minlength=k)
    class_sizes = ", ".join(
        f"{count} ({count / len(X_lca) * 100:.1f}%)"
        for count in class_counts
    )

    # Entropy
    if k > 1:
        entropy = 1 - (
            -np.sum(
                posterior * np.log(np.clip(posterior, 1e-10, 1)),
                axis=1
            ).mean() / np.log(k)
        )
    else:
        entropy = np.nan

    results.append({
        "Classes": k,
        "Log likelihood": m.score(X_lca) * len(X_lca),
        "AIC": m.aic(X_lca),
        "BIC": m.bic(X_lca),
        "Entropy": entropy,
        "Class sizes (N, %)": class_sizes,
    })
    print(f"Finished {k} classes. Converged: {m.converged_}")

# Model comparison
model_comparison = pd.DataFrame(results)

model_comparison[
    ["Log likelihood", "AIC", "BIC", "Entropy"]
] = model_comparison[
    ["Log likelihood", "AIC", "BIC", "Entropy"]
].round(3)

print("\nMODEL COMPARISON")
print(model_comparison.to_string(index=False))

# 2.3 Final model
SELECTED_K = 5
final_model = models[SELECTED_K]
print(f"\nSelected model: {SELECTED_K}")
print("Converged:", final_model.converged_)

# 2.4 Assign classes and posterior probabilities
posterior = final_model.predict_proba(X_lca)
class_data = lca_complete.copy()
class_data["Class"] = posterior.argmax(axis=1) + 1
posterior_df = pd.DataFrame(
    posterior,
    columns=[f"Probability_Class_{i+1}" for i in range(SELECTED_K)],
    index=class_data.index
)
class_data = pd.concat([class_data, posterior_df], axis=1)
class_data["Maximum_Probability"] = posterior.max(axis=1)

# 2.5 Class sizes
class_sizes = class_data.groupby("Class").size().to_frame("N")
class_sizes["Percentage"] = class_sizes.N / len(class_data) * 100
print("\nCLASS SIZES\n", class_sizes.round(2))

# 2.6 Knowledge score by class
knowledge_summary = class_data.groupby("Class")["knowledge_score"].agg(
    ["count", "mean", "std", "median", "min", "max"]
)
print("\nKNOWLEDGE SCORE BY CLASS\n", knowledge_summary.round(3))

# 2.7 Classification certainty
posterior_columns = [
    f"Probability_Class_{i+1}" for i in range(SELECTED_K)
]
classification_certainty = class_data.groupby("Class")[
    posterior_columns
].mean()
print("\nAVERAGE POSTERIOR PROBABILITIES\n",
      classification_certainty.round(3))
print("\nMAXIMUM POSTERIOR PROBABILITY\n",
      class_data["Maximum_Probability"].describe().round(3))

# 2.8 Investor class characteristics
categorical_variables = ["B10_reversed", "B40", "B30_2024", "B3"]
class_table = []
class_sizes = class_data["Class"].value_counts().sort_index()
for class1 in range(1, SELECTED_K + 1):
    subset = class_data[class_data["Class"] == class1]
    class_table.append({
        "Indicator": "Class size (N)",
        f"Class {class1}": len(subset)
    })
    class_table.append({
        "Indicator": "Class size (%)",
        f"Class {class1}": round(
            len(subset) / len(class_data) * 100, 2
        )
    })
    for variable in categorical_variables:
        original_labels = category_maps[variable]

        for code, label in original_labels.items():
            percentage = (
                subset[variable].eq(code).mean() * 100
            )
            class_table.append({
                "Indicator": f"{variable}: {label}",
                f"Class {class1}": round(percentage, 2)
            })
    knowledge_mean = subset["knowledge_score"].mean()
    knowledge_sd = subset["knowledge_score"].std()
    class_table.append({
        "Indicator": "Financial knowledge: Mean",
        f"Class {class1}": round(knowledge_mean, 3)
    })
    class_table.append({
        "Indicator": "Financial knowledge: SD",
        f"Class {class1}": round(knowledge_sd, 3)
    })
class_characteristics = pd.DataFrame(class_table)
class_characteristics = (
    class_characteristics
    .groupby("Indicator", sort=False)
    .first()
    .reset_index()
)
print("\nINVESTOR CLASS CHARACTERISTICS")
print(class_characteristics.to_string(index=False))

# 2.9 Scatter plot of latent classes
final_model = models[SELECTED_K]
posterior = final_model.predict_proba(X_lca)
assignments = posterior.argmax(axis=1) + 1
X_scaled = StandardScaler().fit_transform(X_lca)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plot_data = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Class": assignments
})
fig, ax = plt.subplots(figsize=(11, 8))
colors = sns.color_palette("colorblind", SELECTED_K)
for i, class2 in enumerate(range(1, SELECTED_K + 1)):
    subset = plot_data[plot_data["Class"] == class2]
    x, y = subset["PC1"].values, subset["PC2"].values
    ax.scatter(x, y, color=colors[i], label=f"Class {class2}",
               alpha=0.35, s=18, edgecolors="none")
    if len(subset) > 2:
        cov = np.cov(x, y)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        ellipse = Ellipse(
            (x.mean(), y.mean()),
            2 * np.sqrt(eigenvalues[0] * 5.991),
            2 * np.sqrt(eigenvalues[1] * 5.991),
            angle=angle, facecolor="none",
            edgecolor=colors[i], linewidth=2
        )
        ax.add_patch(ellipse)
ax.set(
    title="Distribution of Five Investor Classes",
    xlabel="Investor Characteristics Dimension 1",
    ylabel="Investor Characteristics Dimension 2"
)
ax.legend(title="Investor Classes")
ax.grid(alpha=0.15)
plt.tight_layout()
plt.show()

# 2.10 Export latent class analysis results
with pd.ExcelWriter("Results/latent_class_analysis_results.xlsx", engine="openpyxl") as writer:
    model_comparison.to_excel(writer, sheet_name="Model comparison", index=False)
    class_sizes.to_excel(writer, sheet_name="Class sizes")
    knowledge_summary.to_excel(writer, sheet_name="Knowledge by class")
    classification_certainty.to_excel(writer, sheet_name="Posterior probabilities")
    class_characteristics.to_excel(writer, sheet_name="Class Characteristics")


#####################################################################
## 3. AUXILIARY VARIABLE ANALYSIS
#####################################################################

# 3.1 Auxiliary variables
auxiliary_variables = ['S_Gender2', 'S_Age', 'S_Ethnicity', 'S_Education', 'S_Income']

# 3.2 Match auxiliary data to LCA data
aux_data = df.loc[class_data.index, auxiliary_variables].copy()
aux_data["Class"] = class_data["Class"]
print("Auxiliary dataset shape:", aux_data.shape)

# 3.3 Auxiliary variables frequencies in each latent class
descriptive_tables = {}
frequency_tables = {}
for var in auxiliary_variables:
    clean = aux_data[["Class", var]]
    frequency = pd.crosstab(
        clean["Class"],
        clean[var]
    )
    percentages = pd.crosstab(
        clean["Class"],
        clean[var],
        normalize="index"
    ) * 100
    frequency_tables[var] = frequency
    descriptive_tables[var] = percentages.round(1)

# 3.4 Chi-Square and Cramer's V tests
results = []
for var in auxiliary_variables:
    clean = aux_data[["Class", var]]
    table = pd.crosstab(
        clean["Class"],
        clean[var]
    )
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    rows, columns = table.shape
    if min(rows - 1, columns - 1) > 0:
        cramers_v = np.sqrt(
            chi2 / (n * min(rows - 1, columns - 1))
        )
    else:
        cramers_v = np.nan
    results.append({
        "Variable": var,
        "N": n,
        "Chi-square": chi2,
        "df": dof,
        "p-value": p,
        "Cramer's V": cramers_v,
    })
print(pd.DataFrame(results))

# 3.5 Combine descriptive tables
combined_tables = []
for var in auxiliary_variables:
    percentages = descriptive_tables[var].copy()
    percentages.index = [
        f"Class {class3}"
        for class3 in percentages.index
    ]
    percentages.columns = [
        f"{var}: {category}"
        for category in percentages.columns
    ]
    combined_tables.append(percentages)
combined_table = pd.concat(
    combined_tables,
    axis=1
)
print("COMBINED CLASS DISTRIBUTIONS (%)")
print(combined_table.round(1))

# 3.6 Export auxiliary variable analysis results
results = pd.DataFrame(results)
with pd.ExcelWriter("Results/auxiliary_variable_analysis_results.xlsx",
                    engine="openpyxl") as writer:
    results.to_excel(
            writer,
            sheet_name="Chi-square Results",
            index=False
        )
    combined_table.to_excel(
        writer,
        sheet_name="Combined Percentages"
        )


#####################################################################
## 4. SAVE CLEANED DATASET
#####################################################################

# Prepare data for Excel/Power BI visualizations
excel_data = pd.concat([lca_complete, aux_data], axis=1)
excel_data.insert(0, "Respondent ID", range(1, len(excel_data) + 1))
value_labels = {
    "S_Gender2": {1: "Male", 2: "Female"},
    "S_Age": {1: "18-34", 2: "35-54", 3: "55+"},
    "S_Ethnicity": {1: "White non-Hispanic", 2: "Non-White"},
    "S_Education": {1: "Some college or less", 2: "College graduate or higher"},
    "S_Income": {1: "Below $50K", 2: "$50K-$100K", 3: "$100K+"}
}
excel_data = excel_data.replace(value_labels)
profile_labels = {
    1: "Moderately Cautious Investors",
    2: "Risk-Averse, Inactive Investors",
    3: "Risk-Seeking, Active Investors",
    4: "Knowledgeable, Risk-Averse Investors",
    5: "Knowledgeable, Risk-Seeking Investors"
}
excel_data["Class"] = excel_data["Class"].replace(profile_labels)
column_names = {
    "S_Gender2": "Gender",
    "S_Age": "Age Group",
    "S_Ethnicity": "Ethnicity",
    "S_Education": "Education",
    "S_Income": "Household Income",
    "B10_reversed": "Risk Tolerance",
    "B40": "Risk-Taking Beliefs",
    "B30_2024": "Investment Experience",
    "B3": "Investment Activity",
    "knowledge_score": "Financial Knowledge",
    "Class": "Investor Class"
}
excel_data = excel_data.rename(columns=column_names)
excel_data

# Export cleaned data for Excel and Power BI visualizations
excel_data.to_excel(
    "cleaned_data.xlsx",
    sheet_name="Data",
    index=False
)