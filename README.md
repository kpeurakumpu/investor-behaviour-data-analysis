# Investor Behaviour Classes Based on Financial Risk Tolerance, Investment Experience, Investment Activity and Financial Knowledge

## Overview

This project investigates whether distinct investor classes can be identified based on financial risk tolerance, investment experience, investment activity and financial knowledge.

Using data from the **2024 National Financial Capability Study (NFCS)**, conducted by the FINRA Investor Education Foundation, the project applies descriptive statistics, Latent Class Analysis (LCA) and auxiliary variable analysis using Python.

The study aims to identify distinct investor classes and examine whether they differ in terms of demographic and socioeconomic characteristics.

The project was developed as part of a university course assignment, providing practical experience in statistical analysis, mixture modelling and data-driven investor segmentation. Additional data exploration and visualization were supported by Power BI and Microsoft Excel.

## Research Questions

The analysis addresses two main research questions:

1. What distinct investor classes can be identified based on financial risk tolerance, investment experience, investment activity and financial knowledge?
2. To what extent do the identified investor classes differ in terms of demographic and socioeconomic characteristics?

## Dataset

- **Dataset:** 2024 National Financial Capability Study (NFCS)
- **Source:** FINRA Investor Education Foundation
- **Original format:** SPSS (`.sav`)

The NFCS contains survey data on individuals’ financial capability, including several investment and finance related questions and answers

This project uses selected survey variables to construct investor classes and investigate differences between the resulting classes.

**Data availability:** The original dataset is not included in this repository. Users must obtain the data through the appropriate NFCS data access channels.

## Methodology

The project consists of three main stages.

### Part 1: Data Preparation and Descriptive Statistics

The first stage focuses on preparing the dataset and exploring the selected variables.

The analysis includes:

- Importing the SPSS dataset into Python using `pyreadstat`.
- Selecting the relevant variables for investor classes.
- Identifying and handling missing values and special response codes.
- Reverse-coding the financial risk tolerance variable to ensure consistent interpretation.
- Constructing a financial knowledge score based on 12 financial knowledge questions.
- Calculating descriptive statistics and frequency distributions.
- Examining relationships between the selected variables using Spearman correlation.
- Preparing the final dataset for latent class analysis.

The financial knowledge score is calculated as the proportion of correctly answered questions, ranging from 0 to 1.

### Part 2: Latent Class Analysis (LCA)

Latent Class Analysis is used to identify groups of respondents with similar financial and investment characteristics.

The analysis is conducted using the `StepMix` Python package.

The modelling process includes:

- Estimating models with one to six latent classes.
- Comparing model fit using Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).
- Examining log-likelihood and entropy.
- Evaluating class sizes and classification certainty.
- Examining average posterior probabilities.
- Selecting and interpreting the final class solution.
- Visualizing the identified investor classes.

A five class solution was selected based on the lowest BIC among the models tested, alongside considerations of other measures, class sizes and interpretability.

### Part 3: Auxiliary Variable Analysis

Following the identification of the investor classes, auxiliary variable analysis is conducted to examine differences in demographic and socioeconomic characteristics.

These variables are not used to construct the latent classes. Instead, they are used to describe and compare the resulting investor classes.

The analysis includes:

**Demographic variables:**
- Gender
- Age
- Ethnicity

**Socioeconomic variables:**
- Educational attainment
- Household income

Statistical methods include:

- Chi-square tests to examine associations between classes and auxiliary variables.
- Cramér's V to measure the strength of these associations.

## Part 4: Data Visualization

Following the data analysis in Python, Power BI and Microsoft Excel were used to create visualizations and interactive dashboards to present the findings.

### Power BI
- Used **Power Query** for data transformation and preparation.
- Created measures for the total number of respondents and the number of investor classes.
- Applied **Min-Max Normalization** to standardize responses and scores.
- Created bar chart, doughnut chart and stacked column chart.
- Sorted visualizations by the number of investor classes or total number of respondents.
- Combined the visualizations into an interactive dashboard.

### Microsoft Excel
- Formatted the dataset as an Excel Table.
- Created **PivotTables** to summarize and organize the data.
- Generated **PivotCharts** based on the PivotTables.
- Combined the charts into a dashboard to provide a visual overview of the findings.

## Investor Class Variables

Five variables are used to construct the investor classes.

| Indicator | NFCS Variable | Description |
|---|---|---|
| Risk tolerance | `B10_reversed` | Willingness to take financial risks when saving or investing |
| Risk-taking beliefs | `B40` | Agreement that taking big investment risks is necessary to achieve financial goals |
| Investment experience | `B30_2024` | Length of investment experience in non-retirement accounts |
| Investment activity | `B3` | Frequency of buying or selling investments during the previous 12 months |
| Financial knowledge | `knowledge_score` | Proportion of 12 financial knowledge questions answered correctly |

The financial knowledge score ranges from 0 to 1, with higher values indicating a greater proportion of correct answers.

For the financial knowledge score, responses coded as 98 (don't know) and 99 (prefer not to say) are treated as incorrect answers. For the four categorical investment indicators, these response codes are treated as missing values.

## Key Findings

### Five Investor Classes Identified

The five class Latent Class Analysis solution identified the following investor profiles:

| Investor Class | Profile Name | Number of Respondents | Percentage |
|---|---|---:|---:|
| Class 1 | Moderately Cautious Investors | 668 | 26,52% |
| Class 2 | Risk-Averse, Inactive Investors | 352 | 13,97% |
| Class 3 | Risk-Seeking, Active Investors | 358 | 14,21% |
| Class 4 | Knowledgeable, Risk-Averse Investors | 617 | 24,49% |
| Class 5 | Knowledgeable, Moderate-to-High-Risk Investors | 524 | 20,80% |
| **Total** | | **2519** | **100%** |

### Main Characteristics of the Investor Classes

- **Class 1 – Moderately Cautious Investors:** The largest class, characterised by moderate risk tolerance, relatively low investment activity and comparatively low financial knowledge.

- **Class 2 – Risk-Averse, Inactive Investors:** Respondents with the lowest risk tolerance, low investment activity and the lowest average financial knowledge score.

- **Class 3 – Risk-Seeking, Active Investors:** Respondents with relatively high risk tolerance, greater investment activity and comparatively low financial knowledge.

- **Class 4 – Knowledgeable, Risk-Averse Investors:** Respondents characterised by low risk tolerance, relatively low investment activity and high financial knowledge.

- **Class 5 – Knowledgeable, Moderate-to-High-Risk Investors:** Respondents with relatively high risk tolerance, higher investment activity and the highest average financial knowledge score.

### Financial Knowledge Across Investor Classes

The average financial knowledge scores differed across the five classes:

| Investor Class | Average Knowledge Score |
|---|---:|
| Class 1 | 0,358 |
| Class 2 | 0,319 |
| Class 3 | 0,379 |
| Class 4 | 0,607 |
| Class 5 | 0,683 |

Classes 4 and 5 demonstrated considerably higher average financial knowledge than Classes 1, 2 and 3. Class 5 had the highest average score, while Class 2 had the lowest.

### Demographic and Socioeconomic Differences

Chi-square tests identified statistically significant associations between investor classes and all five auxiliary variables:

- Gender
- Age
- Ethnicity
- Educational attainment
- Household income

Age showed the strongest association with investor class membership (Cramér's V = 0,318), followed by gender (V = 0,281). Household income showed the weakest association (V = 0,145).

The results suggest that the identified investor classes differ not only in their investment characteristics and financial knowledge, but also in their demographic and socioeconomic structure.

## Power BI and Microsoft Excel dashboards

The project includes two interactive dashboards developed using and **Power BI** and **Microsoft Excel** to analyze the results.

For data privacy reasons, the dashboards are provided as static exports in **PDF** format.

### 1. Power BI Dashboard – Investor Classes & Behaviour

The Power BI dashboard was developed using **Power Query** for data transformation and preparation, alongside Power BI visualizations to explore investor classification and behavioural patterns.

**Features:**
- **Cards**
  - Total number of respondents
  - Number of investor classes
- **Bar Chart:** Number of respondents in each investor class.
- **Doughnut Chart:** Percentage distribution of respondents across investor classes.
- **Stacked Column Chart** Average standardized score (0-100) by investor class.

### 2. Excel Dashboard – Investor Demographics & Socioeconomic Characteristics

The Excel dashboard was developed using **Pivot Tables** and Pivot Charts to visualize the distributions of demographic and socioeconomic characteristics, as well as the distributions of these characteristics in each class.

**Features:**
- **Doughnut Charts:** Distribution of respondents across five demographic and socioeconomic characteristics.
- **Bar Charts:** Distribution of each of the five characteristics by investor class, enabling comparisons between investor classes.

## Language and Libraries

Python was used as the primary programming language for data preparation, statistical analysis and visualization.

| Language / Library | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Data cleaning, manipulation and preparation |
| NumPy | Numerical operations and calculations |
| pyreadstat | Importing and reading SPSS data files |
| SciPy | Statistical tests and correlation analysis |
| statsmodels | Statistical testing and multiple-testing correction |
| Matplotlib | Creating charts and visualizations |
| openpyxl | Exporting analysis results to Excel workbooks |
| Power BI | Data exploration and visualization |
| Microsoft Excel | Data exploration and visualization |

## Project Structure

```text
investor-behaviour-data_analysis/
│
├── report/
│   └── research_report_paper.pdf
│
├── python/
│   ├── 01_data_preparation.ipynb
│   ├── 02_latent_class_analysis.ipynb
│   ├── 03_auxiliary_variable_analysis.ipynb
│   ├── 04_cleaned_data_to_excel.ipynb
│   ├── main.py
│   └── requirements.txt
│
├── excel/
│   ├── excel_dashboard.pdf
│   ├── data_preparation_results.xlsx
│   ├── latent_class_analysis_results.xlsx
│   └── auxiliary_variable_analysis_results.xlsx
│
├── power_bi/
│   └── power_bi_dashboard.pdf
│
└── README.md
```

## Acknowledgements

This project uses data from the **2024 National Financial Capability Study (NFCS)**, conducted by the FINRA Investor Education Foundation.

The analysis was developed for academic purposes as part of a university course assignment.

The findings and interpretations presented in this project are the author's own and should not be interpreted as official conclusions or recommendations from the FINRA Investor Education Foundation.
