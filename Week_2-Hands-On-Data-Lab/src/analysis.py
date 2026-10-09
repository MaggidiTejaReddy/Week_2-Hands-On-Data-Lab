
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the dataset
data_file = DATA_DIR / "student_performance.csv"
df = pd.read_csv(data_file)

print("=" * 60)
print("WEEK 2: HANDS-ON DATA LAB")
print("=" * 60)

print("\n1. DATASET OVERVIEW")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nFirst five records:")
print(df.head())

# 2. Data-quality checks
print("\n2. DATA QUALITY CHECKS")
missing_values = df.isnull().sum()
duplicate_rows = int(df.duplicated().sum())
duplicate_ids = int(df["Student_ID"].duplicated().sum())

print("\nMissing values by column:")
print(missing_values)
print("\nTotal missing values:", int(missing_values.sum()))
print("Duplicate rows:", duplicate_rows)
print("Duplicate student IDs:", duplicate_ids)

quality_report = pd.DataFrame({
    "Check": [
        "Total records",
        "Total columns",
        "Total missing values",
        "Duplicate rows",
        "Duplicate student IDs"
    ],
    "Result": [
        len(df),
        len(df.columns),
        int(missing_values.sum()),
        duplicate_rows,
        duplicate_ids
    ]
})
quality_report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

# 3. Data manipulation and cleaning
# Remove duplicate rows, if any
clean_df = df.drop_duplicates().copy()

# Remove records with missing Student_ID values
clean_df = clean_df.dropna(subset=["Student_ID"])

# Convert score and measurement columns to numeric values
numeric_columns = [
    "Age", "Study_Hours", "Attendance",
    "Assignments_Score", "Midterm_Score", "Final_Score"
]

for column in numeric_columns:
    clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

print("\n3. DATA CLEANING")
print("Records before cleaning:", len(df))
print("Records after cleaning:", len(clean_df))
print("Missing values after cleaning:", int(clean_df.isnull().sum().sum()))

# Save cleaned dataset
clean_df.to_csv(OUTPUT_DIR / "cleaned_data.csv", index=False)

# 4. Descriptive statistics
print("\n4. SUMMARY STATISTICS")
summary = clean_df[numeric_columns].describe().round(2)
print(summary)
summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")

# 5. Grouping and comparison by gender
print("\n5. PERFORMANCE BY GENDER")
gender_summary = clean_df.groupby("Gender").agg(
    Student_Count=("Student_ID", "count"),
    Average_Study_Hours=("Study_Hours", "mean"),
    Average_Attendance=("Attendance", "mean"),
    Average_Final_Score=("Final_Score", "mean")
).round(2)

print(gender_summary)
gender_summary.to_csv(OUTPUT_DIR / "gender_performance_summary.csv")

# 6. Add performance categories
clean_df["Performance"] = pd.cut(
    clean_df["Final_Score"],
    bins=[-float("inf"), 59, 74, 89, float("inf")],
    labels=[
        "Needs Improvement",
        "Average",
        "Good",
        "Excellent"
    ]
)

performance_counts = (
    clean_df["Performance"]
    .value_counts()
    .reindex(
        ["Excellent", "Good", "Average", "Needs Improvement"],
        fill_value=0
    )
)

print("\n6. PERFORMANCE CATEGORY COUNTS")
print(performance_counts)
performance_counts.rename_axis("Performance").to_csv(
    OUTPUT_DIR / "performance_category_counts.csv",
    header=["Student_Count"]
)

# Save analyzed dataset
clean_df.to_csv(OUTPUT_DIR / "student_performance_analyzed.csv", index=False)

# 7. Find the top five students
print("\n7. TOP FIVE STUDENTS")
top_students = clean_df.nlargest(5, "Final_Score")[
    ["Student_ID", "Study_Hours", "Attendance", "Final_Score"]
]
print(top_students.to_string(index=False))
top_students.to_csv(OUTPUT_DIR / "top_five_students.csv", index=False)

# 8. Create visualizations
sns.set_theme(style="whitegrid")

# Chart 1: Distribution of final scores
plt.figure(figsize=(8, 5))
sns.histplot(clean_df["Final_Score"], bins=8, kde=True)
plt.title("Distribution of Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(CHART_DIR / "final_score_distribution.png", dpi=150)
plt.close()

# Chart 2: Study hours versus final score
plt.figure(figsize=(8, 5))
sns.regplot(
    data=clean_df,
    x="Study_Hours",
    y="Final_Score",
    scatter_kws={"s": 60}
)
plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.tight_layout()
plt.savefig(CHART_DIR / "study_hours_vs_final_score.png", dpi=150)
plt.close()

# Chart 3: Attendance versus final score
plt.figure(figsize=(8, 5))
sns.regplot(
    data=clean_df,
    x="Attendance",
    y="Final_Score",
    scatter_kws={"s": 60}
)
plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.tight_layout()
plt.savefig(CHART_DIR / "attendance_vs_final_score.png", dpi=150)
plt.close()

# Chart 4: Correlation heatmap
plt.figure(figsize=(9, 6))
correlation = clean_df[numeric_columns].corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Between Student Variables")
plt.tight_layout()
plt.savefig(CHART_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

# Chart 5: Average final score by gender
plt.figure(figsize=(7, 5))
sns.barplot(
    data=clean_df,
    x="Gender",
    y="Final_Score",
    errorbar=None
)
plt.title("Average Final Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Final Score")
plt.tight_layout()
plt.savefig(CHART_DIR / "average_score_by_gender.png", dpi=150)
plt.close()

print("\n8. ANALYSIS COMPLETE")
print("Output files saved in:", OUTPUT_DIR)
print("Charts saved in:", CHART_DIR)
print("\nPlease check the outputs folder for generated files.")