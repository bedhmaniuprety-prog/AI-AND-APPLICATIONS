import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



path = '/content/student_exam_scores.csv'
print('Loading from:', path)

df = pd.read_csv(path)
print('Shape:', df.shape)
print('\nColumns:')
print(list(df.columns))

display(df.head())

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print('\nNormalized columns:')
print(list(df.columns))


# Preprocessing & feature engineering (in-notebook)
expected_numeric = ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', 'exam_score']
missing = [c for c in expected_numeric if c not in df.columns]
if missing:
    raise KeyError(f"Missing expected numeric columns: {missing}. Please check the CSV column names.")
else:
    for c in expected_numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    print("Missing values per expected numeric column:")
    print(df[expected_numeric].isna().sum())

    # Drop rows with missing essential numerics for simplicity
    before = df.shape[0]
    df = df.dropna(subset=expected_numeric).reset_index(drop=True)
    after = df.shape[0]
    print(f"Dropped {before-after} rows with missing values. New shape: {df.shape}")

    # Feature engineering
    df['study_sleep_ratio'] = df.apply(lambda r: r['hours_studied'] / r['sleep_hours'] if r['sleep_hours'] and pd.notna(r['sleep_hours']) else np.nan, axis=1)
    scaler = StandardScaler()
    z_cols = ['hours_studied', 'attendance_percent', 'previous_scores']
    zdf = pd.DataFrame(scaler.fit_transform(df[z_cols]), columns=[f"{c}_z" for c in z_cols])
    df = pd.concat([df.reset_index(drop=True), zdf], axis=1)
    df['engagement_score'] = 0.5 * df['hours_studied_z'] + 0.3 * df['attendance_percent_z'] + 0.2 * df['previous_scores_z']
    threshold = df['exam_score'].quantile(0.90)
    df['high_performer'] = (df['exam_score'] >= threshold).astype(int)
    def sleep_cat(hours):
        if pd.isna(hours): return np.nan
        if hours < 6: return 'short'
        if hours <= 8: return 'recommended'
        return 'long'
    df['sleep_category'] = df['sleep_hours'].apply(sleep_cat)
    df['sleep_category'] = pd.Categorical(df['sleep_category'], categories=['short','recommended','long'], ordered=True)

print('Preprocessing complete. Processed shape:', df.shape)

# show a preview
from IPython.display import display
proc_cols = ['student_id','hours_studied','sleep_hours','sleep_category','study_sleep_ratio','attendance_percent','previous_scores','exam_score','engagement_score','high_performer']
proc_cols = [c for c in proc_cols if c in df.columns]
display(df[proc_cols].head(12))


import matplotlib.pyplot as plt, seaborn as sns
sns.set(style='whitegrid', rc={'figure.figsize':(8,4)})

print('Correlation (hours_studied vs exam_score):', df['hours_studied'].corr(df['exam_score']).round(3))

plt.figure()
sns.regplot(x='hours_studied', y='exam_score', data=df, scatter_kws={'alpha':0.6})
plt.title('Hours Studied vs Exam Score (with regression)')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

# Print grouped statistics
grouped = df.groupby(pd.cut(df['hours_studied'], bins=5))['exam_score'].agg(['count','mean','median']).reset_index()
print('\nAverage exam score by study-hours bin:')
display(grouped)


print('Correlation (sleep_hours vs exam_score):', df['sleep_hours'].corr(df['exam_score']).round(3))

plt.figure()
sns.boxplot(x='sleep_category', y='exam_score', data=df)
plt.title('Exam Score by Sleep Category')
plt.xlabel('Sleep Category')
plt.ylabel('Exam Score')
plt.show()

# Show mean scores by sleep category
print('\nMean exam score by sleep category:')
display(df.groupby('sleep_category')['exam_score'].agg(['count','mean','median']).reset_index())


print('Correlation matrix (attendance, engagement, exam score):')
display(df[['attendance_percent','engagement_score','exam_score']].corr().round(3))

plt.figure()
sns.scatterplot(x='attendance_percent', y='exam_score', hue='high_performer', data=df, alpha=0.7)
plt.title('Attendance vs Exam Score (colored by high performers)')
plt.xlabel('Attendance %')
plt.ylabel('Exam Score')
plt.show()

# Attendance bins
bins = [0,60,75,90,100]
labels = ['<60','60-75','75-90','90-100']
df['att_bin'] = pd.cut(df['attendance_percent'], bins=bins, labels=labels, include_lowest=True)
display(df.groupby('att_bin')['exam_score'].agg(['count','mean','median']).reset_index())


top = df[df['high_performer']==1]
print('Top performers count:', top.shape[0])
display(top[['hours_studied','sleep_hours','attendance_percent','previous_scores','engagement_score']].describe().round(2))

# Plot distribution comparison: engagement_score
plt.figure()
sns.kdeplot(df['engagement_score'], label='All', fill=False)
sns.kdeplot(top['engagement_score'], label='Top 10%', fill=False)
plt.title('Engagement Score Distribution: All vs Top 10%')
plt.legend()
plt.show()

# Show sample top students
print('\nSample top students:')
display(top.sort_values('exam_score', ascending=False).head(8))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import sklearn

# optional: check sklearn version if you want to know why 'squared' isn't available
print("scikit-learn version:", sklearn.__version__)

# features / target (keep your df in scope)
feature_cols = ['engagement_score','hours_studied','sleep_hours']
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].fillna(0).values
y = df['exam_score'].values

# split / fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# predictions & metrics
y_hat = model.predict(X_test)
r2 = r2_score(y_test, y_hat)
mse = mean_squared_error(y_test, y_hat)   # MSE (no squared kwarg used)
rmse = np.sqrt(mse)                       # RMSE from MSE

print(f"Model R^2 (test): {r2:.3f}, RMSE: {rmse:.3f}")
print('\nCoefficients:')
for name, coef in zip(feature_cols, model.coef_):
    print(f"  {name}: {coef:.4f}")

# Residual plot
resid = y_test - y_hat
plt.figure()
plt.scatter(y_hat, resid, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residuals vs Predicted')
plt.tight_layout()
plt.show()

# Show sample predictions
comp = pd.DataFrame({'actual': y_test, 'predicted': np.round(y_hat,3)})
print('\nSample predictions:')
display(comp.head(8))

