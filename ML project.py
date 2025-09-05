
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("ai_job_trends_dataset.csv")
df

print("Dataset preview: ")
print(df.head())

print(df.dtypes)

number_column = df.select_dtypes(include=['int64','float64']).columns.tolist()
Categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print("\n numeric columns: ")
print(number_column)

print("\n categorical columns: ")
print(Categorical_columns)

print("Missing values before cleaning:")
print(df.isnull().sum())

df['Median Salary (USD)'] = pd.to_numeric(df['Median Salary (USD)'], errors='coerce')
print(df['Median Salary (USD)'])

print(df.isnull().sum())

import pandas as pd
from sklearn.preprocessing import StandardScaler
df['Median Salary (USD)'] = pd.to_numeric(df['Median Salary (USD)'], errors='coerce')
df['Median Salary (USD)'].fillna(df['Median Salary (USD)'].mean(), inplace=True)
df.dropna(inplace=True)

numeric_columns = ['Median Salary (USD)', 'Experience Required (Years)', 'Job Openings (2024)', 'Projected Openings (2030)', 'Remote Work Ratio (%)', 'Automation Risk (%)', 'Gender Diversity (%)']
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("Rescaled Data:")
print(df[numeric_columns].head())

import pandas as pd

df = pd.read_csv("ai_job_trends_dataset.csv")

X = df[["Experience Required (Years)"]]
y = df["Median Salary (USD)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

slope = model.coef_[0]
intercept = model.intercept_
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n  Simple Linear Regression Results  ")
print(f"Equation: Median Salary = {intercept:.2f} + {slope:.2f} × (Experience in Years)")
print(f"Slope (Coefficient): {slope}")
print(f"Intercept: {intercept}")
print(f"R² Score: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print("\n  Actual vs Predicted (first 10)  ")
print(comparison_df.head(10))

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Experience Required (Years)")
plt.ylabel("Median Salary (USD)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("ai_job_trends_dataset.csv")

X = df[["Job Openings (2024)"]].values
y = df["Projected Openings (2030)"].values

lin_model = LinearRegression()
lin_model.fit(X, y)
y_pred_lin = lin_model.predict(X)


# Check X and y values
print("\nFirst 5 X values:\n", X[:5])
print("\nFirst 5 y values:\n", y[:5])

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

r2_poly = r2_score(y, y_pred_poly)
mse_poly = mean_squared_error(y, y_pred_poly)
mae_poly = mean_absolute_error(y, y_pred_poly)

print("   Polynomial Regression (Degree=3)  ")
print("Coefficients:", poly_model.coef_)
print("Intercept:", poly_model.intercept_)
print(f"R² Score: {r2_poly}")
print(f"Mean Squared Error: {mse_poly}")
print(f"Mean Absolute Error: {mae_poly}")

X_sorted = np.sort(X, axis=0)
plt.figure(figsize=(8, 5))

plt.scatter(X, y, color='blue', alpha=0.4, label='Actual Data')

plt.plot(X_sorted, lin_model.predict(X_sorted), color='green', linewidth=2, label='Linear Regression')

plt.plot(X_sorted, poly_model.predict(poly.transform(X_sorted)), color='red', linewidth=2, label='Polynomial Regression')

plt.xlabel("Job Openings (2024)")
plt.ylabel("Projected Openings (2030)")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("ai_job_trends_dataset.csv")

print(df.dtypes)

print("\n  First 5 Rows ")
print(df.head())

number_column = df.select_dtypes(include=['int64','float64']).columns.tolist()
Categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

print("\n categorical columns: ")
print(Categorical_columns)

print("Missing values before cleaning:")
print(df.isnull().sum())

features = ["Experience Required (Years)", "Job Openings (2024)",
            "Remote Work Ratio (%)", "Automation Risk (%)", "Gender Diversity (%)"]
X = df[features]
y = df["Median Salary (USD)"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\n      Multiple Linear Regression Results     ")
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature}: {coef}")

print(f"\nR² Score: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print("\n       Actual vs Predicted (first 10)     ")
print(comparison_df.head(10))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("ai_job_trends_dataset.csv")

X = df[["Job Openings (2024)"]].values
y = df["Projected Openings (2030)"].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_scaled, y_scaled)

y_pred_scaled = svr.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

print("   SVR Results   ")
print(f"R² Score: {r2_score(y, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred)}")

X_sorted = np.sort(X, axis=0)
X_sorted_scaled = scaler_X.transform(X_sorted)
y_sorted_pred_scaled = svr.predict(X_sorted_scaled)
y_sorted_pred = scaler_y.inverse_transform(y_sorted_pred_scaled.reshape(-1, 1)).ravel()

plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_sorted, y_sorted_pred, color='red', linewidth=2, label='SVR Prediction')
plt.xlabel("Job Openings (2024)")
plt.ylabel("Projected Openings (2030)")
plt.title("Support Vector Regression (SVR)")
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("ai_job_trends_dataset.csv")

X = df[["Job Openings (2024)"]].values
y = df["Projected Openings (2030)"].values

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

knn = KNeighborsRegressor(n_neighbors=10, weights='distance', metric='euclidean')
knn.fit(X_scaled, y)

X_sorted = np.sort(X, axis=0)
X_sorted_scaled = scaler_X.transform(X_sorted)
y_pred_sorted = knn.predict(X_sorted_scaled)
y_pred_all = knn.predict(X_scaled)

print("=== KNN Regression (Single Feature) ===")
print(f"R² Score: {r2_score(y, y_pred_all)}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred_all)}")
print(f"Mean Absolute Error: {mean_absolute_error(y, y_pred_all)}")

plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='KNN Prediction')
plt.xlabel("Job Openings (2024)")
plt.ylabel("Projected Openings (2030)")
plt.title("KNN Regression (n_neighbors=10, weights='distance')")
plt.legend()
plt.show()

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("ai_job_trends_dataset.csv")

features = ["Experience Required (Years)", "Job Openings (2024)",
            "Remote Work Ratio (%)", "Automation Risk (%)", "Gender Diversity (%)"]
X = df[features].values
y = df["Median Salary (USD)"].values

results = []

X1 = df[["Experience Required (Years)"]]
Xtr,Xte,ytr,yte = train_test_split(X1,y,test_size=0.2,random_state=42)
m=LinearRegression().fit(Xtr,ytr); yp=m.predict(Xte)
results.append(["Simple Linear", round(r2_score(yte,yp),4)])

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
m=LinearRegression().fit(Xtr,ytr); yp=m.predict(Xte)
results.append(["Multiple Linear", round(r2_score(yte,yp),4)])

Xp = PolynomialFeatures(degree=3).fit_transform(X1)
m=LinearRegression().fit(Xp,y); yp=m.predict(Xp)
results.append(["Polynomial (deg=3)", round(r2_score(y,yp),4)])

scX,scy=StandardScaler(),StandardScaler()
Xs,ys=scX.fit_transform(X),scy.fit_transform(y.reshape(-1,1)).ravel()
m=SVR(kernel="rbf",C=100,gamma=0.1,epsilon=0.1).fit(Xs,ys)
yp=scy.inverse_transform(m.predict(Xs).reshape(-1,1)).ravel()
results.append(["SVR (RBF)", round(r2_score(y,yp),4)])

Xs=StandardScaler().fit_transform(X)
m=KNeighborsRegressor(n_neighbors=10,weights="distance").fit(Xs,y); yp=m.predict(Xs)
results.append(["KNN (k=10)", round(r2_score(y,yp),4)])

res=pd.DataFrame(results,columns=["Model","R²"])
res["Accuracy %"]=res["R²"]*100
print("\n Final Model Accuracy Comparison")
print(res)

plt.figure(figsize=(8,5))
plt.bar(res["Model"], res["Accuracy %"], color="skyblue")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy % (R² * 100)")
plt.xticks(rotation=30)
plt.ylim(0, 100)
for i, v in enumerate(res["Accuracy %"]):
    plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontsize=9)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ai_job_trends_dataset.csv")

# 1. Distribution of Median Salary

plt.figure(figsize=(8,5))
sns.histplot(df["Median Salary (USD)"], kde=True, bins=30, color="blue")
plt.title("Distribution of Median Salary (USD)")
plt.xlabel("Median Salary (USD)")
plt.ylabel("Count")
plt.show()

# 2. Correlation Heatmap

plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=['float64','int64']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# 3. Top 10 Industries by Salary

plt.figure(figsize=(12,6))
top_industries = df.groupby("Industry")["Median Salary (USD)"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_industries.index, y=top_industries.values, palette="viridis")
plt.title("Top 10 Industries by Average Median Salary")
plt.xticks(rotation=45)
plt.ylabel("Average Median Salary (USD)")
plt.show()

# 4. Experience vs Salary

plt.figure(figsize=(8,5))
sns.scatterplot(x="Experience Required (Years)", y="Median Salary (USD)", data=df, alpha=0.6)
sns.regplot(x="Experience Required (Years)", y="Median Salary (USD)", data=df, scatter=False, color="red")
plt.title("Experience vs Median Salary")
plt.show()

# 5. AI Impact Level vs Automation Risk

plt.figure(figsize=(8,5))
sns.boxplot(x="AI Impact Level", y="Automation Risk (%)", data=df, palette="Set2")
plt.title("Automation Risk across AI Impact Levels")
plt.show()

# Define features (same order as training)
features = [
    "Experience Required (Years)",
    "Job Openings (2024)",
    "Remote Work Ratio (%)",
    "Automation Risk (%)",
    "Gender Diversity (%)"
]

# Create new test sample (must include ALL features used for training)
test_sample = pd.DataFrame([{
    "Experience Required (Years)": 5,
    "Job Openings (2024)": 1200,
    "Remote Work Ratio (%)": 70,
    "Automation Risk (%)": 15,
    "Gender Diversity (%)": 45
}])

# Scale numeric feature columns
test_sample_scaled = scaler.transform(test_sample[features])

# Predict salary
predicted_salary = model.predict(test_sample_scaled)[0]

# Convert to Yes/No (using threshold)
threshold = 100000  # change this if you want another cutoff
prediction = "Yes" if predicted_salary >= threshold else "No"

print("Predicted Median Salary (USD):", predicted_salary)
print("Prediction (Above Threshold?):", prediction)