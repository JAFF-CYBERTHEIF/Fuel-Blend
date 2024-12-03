# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:26:56 2024

@author: admin
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.tri as mtri
import seaborn as sb
from sklearn.model_selection import learning_curve
from sklearn.inspection import partial_dependence

# Loading data
data = pd.read_excel('data.xlsx')

output_factors = ['BTE (%)', 'SFC (kg/kw-h)', 'NOx (ppm)', 'Smoke (%)', 'HC (ppm)', 'CO (%)']

emissions_columns = ['NOx (ppm)', 'HC (ppm)', 'Smoke (%)', 'CO (%)']
data_aggregated = data.groupby('Fuel Blends')[emissions_columns].sum()

# Plotting the stacked bar chart
ax = data_aggregated.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')

# Customizing the plot
plt.title("Contributions of Emissions by Fuel Blends", fontsize=16)
plt.xlabel("Fuel Blends", fontsize=12)
plt.ylabel("Emission Levels", fontsize=12)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title="Emissions", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Adjust layout to prevent label clipping

# Show plot
plt.show()

for i in output_factors:
    plt.figure(figsize=(10, 6))
    sb.boxplot(x='Fuel Blends', y=i, data=data, palette="coolwarm")
    
    # Adding labels and title
    plt.xlabel("Fuel Blends", fontsize=12)
    plt.ylabel(i, fontsize=12)
    plt.title(f"Distribution of BTE {i} Across Fuel Blends", fontsize=14)
    plt.xticks(rotation=45)  
    plt.show()

def plot_learning_curve(estimator, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
    plt.plot(train_sizes, validation_mean, label="Validation Score", color="orange")

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
    plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color="orange", alpha=0.2)

    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel(scoring, fontsize=12)
    plt.title("Learning Curve", fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    
# Getting list of unique blend names
unique_blends = data['Fuel Blends'].unique()
Input = data.drop(output_factors , axis =1)

categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse=False)

df = data[categorical_columns]
df_encoded = pd.get_dummies(df, dtype=int)

data = pd.concat([data, df_encoded], axis=1)


# Splitting the data into input (X) and output (y) variables
X = data.drop(output_factors , axis =1)
X = X.drop("Fuel Blends", axis =1)
y = data[output_factors]

# Creating polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Creating a Linear Regression model and fit it to the training data
model = LinearRegression()

plot_learning_curve(model, X, y)

model.fit(X_poly, y)

# predictions on the test data
y_pred = model.predict(X_poly)

#model's performance
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print("R-Squared :", r2)
print("Mean Squared Error :",mse )
print("Mean Absolute Error :", mae)

Prediction = pd.DataFrame({'BTE (%)': y_pred[:, 0],
                'SFC (kg/kw-h)': y_pred[:, 1],
                'NOx (ppm)': y_pred[:, 2],
                'Smoke (%)': y_pred[:, 3],
                'HC (ppm)': y_pred[:, 4],
                'CO (%)': y_pred[:, 5],})

predictions_df = pd.concat([Input, Prediction], axis = 1)

for i, output_column in enumerate(y.columns):
    plt.figure(figsize=(8, 6))
    plt.scatter(y[output_column], y_pred[:, i], alpha=0.5)
    plt.xlabel(f"Real {output_column}")
    plt.ylabel(f"Predicted {output_column}")
    plt.title(f"Real vs. Predicted {output_column}")

    # Fitting a line to the scatter plot
    z = np.polyfit(y[output_column], y_pred[:, i], 1)
    p = np.poly1d(z)
    plt.plot(y[output_column], p(y[output_column]), color='red')

    plt.show()


predictions_df.to_csv('blend_predictions_with_input.csv', index=False)

df = predictions_df


num_points = 100

for blend in unique_blends:
    # Filter data for the current blend
    blend_data = data[data['Fuel Blends'] == blend]
    for output_factor in output_factors:
        # Creating a 3D scatter plot
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        # Input and Output
        x = df['CR (bar)']
        Y = df['BP (kw)']
        z = df[output_factor]

        # Creating a finer mesh for triangulation
        xi = np.linspace(min(x), max(x), num_points)
        yi = np.linspace(min(Y), max(Y), num_points)
        Xi, Yi = np.meshgrid(xi, yi)

        triang = mtri.Triangulation(x, Y)

        # Interpolating the z values on the finer mesh
        interpolator = mtri.LinearTriInterpolator(triang, z)
        zi = interpolator(Xi, Yi)

        ax.plot_surface(Xi, Yi, zi, cmap='jet')
        #labels and title
        ax.set_xlabel('CR (bar)', fontname="Times New Roman", fontsize=14, fontweight="bold")
        ax.set_ylabel('BP (kw)', fontname="Times New Roman", fontsize=14, fontweight="bold")
        ax.set_zlabel(output_factor, fontname="Times New Roman", fontsize=14, fontweight="bold")
        ax.zaxis.labelpad = -2
        ax.set_title(f'{output_factor} vs. CR (bar) and BP (kw) for ' + blend, fontname="Times New Roman", fontsize=14, fontweight="bold")
        if output_factor in ["SFC (kg/kw-h)"]:
            ax.invert_zaxis()
        plt.show()

print("Correlation heat map")
print(df.corr(numeric_only=True))
# Plotting correlation heatmap
dataplot = sb.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True)

plt.show()
