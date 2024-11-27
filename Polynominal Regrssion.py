
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:26:56 2024

@author: admin
"""


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.tri as mtri

input_columns = ['CR (bar)', 'Fuel Blends', 'BP (kw)']
output_columns = ['BTE (%)', 'SFC (kg/kw-h)', 'NOx (ppm)', 'Smoke (%)', 'HC (ppm)', 'CO (%)']

# Load your data into a pandas DataFrame
data = pd.read_excel('data.xlsx')

# Get a list of unique blend names
unique_blends = data['Fuel Blends'].unique()

regression_results = {}
predictions_df = pd.DataFrame()

# Iterate through each blend and perform linear regression
for blend in unique_blends:
    # Filter data for the current blend
    blend_data = data[data['Fuel Blends'] == blend]
    
    # Split the data into input (X) and output (y) variables
    X = blend_data[['CR (bar)', 'BP (kw)']]
    y = blend_data[['BTE (%)', 'SFC (kg/kw-h)', 'NOx (ppm)', 'Smoke (%)', 'HC (ppm)', 'CO (%)']]
   
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=3)
    X_poly = poly_features.fit_transform(X)
    
    # Create a Linear Regression model and fit it to the training data
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions on the test data
    y_pred = model.predict(X)
    
    # Evaluate the model's performance
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
     # Add the input values and predictions to the DataFrame
    predictions_df = predictions_df.append(pd.DataFrame({'CR (bar)': X['CR (bar)'],
                                                         'BP (kw)': X['BP (kw)'],
                                                         'BTE (%)': y_pred[:, 0],
                                                         'SFC (kg/kw-h)': y_pred[:, 1],
                                                         'NOx (ppm)': y_pred[:, 2],
                                                         'Smoke (%)': y_pred[:, 3],
                                                         'HC (ppm)': y_pred[:, 4],
                                                         'CO (%)': y_pred[:, 5],
                                                         'Fuel Blends': blend}))
    # Store the regression results in the dictionary
    regression_results[blend] = {
        'MSE': mse,
        'R-squared': r2
    }

    for i, output_column in enumerate(y.columns):
        plt.figure(figsize=(8, 6))
        plt.scatter(y[output_column], y_pred[:, i], alpha=0.5)
        plt.xlabel(f"Real {output_column}")
        plt.ylabel(f"Predicted {output_column}")
        plt.title(f"Blend: {blend} - Real vs. Predicted {output_column}")
        
        # Fit a line to the scatter plot
        z = np.polyfit(y[output_column], y_pred[:, i], 1)
        p = np.poly1d(z)
        plt.plot(y[output_column], p(y[output_column]), color='red')
        
        plt.show()
predictions_df.to_csv('blend_predictions_with_input.csv', index=False)
# Print the regression results for each blend
for blend, results in regression_results.items():
    print(f"Blend: {blend}")
    print(f"Mean Squared Error: {results['MSE']}")
    print(f"R-squared: {results['R-squared']}")
    print()


df = predictions_df
    
output_factors = ['BTE (%)', 'SFC (kg/kw-h)', 'NOx (ppm)', 'Smoke (%)', 'HC (ppm)', 'CO (%)']
   
num_points = 100
# Iterate through each output factor
for blend in unique_blends:
    # Filter data for the current blend
    blend_data = data[data['Fuel Blends'] == blend]
    for output_factor in output_factors:
        # Create a 3D scatter plot for the current output factor
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract input factors and the current output factor
        x = df['CR (bar)']
        Y = df['BP (kw)']
        z = df[output_factor]
    
        # Create a finer mesh for triangulation
        xi = np.linspace(min(x), max(x), num_points)
        yi = np.linspace(min(Y), max(Y), num_points)
        Xi, Yi = np.meshgrid(xi, yi)

    
        triang = mtri.Triangulation(x, Y)

        # Interpolate the z values on the finer mesh
        interpolator = mtri.LinearTriInterpolator(triang, z)
        zi = interpolator(Xi, Yi)
        
        # Plot the 3D scatter plot
        
        ax.plot_surface(Xi, Yi, zi, cmap='jet')
        # Set labels and title
        ax.set_xlabel('CR (bar)', fontname="Times New Roman", fontsize=14, fontweight="bold")
        ax.set_ylabel('BP (kw)', fontname="Times New Roman", fontsize=14, fontweight="bold")
        ax.set_zlabel(output_factor, fontname="Times New Roman", fontsize=14, fontweight="bold")
        ax.set_title(f'{output_factor} vs. CR (bar) and BP (kw) for ' + blend, fontname="Times New Roman", fontsize=14, fontweight="bold")
        if output_factor in ["SFC (kg/kw-h)"]:
            ax.invert_zaxis()
        # Show the plot
        plt.show()
