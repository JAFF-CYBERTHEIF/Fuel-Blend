# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:35:25 2024

@author: admin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

data = pd.read_excel('data.xlsx')

output_factors = ['BTE', 'NOx', 'Smoke', 'HC', 'CO','BSFC' ]

Fuels = ["Diesel", "B20", "B20+CaO50", "B20+CaO75", "B20+CaO100",
         "B20+MgO50", "B20+MgO75", "B20+MgO100"]
DPI = 300
i = 0
mini = 0
maxi = 4

while i <= len(Fuels):
    Fuel = Fuels[i]
    # Separate the input (Load) and output variables
    X = data[['CR (bar)', "Fuel Blends", "BP (kw)"]][data['Fuel Blends'] == Fuel]
    y = data[['BTE (%)', 'SFC (kg/kw-h)', 'NOx (ppm)', 'Smoke (%)', 'HC (ppm)', "CO (%)"]][data['Fuel Blends'] == Fuel]
  
    # Define the degree of the polynomial
    degree = 2
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    # Create the polynomial regression model
    model = LinearRegression()
    
    # Fit the model to the polynomial features
    model.fit(X_poly, y)

    # Predict the outputs for load values from 0 to 100
    load_values = np.arange(0, 101).reshape(-1, 1)
    load_poly = poly_features.transform(load_values)
    predicted_outputs = model.predict(load_poly)

    intercepts = model.intercept_
    coefficients = model.coef_


    # Print the intercepts and coefficients for each output factor
    print("Intercepts for", Fuel)
    for factor, intercept in zip(output_factors, intercepts):
        print(factor, ":", intercept)

    print(" ")
    print("Coefficients for", Fuel)
    for factor, coef in zip(output_factors, coefficients):
        print(factor, ":", coef)
        
    print(" ")
    # Create a DataFrame for the predicted outputs
    predicted_df = pd.DataFrame(predicted_outputs, columns=['BTE (%)', 'SFC (kg/kw-h)', 'NOx (ppm)', 'Smoke (%)', 'HC (ppm)', "CO (%)"])
         
    # Save the predicted outputs to a CSV file
    predicted_df.to_csv('predicted_outputs for '+ Fuel +'.csv', index=False)
   
    # Calculate R2 score for each column
    r2_scores = {}
    for column in predicted_df.columns:
        if column != 'Load (kg)':
            true_values = data[column].iloc[mini:maxi]
            predicted_values = predicted_df[column].iloc[[0,25,50,75,100]]
            
            fig, ax = plt.subplots(figsize=(8, 6), dpi= DPI)
            ax.scatter(true_values, predicted_values)
            ax.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], '--', color='gray')
            ax.set_xlabel('Real Values', fontname="Times New Roman", fontsize=14,fontweight="bold")
            ax.set_ylabel('Predicted Values', fontname="Times New Roman", fontsize=14,fontweight="bold")
            ax.set_title(column + " of " + Fuel)
            ax.set_xlim([min(true_values.min(), predicted_values.min()), max(true_values.max(), predicted_values.max())])
            ax.set_ylim([min(true_values.min(), predicted_values.min()), max(true_values.max(), predicted_values.max())])
            plt.show()

            r2_scores[column] = r2_score(true_values, predicted_values)
            

    print("R2 scores for", Fuel)
    for column, score in r2_scores.items():
        print(column, ":", score)
    print(" ")    
    i = i + 1

    
for j in len(Fuels):
    fuel = Fuels[j]
    predicted_outputs_fuel1 = pd.read_csv('predicted_outputs for '+ Fuel +'.csv')
    
    lables = ['BTE', 'NOx', 'Smoke', 'HC', 'CO','BSFC' ]
    
    i=0
    num_points = 100
    
    # Iterate over each output factor and plot the 3D graph
    for factor in output_factors:
        fig = plt.figure(figsize=(8,6), dpi= DPI)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('BP (kw)', fontname="Times New Roman", fontsize=14,fontweight="bold")
        ax.set_ylabel('CR (bar)', fontname="Times New Roman", fontsize=14,fontweight="bold")
        ax.set_zlabel(lables[i], fontname="Times New Roman", fontsize=14,fontweight="bold")
    
        # Prepare data for plotting
        x = []
        y = []
        z = []
    
        # Append data for Fuel 1
        x += j * len(predicted_outputs_fuel1)
        y += predicted_outputs_fuel1['Load (kg)'].tolist()
        z += predicted_outputs_fuel1[factor].tolist()
    
    
    
        # Create a finer mesh for triangulation
        xi = np.linspace(min(x), max(x), num_points)
        yi = np.linspace(min(y), max(y), num_points)
        Xi, Yi = np.meshgrid(xi, yi)
    
        # Triangulate the data
        triang = mtri.Triangulation(x, y)
    
        # Interpolate the z values on the finer mesh
        interpolator = mtri.LinearTriInterpolator(triang, z)
        zi = interpolator(Xi, Yi)
    
        # Plot the 3D graph with the finer mesh
        ax.plot_surface(Xi, Yi, zi, cmap='jet')
        #ax.scatter(x, y, z)
        ax.set_xticks([1, 2, 3])

        # Adjust the aspect ratio and position of the plot
        ax.dist = 12
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        
        i = i + 1
        plt.title(factor)
        plt.show()
        


  
