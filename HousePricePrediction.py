import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target * 100000  # Scale prices to realistic values

# Define available dropdown options
locations = sorted(df['Latitude'].astype(str).unique())  # Latitude as location proxy
rooms = sorted(df['AveRooms'].astype(int).unique())
areas = sorted(df['AveOccup'].astype(int).unique())
ages = sorted(df['HouseAge'].astype(int).unique())
incomes = sorted(df['MedInc'].round(1).unique())

# Prepare dataset for training
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)  # Increased estimators for better accuracy
accuracy = 0  # Initialize accuracy variable

# Train model
def train_model():
    global rf_reg, accuracy, X_train, X_test, y_train, y_test
    status_label.config(text="Training model, please wait...")
    root.update()  # Update GUI
    
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    status_label.config(text=f"Model trained successfully! Accuracy: {accuracy:.2f}")

def predict_price():
    try:
        # Automatically retrain before prediction
        train_model()
        
        med_inc = float(income_dropdown.get())
        house_age = float(age_dropdown.get())
        rooms = float(room_dropdown.get())
        area = float(area_dropdown.get())
        lat = float(location_dropdown.get())
        lon = float(lon_entry.get())
        
        input_data = pd.DataFrame([[med_inc, house_age, rooms, area, lat, lon]], columns=X.columns)
        rf_pred = rf_reg.predict(input_data)[0]
        
        price_label.config(text=f"Predicted Price: ${rf_pred:,.2f}\nModel Accuracy: {accuracy:.2f}")
    except Exception as e:
        price_label.config(text=f"Error: {e}")

def visualize_data():
    plt.figure(figsize=(12, 5))
    
    # Scatter plot with regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X_test['MedInc'], y_test, alpha=0.5, label='Actual Prices')
    sorted_indices = np.argsort(X_test['MedInc'])
    plt.plot(X_test['MedInc'].values[sorted_indices], rf_reg.predict(X_test)[sorted_indices], color='red', label='Regression Line')
    plt.xlabel("Median Income")
    plt.ylabel("House Price")
    plt.legend()
    plt.title("Income vs Price Prediction")
    
    # Histogram for house price distribution
    plt.subplot(1, 2, 2)
    plt.hist(df['Price'], bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel("House Price")
    plt.ylabel("Frequency")
    plt.title("House Price Distribution")
    
    plt.tight_layout()
    plt.show()

# Create main window
root = tk.Tk()
root.title("House Price Prediction")
root.geometry("500x600")

# Dropdowns for categorical data
income_label = tk.Label(root, text="Median Income")
income_label.grid(row=0, column=0, padx=10, pady=5, sticky='w')
income_dropdown = ttk.Combobox(root, values=incomes)
income_dropdown.grid(row=0, column=1, padx=10, pady=5)
income_dropdown.current(0)

age_label = tk.Label(root, text="House Age")
age_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
age_dropdown = ttk.Combobox(root, values=ages)
age_dropdown.grid(row=1, column=1, padx=10, pady=5)
age_dropdown.current(0)

room_label = tk.Label(root, text="Rooms")
room_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
room_dropdown = ttk.Combobox(root, values=rooms)
room_dropdown.grid(row=2, column=1, padx=10, pady=5)
room_dropdown.current(0)

area_label = tk.Label(root, text="Area")
area_label.grid(row=3, column=0, padx=10, pady=5, sticky='w')
area_dropdown = ttk.Combobox(root, values=areas)
area_dropdown.grid(row=3, column=1, padx=10, pady=5)
area_dropdown.current(0)

location_label = tk.Label(root, text="Location (Latitude)")
location_label.grid(row=4, column=0, padx=10, pady=5, sticky='w')
location_dropdown = ttk.Combobox(root, values=locations)
location_dropdown.grid(row=4, column=1, padx=10, pady=5)
location_dropdown.current(0)

lon_label = tk.Label(root, text="Longitude")
lon_label.grid(row=5, column=0, padx=10, pady=5, sticky='w')
lon_entry = tk.Entry(root)
lon_entry.grid(row=5, column=1, padx=10, pady=5)

# Train Model button
train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.grid(row=6, column=0, columnspan=2, pady=10)

# Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=7, column=0, columnspan=2, pady=10)

# Visualization button
visualize_button = tk.Button(root, text="Show Analysis", command=visualize_data)
visualize_button.grid(row=8, column=0, columnspan=2, pady=10)

# Label to display prediction
price_label = tk.Label(root, text="Predicted Price: ")
price_label.grid(row=9, column=0, columnspan=2, pady=10)

# Status label for training progress
status_label = tk.Label(root, text="Click 'Train Model' before prediction")
status_label.grid(row=10, column=0, columnspan=2, pady=10)

# Run the application
root.mainloop()
