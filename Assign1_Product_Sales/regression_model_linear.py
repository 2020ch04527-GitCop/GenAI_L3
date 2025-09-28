"""
This script builds and evaluates a multivariate linear regression model and provides
a GUI for real-time sales prediction and performance visualization.

Features:
- Displays model performance metrics (R-squared, MSE).
- Shows two plots:
  1. Model performance on the unseen test set.
  2. Overall model fit across all data points.
- Provides input fields for real-time sales prediction.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tkinter as tk
from tkinter import font as tkFont
import matplotlib.pyplot as plt
import os
import sys

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow library not found. Please install it using: pip install pillow")
    sys.exit(1)

def run_regression_and_generate_plots():
    """
    Trains the model, evaluates it, generates two plots, and returns GUI components.

    Returns:
        tuple: Contains metrics, path to test plot, path to full data plot, the model, and the scaler.
    """
    data = pd.read_csv("sales.csv")
    scaler = StandardScaler()
    features_to_scale = ['TV Budget ($)', 'Radio Budget ($)', 'Newspaper Budget ($)']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    X = data[features_to_scale]
    y = data['Sales (units)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Evaluation on Test Set ---
    y_pred_test = model.predict(X_test)
    metrics = {
        "R-squared (Test Set)": f"{r2_score(y_test, y_pred_test):.4f}",
        "MSE (Test Set)": f"{mean_squared_error(y_test, y_pred_test):.4f}"
    }

    # Add feature coefficients to metrics
    feature_coefficients = {}
    for i, feature in enumerate(features_to_scale):
        feature_coefficients[feature] = model.coef_[i]
    metrics["Feature Coefficients"] = feature_coefficients

    # --- Generate Plots ---
    # Plot 1: Performance on Test Set
    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolors='k', label='Test Data')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title('Performance on Test Set (2 points)')
    plt.xlabel('Actual Sales'); plt.ylabel('Predicted Sales')
    plt.legend(); plt.grid(True)
    test_plot_path = "test_set_performance.png"
    plt.savefig(test_plot_path, dpi=80)
    plt.close()

    # Plot 2: Fit on Full Dataset
    y_pred_full = model.predict(X)
    plt.figure(figsize=(5, 4))
    plt.scatter(y, y_pred_full, alpha=0.7, edgecolors='b', label='All Data')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title('Overall Model Fit (All 10 points)')
    plt.xlabel('Actual Sales'); plt.ylabel('Predicted Sales')
    plt.legend(); plt.grid(True)
    full_plot_path = "full_dataset_fit.png"
    plt.savefig(full_plot_path, dpi=80)
    plt.close()

    return metrics, test_plot_path, full_plot_path, model, scaler

def create_gui():
    metrics, test_plot_path, full_plot_path, model, scaler = run_regression_and_generate_plots()

    root = tk.Tk()
    root.title("Sales Prediction Dashboard")

    # --- Main Frames ---
    plots_frame = tk.Frame(root)
    plots_frame.pack(pady=10, padx=10, fill="x")
    prediction_frame = tk.Frame(root, padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
    prediction_frame.pack(pady=10, padx=10, fill="x")

    # --- Plots Section ---
    left_plot_frame = tk.Frame(plots_frame)
    left_plot_frame.pack(side="left", padx=(0, 10))
    right_plot_frame = tk.Frame(plots_frame)
    right_plot_frame.pack(side="right")

    img_test = Image.open(test_plot_path)
    photo_test = ImageTk.PhotoImage(img_test)
    img_label_test = tk.Label(left_plot_frame, image=photo_test)
    img_label_test.pack()

    img_full = Image.open(full_plot_path)
    photo_full = ImageTk.PhotoImage(img_full)
    img_label_full = tk.Label(right_plot_frame, image=photo_full)
    img_label_full.pack()

    # --- Metrics & Prediction ---
    bold_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
    tk.Label(prediction_frame, text="Performance Metrics", font=bold_font).grid(row=0, column=0, columnspan=2, pady=5, sticky="w")
    row_idx = 1
    for key, value in metrics.items():
        if key == "Feature Coefficients":
            tk.Label(prediction_frame, text="Feature Coefficients:", font=bold_font).grid(row=row_idx, column=0, columnspan=2, sticky="w")
            row_idx += 1
            for feature, coef in value.items():
                tk.Label(prediction_frame, text=f"  {feature}: {coef:.4f}").grid(row=row_idx, column=0, columnspan=2, sticky="w")
                row_idx += 1
        else:
            tk.Label(prediction_frame, text=f"{key}: {value}").grid(row=row_idx, column=0, columnspan=2, sticky="w")
            row_idx += 1

    tk.Label(prediction_frame, text="Predict Sales", font=bold_font).grid(row=row_idx, column=0, columnspan=3, pady=(15, 5), sticky="w")
    row_idx += 1
    
    tk.Label(prediction_frame, text="TV Budget ($):").grid(row=row_idx, column=0, sticky="w", pady=2)
    entry_tv = tk.Entry(prediction_frame); entry_tv.grid(row=row_idx, column=1, pady=2)
    row_idx += 1

    tk.Label(prediction_frame, text="Radio Budget ($):").grid(row=row_idx, column=0, sticky="w", pady=2)
    entry_radio = tk.Entry(prediction_frame); entry_radio.grid(row=row_idx, column=1, pady=2)
    row_idx += 1

    tk.Label(prediction_frame, text="Newspaper Budget ($):").grid(row=row_idx, column=0, sticky="w", pady=2)
    entry_newspaper = tk.Entry(prediction_frame); entry_newspaper.grid(row=row_idx, column=1, pady=2)
    row_idx += 1

    result_label = tk.Label(prediction_frame, text="Predicted Sales: -", font=bold_font, fg="blue")
    result_label.grid(row=row_idx+1, column=0, columnspan=2, pady=10)

    def predict_sales():
        try:
            tv_budget_str = entry_tv.get()
            tv_budget = float(tv_budget_str) if tv_budget_str else 0.0
        except ValueError:
            result_label.config(text="Error: Invalid TV Budget. Please enter a number.", fg="red")
            entry_tv.delete(0, tk.END)
            return

        try:
            radio_budget_str = entry_radio.get()
            radio_budget = float(radio_budget_str) if radio_budget_str else 0.0
        except ValueError:
            result_label.config(text="Error: Invalid Radio Budget. Please enter a number.", fg="red")
            entry_radio.delete(0, tk.END)
            return

        try:
            newspaper_budget_str = entry_newspaper.get()
            newspaper_budget = float(newspaper_budget_str) if newspaper_budget_str else 0.0
        except ValueError:
            result_label.config(text="Error: Invalid Newspaper Budget. Please enter a number.", fg="red")
            entry_newspaper.delete(0, tk.END)
            return

        input_data = pd.DataFrame({
            'TV Budget ($)': [tv_budget],
            'Radio Budget ($)': [radio_budget],
            'Newspaper Budget ($)': [newspaper_budget]
        })
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        result_label.config(text=f"Predicted Sales: {prediction[0]:.2f} units", fg="blue")

    predict_button = tk.Button(prediction_frame, text="Predict", command=predict_sales)
    predict_button.grid(row=row_idx, column=0, columnspan=2, pady=10)

    # --- Exit and Cleanup ---
    def on_closing():
        if os.path.exists(test_plot_path): os.remove(test_plot_path)
        if os.path.exists(full_plot_path): os.remove(full_plot_path)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    exit_button = tk.Button(root, text="Exit", command=on_closing)
    exit_button.pack(pady=10)

    # Keep a reference to the images to prevent garbage collection
    img_label_test.image = photo_test
    img_label_full.image = photo_full

    root.mainloop()

if __name__ == '__main__':
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib not found. Please install it using: pip install matplotlib")
        sys.exit(1)
    create_gui()