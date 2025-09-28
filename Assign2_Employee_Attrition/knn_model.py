"""
This script performs a k-Nearest Neighbors (KNN) analysis on the employee attrition dataset.

It includes the following steps:
1.  Loads and preprocesses the data (`employee_attrition.csv`).
2.  Iterates through k-values to find an optimal 'elbow' point for the KNN model.
3.  Trains a final KNN model based on the identified elbow point.
4.  Generates several plots for analysis:
    - Accuracy vs. K Value (with the elbow point marked).
    - A confusion matrix for the best model.
    - A 2D decision boundary plot.
5.  Displays these metrics and plots in a single tkinter GUI window.

Dependencies: pandas, scikit-learn, matplotlib, numpy, seaborn, Pillow
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

# --- Machine Learning Logic ---
def run_ml_pipeline():
    """Runs the core machine learning pipeline and generates analysis plots."""
    # --- 1. Load and Preprocess Data ---
    df = pd.read_csv("employee_attrition.csv")
    
    # Separate features (X) and target (y)
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # WARNING: This simplified approach encodes the whole dataset before splitting,
    # which can lead to data leakage. For a production model, preprocessing
    # should be done after splitting or within a scikit-learn Pipeline.
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[['JobRole']]))
    X_encoded.columns = encoder.get_feature_names_out(['JobRole'])
    X = X.drop('JobRole', axis=1)
    X = pd.concat([X, X_encoded], axis=1)

    # --- 2. Split Data for Training and Testing ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- 3. Find Optimal K using the Elbow Method ---
    # Define the range of k to test. It cannot be larger than the number of training samples.
    k_range = np.array(range(1, min(8, len(X_train))))
    # Calculate accuracy for each k
    accuracies = np.array([accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test)) for k in k_range])

    # Find the elbow point using the 'distance from line' method.
    # This identifies the point of maximum curvature on the plot.
    norm_k = (k_range - k_range.min()) / (k_range.max() - k_range.min())
    norm_acc = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min() if (accuracies.max() - accuracies.min()) != 0 else 1)
    p1 = np.array([norm_k[0], norm_acc[0]]) # Start point
    p2 = np.array([norm_k[-1], norm_acc[-1]]) # End point
    # Calculate the perpendicular distance of each point from the line connecting start and end
    dists = [np.abs(np.cross(p2-p1, p-p1))/np.linalg.norm(p2-p1) for p in zip(norm_k, norm_acc)]
    elbow_index = np.argmax(dists)
    best_k = k_range[elbow_index]
    best_accuracy = accuracies[elbow_index]

    # --- 4. Generate Analysis Plots ---
    # Plot 1: Accuracy vs. K, highlighting the identified elbow point
    plt.figure(figsize=(5, 4))
    plt.plot(k_range, accuracies, marker='o', label='Accuracy')
    plt.plot(best_k, best_accuracy, 'ro', markersize=10, label=f'Elbow (K={best_k})')
    plt.title('Accuracy vs. K Value'); plt.xlabel('K'); plt.ylabel('Accuracy')
    plt.xticks(k_range); plt.grid(True); plt.legend(); plt.tight_layout()
    accuracy_k_plot_filename = 'accuracy_vs_k.png'
    plt.savefig(accuracy_k_plot_filename, dpi=70)
    plt.close()

    # Train the final model using the determined best_k
    best_knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
    best_y_pred = best_knn.predict(X_test)
    final_accuracy = f"Elbow K = {best_k} with Accuracy: {best_accuracy:.2f}"
    all_labels = [0, 1]
    class_report = f"Classification Report (K={best_k}):\n{classification_report(y_test, best_y_pred, labels=all_labels, zero_division=0)}"

    # Plot 2: Confusion Matrix for the best model
    cm = confusion_matrix(y_test, best_y_pred, labels=all_labels)
    plt.figure(figsize=(5, 4))
    tn, fp, fn, tp = cm.ravel()
    annot = [[f'{tn}\n(TN)', f'{fp}\n(FP)'], [f'{fn}\n(FN)', f'{tp}\n(TP)']]
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=['No Attrition', 'Attrition'], yticklabels=['No Attrition', 'Attrition'])
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Confusion Matrix (K={best_k})'); plt.tight_layout()
    cm_plot_filename = 'confusion_matrix.png'
    plt.savefig(cm_plot_filename, dpi=70)
    plt.close()

    # Plot 3: Decision Boundary for the best model (using two features for visualization)
    X_vis = df[['MonthlyIncome', 'YearsAtCompany']]
    y_vis = df['Attrition']
    knn_vis = KNeighborsClassifier(n_neighbors=best_k).fit(X_vis, y_vis)
    x_min, x_max = X_vis.iloc[:, 0].min() - 1000, X_vis.iloc[:, 0].max() + 1000
    y_min, y_max = X_vis.iloc[:, 1].min() - 1, X_vis.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 100), np.arange(y_min, y_max, 1))
    Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()]); Z = Z.reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis, s=20, edgecolor='k')
    plt.title(f'Decision Boundary (K={best_k})'); plt.xlabel('Monthly Income'); plt.ylabel('Years At Company'); plt.tight_layout()
    boundary_plot_filename = 'decision_boundary.png'
    plt.savefig(boundary_plot_filename, dpi=70)
    plt.close()

    # Return all metrics and plot filenames for the GUI
    return final_accuracy, class_report, cm_plot_filename, accuracy_k_plot_filename, boundary_plot_filename

# --- GUI Logic ---
def create_gui(accuracy, class_report, cm_plot_fn, acc_k_plot_fn, boundary_plot_fn):
    """Creates and displays the tkinter GUI window with all analysis results."""
    root = tk.Tk()
    root.title("KNN Model Analysis")

    # --- Top Frame for Text Metrics ---
    metrics_frame = tk.Frame(root, padx=10, pady=5)
    metrics_frame.pack(fill="x", side="top")
    tk.Label(metrics_frame, text="Performance Metrics", font=("Helvetica", 11, "bold")).pack(anchor="w")
    tk.Label(metrics_frame, text=accuracy, font=("Helvetica", 10)).pack(anchor="w", pady=(5,0))
    tk.Label(metrics_frame, text=class_report, font=font.Font(family="Courier", size=9), justify="left").pack(anchor="w")

    # --- Bottom Frame for 2x2 Plot Grid ---
    plots_frame = tk.Frame(root, padx=10, pady=5)
    plots_frame.pack(fill="both", expand=True)

    # Load images. Storing them in a dictionary is crucial to prevent Python's
    # garbage collector from deleting them before they are displayed.
    img_refs = {
        'k': ImageTk.PhotoImage(Image.open(acc_k_plot_fn)),
        'cm': ImageTk.PhotoImage(Image.open(cm_plot_fn)),
        'bound': ImageTk.PhotoImage(Image.open(boundary_plot_fn))
    }

    # Arrange plots in a grid. The decision boundary spans two columns on the bottom row.
    tk.Label(plots_frame, image=img_refs['k']).grid(row=0, column=0, padx=5, pady=5)
    tk.Label(plots_frame, image=img_refs['cm']).grid(row=0, column=1, padx=5, pady=5)
    tk.Label(plots_frame, image=img_refs['bound']).grid(row=1, column=0, columnspan=2, padx=5, pady=5)

    # Exit Button
    tk.Button(root, text="Exit", command=root.destroy, width=10).pack(pady=10, side="bottom")

    # Start the GUI event loop
    root.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    # This block runs when the script is executed directly.
    # It calls the pipeline function to get the results...
    acc, cr, cm_plot, acc_k_plot, boundary_plot = run_ml_pipeline()
    # ...and then passes them to the GUI function to be displayed.
    create_gui(acc, cr, cm_plot, acc_k_plot, boundary_plot)
