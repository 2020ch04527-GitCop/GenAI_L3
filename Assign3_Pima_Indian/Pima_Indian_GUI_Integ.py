"""
This script creates a GUI application for predicting diabetes using a trained
neural network. The application has two main sections:
- An input panel on the left for users to enter patient data.
- An output panel on the right that displays the prediction result and model 
  performance charts.

The script first trains a neural network on the Pima Indians Diabetes dataset using
Dropout for regularization and EarlyStopping to prevent overfitting, and then 
launches the tkinter-based GUI.
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox

# --- Data Loading, Preprocessing, and Model Training ---

# Load the dataset from a URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, names=column_names)

# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model and GUI Globals ---
model = None
history = None
canvas = None
fig = None
axs = None

# --- GUI Functions ---

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_colors, layer_labels):
    """
    Draws a neural network diagram on a matplotlib axes.
    """
    ax.clear()
    ax.axis('off')
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    for n, (layer_size, layer_color, layer_label) in enumerate(zip(layer_sizes, layer_colors, layer_labels)):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        ax.text(n*h_spacing + left, top + 0.05, layer_label, ha='center', va='center', fontsize=12)
        for m in range(layer_size):
            x = n*h_spacing + left
            y = layer_top - m*v_spacing
            circle = plt.Circle((x,y), v_spacing/4., color=layer_color, ec='k', zorder=4)
            ax.add_artist(circle)

    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                x1 = n*h_spacing + left
                y1 = layer_top_a - m*v_spacing
                x2 = (n + 1)*h_spacing + left
                y2 = layer_top_b - o*v_spacing
                line = plt.Line2D([x1,x2], [y1,y2], c='gray', alpha=0.5)
                ax.add_artist(line)
    ax.set_title('Neural Network Architecture')


def retrain_model():
    """
    Builds and trains the neural network based on the specified architecture.
    """
    global model, history
    try:
        neuron_counts_str = neurons_entry.get()
        neuron_counts = [int(x.strip()) for x in neuron_counts_str.split(',')]

        layers = [Dense(neuron_counts[0], input_shape=(8,), activation='relu', name='Hidden_Layer_1')]
        if len(neuron_counts) > 1:
            for i, neurons in enumerate(neuron_counts[1:]):
                layers.append(Dropout(0.2))
                layers.append(Dense(neurons, activation='relu', name=f'Hidden_Layer_{i+2}'))
        
        layers.append(Dropout(0.2))
        layers.append(Dense(1, activation='sigmoid', name='Output_Layer'))
        
        model = Sequential(layers)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        history = model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0,
                            validation_split=0.2, callbacks=[early_stopping])
        
        update_plots()
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a comma-separated list of numbers for neurons (e.g., 12,8).")


def update_plots():
    """
    Updates the matplotlib plots with the new model's performance.
    """
    global canvas
    # Clear previous plots
    for ax_row in axs:
        for ax in ax_row:
            ax.clear()

    # Plot training & validation accuracy values
    axs[0, 0].plot(history.history['accuracy'])
    axs[0, 0].plot(history.history['val_accuracy'])
    axs[0, 0].set_title('Model accuracy')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    axs[0, 1].plot(history.history['loss'])
    axs[0, 1].plot(history.history['val_loss'])
    axs[0, 1].set_title('Model loss')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].legend(['Train', 'Validation'], loc='upper left')

    # Plot the smoothed ROC curve
    y_pred_keras = model.predict(X_test).ravel()
    fpr_keras, tpr_keras, _ = roc_curve(y_test, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    roc_data = pd.DataFrame({'fpr': fpr_keras, 'tpr': tpr_keras}).drop_duplicates(subset=['fpr'])
    fpr_smooth = np.linspace(0, 1, 300)
    tpr_smooth = interp1d(roc_data['fpr'], roc_data['tpr'], kind='cubic')(fpr_smooth)
    axs[1, 0].plot(fpr_smooth, tpr_smooth, label=f'Keras (area = {auc_keras:.2f})')
    axs[1, 0].plot([0, 1], [0, 1], 'k--')
    axs[1, 0].set_xlabel('False Positive Rate')
    axs[1, 0].set_ylabel('True Positive Rate')
    axs[1, 0].set_title('Smoothed ROC Curve')
    axs[1, 0].legend(loc='lower right')

    # Draw the neural network architecture
    ax_nn = axs[1, 1]
    layer_sizes = [8] + [layer.units for layer in model.layers if isinstance(layer, Dense)]
    layer_colors = ['lightblue'] + ['lightgreen'] * (len(layer_sizes) - 2) + ['lightcoral']
    layer_labels = ['Input Layer'] + [f'Hidden Layer {i+1}' for i in range(len(layer_sizes) - 2)] + ['Output Layer']
    draw_neural_net(ax_nn, .1, .9, .1, .9, layer_sizes, layer_colors, layer_labels)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    canvas.draw()


def predict_diabetes():
    """
    Gets user input from the GUI, preprocesses it, and uses the trained model
    to predict diabetes. Displays the result in the GUI.
    """
    if model is None:
        messagebox.showerror("Model Error", "Please train the model first.")
        return
    try:
        # Get user input from the entry fields
        input_data = [float(entries[i].get()) for i in range(len(column_names)-1)]
        
        # Create a pandas DataFrame from the input data with feature names
        input_df = pd.DataFrame([input_data], columns=column_names[:-1])
        
        # Preprocess the input using the fitted scaler
        input_scaled = scaler.transform(input_df)
        
        # Make a prediction with the trained model
        prediction = model.predict(input_scaled)[0][0]
        
        # Display the prediction result in the result label
        result = "Diabetic" if prediction > 0.5 else "Not Diabetic"
        result_label.config(text=f"Prediction: {result}\nProbability: {prediction:.2f}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")

def fill_random_values():
    """
    Fills the input fields with a random sample from the dataset to provide
    default values.
    """
    random_sample = df.sample(1).iloc[0]
    for i, col_name in enumerate(column_names[:-1]):
        entries[i].delete(0, tk.END)
        entries[i].insert(0, random_sample[col_name])

# --- Main GUI Setup ---

# Create the main window
root = tk.Tk()
root.title("Diabetes Prediction")

# Create main frames for the layout
left_frame = tk.Frame(root, padx=10, pady=10)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = tk.Frame(root, padx=10, pady=10)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# --- Left Frame Widgets (Input Panel) ---

# Create input fields for patient data
entries = []
for i, col_name in enumerate(column_names[:-1]):
    label = tk.Label(left_frame, text=col_name)
    label.grid(row=i, column=0, padx=10, pady=5, sticky='w')
    entry = tk.Entry(left_frame)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

# --- Model Configuration ---
model_config_frame = tk.LabelFrame(left_frame, text="Model Configuration", padx=10, pady=10)
model_config_frame.grid(row=len(column_names)-1, column=0, columnspan=2, pady=10, sticky='ew')

neurons_label = tk.Label(model_config_frame, text="Neurons per Layer (e.g., 12,8):")
neurons_label.pack()
neurons_entry = tk.Entry(model_config_frame)
neurons_entry.pack()
neurons_entry.insert(0, "12,8")

retrain_button = tk.Button(model_config_frame, text="Train/Retrain Model", command=retrain_model)
retrain_button.pack(pady=10)

# --- Action Buttons ---
action_frame = tk.Frame(left_frame)
action_frame.grid(row=len(column_names), column=0, columnspan=2, pady=10)

predict_button = tk.Button(action_frame, text="Predict", command=predict_diabetes)
predict_button.pack(side=tk.LEFT, padx=5)

random_button = tk.Button(action_frame, text="Fill Random", command=fill_random_values)
random_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(action_frame, text="Exit", command=root.quit)
exit_button.pack(side=tk.LEFT, padx=5)


# --- Right Frame Widgets (Output Panel) ---

# Create a label to display the prediction result
result_label = tk.Label(right_frame, text="Prediction: ", font=('Helvetica', 14, 'bold'))
result_label.pack(pady=10)

# Create a figure with a 2x2 grid of subplots for model analysis
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Embed the matplotlib plots in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# --- Initial Actions ---
fill_random_values()
retrain_model()

# Start the tkinter event loop
root.mainloop()