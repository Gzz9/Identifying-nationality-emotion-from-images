import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# =======================
# Load models
# =======================
age_nat_model = load_model('utkface_model.keras')
emotion_model = load_model('emotion_model.keras')

# =======================
# Helper functions
# =======================
def preprocess_utkface(image):
    img = cv2.resize(image, (64,64))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_emotion(image):
    img = cv2.resize(image, (64,64))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def extract_dress_color(image, k=3):
    h = image.shape[0]
    dress_region = image[h//2:, :, :]
    dress_region_small = cv2.resize(dress_region, (50, 50))
    dress_pixels = dress_region_small.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(dress_pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))].astype(int)
    color_name = map_rgb_to_name(dominant_color)
    return color_name, dominant_color

def map_rgb_to_name(rgb):
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif r > 150 and g < 80 and b < 80:
        return "Red"
    elif r < 80 and g > 150 and b < 80:
        return "Green"
    elif r < 80 and g < 80 and b > 150:
        return "Blue"
    elif r > 150 and g > 150 and b < 80:
        return "Yellow"
    elif r > 150 and b > 150 and g < 80:
        return "Pink/Purple"
    else:
        return f"RGB({r},{g},{b})"

def format_probs(prob_dict):
    lines = []
    for label, prob in prob_dict.items():
        lines.append(f"{label}: {prob:.2f}")
    return "\n".join(lines)

def predict(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Nationality + age
    utk_input = preprocess_utkface(img_rgb)
    nat_pred, age_pred = age_nat_model.predict(utk_input)
    nat_class = np.argmax(nat_pred)
    age_class = np.argmax(age_pred)

    nat_label = {0:'Indian', 1:'US', 2:'African', 3:'Others'}.get(nat_class, 'Unknown')
    age_mapping = {
        0: '0-9', 1: '10-19', 2: '20-29', 3: '30-39',
        4: '40-49', 5: '50-59', 6: '60+'
    }
    age_range = age_mapping.get(age_class, str(age_class))

    # Emotion
    emo_input = preprocess_emotion(img_rgb)
    emo_pred = emotion_model.predict(emo_input)
    emo_class = np.argmax(emo_pred)
    emo_label_map = {
        0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',
        4:'Sad', 5:'Surprise', 6:'Neutral'
    }
    emo_label = emo_label_map.get(emo_class, 'Unknown')

    emo_probs = {emo_label_map[i]: emo_pred[0][i] for i in range(len(emo_pred[0]))}
    age_probs = {age_mapping.get(i, str(i)): age_pred[0][i] for i in range(len(age_pred[0]))}

    # Dress color
    dress_color = None
    if nat_label in ['Indian', 'African']:
        dress_color, _ = extract_dress_color(img_rgb)

    return nat_label, age_range, emo_label, dress_color, emo_probs, age_probs

# =======================
# GUI functions
# =======================
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display image
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Run prediction
        nat, age, emo, color, emo_probs, age_probs = predict(file_path)

        result_textbox.delete('1.0', tk.END)
        result_textbox.insert(tk.END, f"Nationality: {nat}\n\n")
        result_textbox.insert(tk.END, f"Emotion: {emo}\n")
        result_textbox.insert(tk.END, "Emotion probabilities:\n")
        result_textbox.insert(tk.END, format_probs(emo_probs) + "\n\n")

        if nat in ['Indian', 'US']:
            result_textbox.insert(tk.END, f"Age group: {age}\n")
            result_textbox.insert(tk.END, "Age probabilities:\n")
            result_textbox.insert(tk.END, format_probs(age_probs) + "\n\n")

        if nat in ['Indian', 'African'] and color:
            result_textbox.insert(tk.END, f"Dress Color: {color}\n")

# =======================
# Build GUI
# =======================
root = tk.Tk()
root.title("Face Attribute Predictor")

btn = tk.Button(root, text="Upload Image", command=open_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

result_textbox = tk.Text(root, width=50, height=20, font=("Arial", 12))
result_textbox.pack()

root.mainloop()
