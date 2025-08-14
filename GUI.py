import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import threading

# Load trained model
model = load_model("BEST_MODEL.h5")
IMG_SIZE = (124, 124)

# GUI setup
root = tk.Tk()
root.title("Fire Detection System")
root.geometry("600x750")
root.configure(bg="white")

# ======== Main Center Frame ========
main_frame = tk.Frame(root, bg="white")
main_frame.place(relx=0.5, rely=0.5, anchor='center')  # Centered

# Result Label
label_result = Label(main_frame, text="", font=("Arial", 20, "bold"), fg="black", bg="white")
label_result.pack(pady=20)

# Canvas Frame
canvas_frame = tk.Frame(main_frame, bg="white")
canvas_frame.pack(pady=10)

canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="lightgray", bd=2, relief="ridge")
canvas.pack()

# Preprocess image
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# Classify uploaded image
def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(150, 150, image=img_tk)
    canvas.image = img_tk

    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)[0].item()

    if prediction < 0.5:
        label_result.config(text=f"FIRE DETECTED! ({prediction * 100:.2f}%)", fg="red")
    else:
        label_result.config(text=f"No Fire Detected ({(1 - prediction) * 100:.2f}%)", fg="green")

# Start webcam detection in thread
def start_webcam_detection():
    threading.Thread(target=webcam_detection, daemon=True).start()

# Webcam detection logic
def webcam_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, IMG_SIZE)
        normalized_frame = resized_frame.astype("float32") / 255.0
        input_tensor = np.expand_dims(normalized_frame, axis=0)
        prediction = model.predict(input_tensor)[0].item()

        if prediction < 0.5:
            label = f"FIRE DETECTED! ({prediction * 100:.2f}%)"
            color = (0, 0, 255)
        else:
            label = f"No Fire ({(1 - prediction) * 100:.2f}%)"
            color = (0, 255, 0)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Fire Detection (Webcam)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Buttons Frame
button_frame = tk.Frame(main_frame, bg="white")
button_frame.pack(pady=20)

btn_upload = Button(button_frame, text="Upload Image", command=classify_image, font=("Arial", 14), width=20, bg="#4CAF50", fg="white")
btn_upload.grid(row=0, column=0, padx=10, pady=10)

btn_webcam = Button(button_frame, text="Start Webcam Detection", command=start_webcam_detection, font=("Arial", 14), width=20, bg="#2196F3", fg="white")
btn_webcam.grid(row=1, column=0, padx=10, pady=10)

# Run GUI
root.mainloop()
