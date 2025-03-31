import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk

# Load YOLO model (Ensure you have a trained model for wild animals and humans)
model = YOLO("yolov8n.pt")


def select_file(file_type):
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi"), ("Image Files", "*.jpg;*.png")])
    if file_path:
        analyze_media(file_path, file_type)


def analyze_media(file_path, file_type):
    if file_type == "image":
        image = cv2.imread(file_path)
        analyze_frame(image)
    elif file_type == "video":
        cap = cv2.VideoCapture(file_path)
        process_video(cap)


def analyze_camera():
    cap = cv2.VideoCapture(0)
    process_video(cap)


def process_video(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        analyze_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def analyze_frame(frame):
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)
    cv2.imshow("Detection", frame)


def create_gui():
    root = tk.Tk()
    root.title("Animal & Human Detection")
    root.geometry("400x300")

    tk.Label(root, text="Select an Option", font=("Arial", 14)).pack(pady=10)

    tk.Button(root, text="Analyze Image", command=lambda: select_file("image"), font=("Arial", 12)).pack(pady=5)
    tk.Button(root, text="Analyze Video", command=lambda: select_file("video"), font=("Arial", 12)).pack(pady=5)
    tk.Button(root, text="Analyze Camera", command=analyze_camera, font=("Arial", 12)).pack(pady=5)

    root.mainloop()


create_gui()


def exit_app():
    root.destroy()


# GUI Setup
root = tk.Tk()
root.title("Animal Detection in Video")
root.geometry("600x400")

btn_select = tk.Button(root, text="Upload Video", command=select_video)
btn_select.pack(pady=10)

video_label = tk.Label(root, text="No video selected", wraplength=500)
video_label.pack(pady=5)

video_display = tk.Label(root)
video_display.pack()

btn_exit = tk.Button(root, text="Exit", command=exit_app)
btn_exit.pack(pady=10)

root.mainloop()
