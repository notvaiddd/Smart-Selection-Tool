import torch
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

#(loading the pretrained yolov5 model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 192, 203), 
          (128, 0, 128), (255, 255, 0), (0, 255, 255), (255, 20, 147), 
          (0, 128, 128), (0, 128, 0), (139, 69, 19), (75, 0, 130), (255, 105, 180), 
          (220, 20, 60), (112, 128, 144), (0, 0, 139), (34, 139, 34)]

mango_details = []

def extract_shape(contour):
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    return "Oval" if 0.9 < aspect_ratio < 1.1 else "Elongated"

def extract_color_pattern(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=3).fit(img)
    return kmeans.cluster_centers_.astype(int)

def detect_stem_and_crown(image):
    edges = cv2.Canny(image, 50, 150)
    return np.count_nonzero(edges) > 500  

def detect_mangoes(image_path):
    img = Image.open(image_path)
    results = model(img)
    labels, coords, confidences = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.xyxyn[0][:, -2]
    return labels, coords, confidences

def display_mango_features(image, labels, coords, confidences):
    global mango_details  
    output_box.delete("1.0", tk.END)
    image_np = np.array(image)
    output_image = image_np.copy()
    
    
    mango_details.clear()

    for idx, (label, coord) in enumerate(zip(labels, coords)):
        x1, y1, x2, y2 = int(coord[0] * image_np.shape[1]), int(coord[1] * image_np.shape[0]), int(coord[2] * image_np.shape[1]), int(coord[3] * image_np.shape[0])
        mango_img = image_np[y1:y2, x1:x2]
        contour, _ = cv2.findContours(cv2.cvtColor(mango_img, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape = extract_shape(contour[0])
        colors = extract_color_pattern(mango_img)
        stem_present = detect_stem_and_crown(mango_img)
        area = (x2 - x1) * (y2 - y1)
        
       
        mango_details.append((idx + 1, shape, area, stem_present))

        output_box.insert(tk.END, f"Mango {idx + 1}:\n", "mango_title")
        output_box.insert(tk.END, f"  Shape: {shape}\n")
        
        for color in colors:
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
            output_box.insert(tk.END, f"  Color Pattern: {color_hex}\n")

        output_box.insert(tk.END, f"  Stem/Crown Present: {'Yes' if stem_present else 'No'}\n")
        output_box.insert(tk.END, f"  Size (Length x Width): {(y2-y1)} x {(x2-x1)}\n\n")
        
        
        box_color = COLORS[idx % len(COLORS)]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 2)

        
        cv2.putText(output_image, str(idx + 1), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    
    output_image_pil = Image.fromarray(output_image)
    img_tk = ImageTk.PhotoImage(output_image_pil)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

def evaluate_best_mango():
    if not mango_details:
        messagebox.showwarning("No Mangoes Detected", "Please upload an image first and perform detection.")
        return

    best_mango = max(mango_details, key=lambda x: (x[2], x[3]))  
    show_best_mango(best_mango)

def show_best_mango(mango):
    best_mango_window = Toplevel(root)
    best_mango_window.title("Best Mango Evaluation")
    best_mango_window.geometry("400x300")

    title_label = tk.Label(best_mango_window, text="Best Mango Details", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=10)

    details = f"Mango {mango[0]}:\n" \
              f"Shape: {mango[1]}\n" \
              f"Size: {mango[2]}\n" \
              f"Stem/Crown Present: {'Yes' if mango[3] else 'No'}"

    details_label = tk.Label(best_mango_window, text=details, font=("Helvetica", 14))
    details_label.pack(pady=20)

    close_btn = tk.Button(best_mango_window, text="Close", command=best_mango_window.destroy)
    close_btn.pack(pady=10)

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((600, 600))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

        labels, coords, confidences = detect_mangoes(file_path)
        display_mango_features(img, labels, coords, confidences)



#Tkinter




def confirm_exit():
    if messagebox.askyesno("Quit", "Are you sure you want to exit?"):
        root.destroy()




root = tk.Tk()
root.title("Mango Detection and Feature Extraction")
root.state("zoomed")  


title_label = tk.Label(root, text="Feed", font=("Helvetica", 24, "bold"), pady=10)
title_label.pack()


img_label = tk.Label(root)
img_label.pack()



upload_btn = tk.Button(root, text="Upload Image", font=("Helvetica", 14), command=upload_image)
upload_btn.pack(pady=10)



evaluate_btn = tk.Button(root, text="EVALUATE", font=("Helvetica", 14), command=evaluate_best_mango)
evaluate_btn.pack(pady=10)



output_box = tk.Text(root, height=20, width=80, font=("Times New Roman", 14), bg="#e6e6e6", fg="black")
output_box.tag_configure("mango_title", foreground="orange", font=("Times New Roman", 16, "bold"))
output_box.pack(pady=10)


quit_btn = tk.Button(root, text="Quit", font=("Helvetica", 14), command=confirm_exit)
quit_btn.pack(pady=10)










root.mainloop()
