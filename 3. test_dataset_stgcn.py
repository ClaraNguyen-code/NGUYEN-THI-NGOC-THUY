import json
import cv2
import numpy as np
import torch
import pickle
import os
from torch_geometric.nn import GCNConv
from scipy.spatial import distance

# Import your ST-GCN model
from model_stgcn_origin import STGCN_Model

# --- Settings ---
JSON_FILE_PATH = "JSON file/P700C001A0096R001.json"  # Path to your JSON file
INPUT_VIDEO = "data_test/A0096R001.mp4"  # Adjust to your video file
OUTPUT_VIDEO_PATH = "output_video_A0096R001_stgcn.mp4"
MODEL_PATH = "models/stgcn_origin_model.pth"  # Path to your trained model
LABEL_MAP_PATH = "data/label_mapping.pkl"
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
DEVICE = torch.device("cpu")  # Use CPU to avoid cuBLAS errors; change to "cuda" if available

# --- Action name mapping ---
ACTION_NAME_MAP = {
    (0, 4): "Walking",
    (0, 21): "Transition state",
    (0, 28): "Lay down",
    (0, 31): "Access tool",
    (0, 33): "Enter working area",
    (0, 34): "Leave working area",
    (0, 35): "Carry object",
    (0, 36): "Work in progress",
    (0, 37): "Falling down",
    (22, 31): "Pick up object",
    (23, 31): "Put down object",
    (24, 31): "Setup machine"
}

# --- COCO skeleton ---
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (5, 11), (6, 12)  # Torso
]

# --- Function to create edge index from skeleton ---
def create_edge_index(skeleton, num_keypoints=17):
    adj = np.zeros((num_keypoints, num_keypoints))
    for i, j in skeleton:
        adj[i, j] = 1
        adj[j, i] = 1
    edge_index = np.array(np.where(adj))
    return torch.tensor(edge_index, dtype=torch.long).to(DEVICE)

# --- Function to draw skeleton, bounding box, ID, and action label ---
def draw_skeleton(frame, keypoints, connections, bbox=None, person_id="1", action_label="Unknown"):
    keypoints = keypoints.reshape(-1, 3)  # [x, y, confidence]
    for start_idx, end_idx in connections:
        if keypoints[start_idx, 2] > 0.1 and keypoints[end_idx, 2] > 0.1:
            start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
            end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # Green lines
    for i in range(len(keypoints)):
        if keypoints[i, 2] > 0.1:
            cv2.circle(frame, (int(keypoints[i, 0]), int(keypoints[i, 1])), 5, (0, 0, 255), -1)  # Red points
    if bbox is not None:
        x_min, y_min, width, height = bbox
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_min + width), int(y_min + height)), (255, 0, 0), 2)  # Blue bbox
        text = f"ID: {person_id}, Action: {action_label}"
        text_position = (int(x_min), max(int(y_min - 10), 10))
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (int(x_min), int(y_min - 30)), (int(x_min + text_size[0]), int(y_min)), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text
    return frame

# --- Load JSON data ---
with open(JSON_FILE_PATH, "r") as f:
    data = json.load(f)

image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# --- Load model and label mapping ---
model = STGCN_Model(
    in_channels=2,
    num_class=len(ACTION_NAME_MAP),
    edge_index=create_edge_index(SKELETON_CONNECTIONS),
    num_keypoints=NUM_KEYPOINTS
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Load label mapping ---
with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

# Create mapping from action_XX to action name
action_to_name = {action_label: ACTION_NAME_MAP.get(pair, "Unknown") for pair, action_label in label_map.items()}

# --- Create edge index ---
edge_index = create_edge_index(SKELETON_CONNECTIONS)

# --- Open video capture ---
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise ValueError(f"Cannot open video: {INPUT_VIDEO}")

# --- Initialize video writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# --- Function to normalize keypoints ---
def normalize_keypoints(keypoints, bbox):
    x_min, y_min, width, height = bbox
    normalized = []
    for i in range(0, len(keypoints), 3):
        x, y = keypoints[i], keypoints[i + 1]
        x_norm = (x - x_min) / width if width > 1 else x / IMAGE_WIDTH
        y_norm = (y - y_min) / height if height > 1 else y / IMAGE_HEIGHT
        normalized.extend([x_norm, y_norm])
    return np.array(normalized).reshape(NUM_KEYPOINTS, 2)

# --- Function to assign ID based on centroid ---
def assign_id(bbox, prev_centroids, max_distance=100):
    global next_id
    centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if not prev_centroids:
        prev_centroids.append((next_id, centroid))
        return str(next_id), prev_centroids
    distances = [distance.euclidean(centroid, prev[1]) for prev in prev_centroids]
    if min(distances) < max_distance:
        idx = np.argmin(distances)
        prev_centroids[idx] = (prev_centroids[idx][0], centroid)
        return str(prev_centroids[idx][0]), prev_centroids
    prev_centroids.append((next_id, centroid))
    next_id += 1
    return str(next_id - 1), prev_centroids

# --- Buffer for action prediction ---
sequence_buffer = []
prev_centroids = []
next_id = 1
frame_idx = 0

# --- Process each frame ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(data["images"]):
        image_id = data["images"][frame_idx]["id"]
        try:
            annotation = next(ann for ann in data["annotations"] if ann["image_id"] == image_id)
        except StopIteration:
            print(f"No annotation found for frame {frame_idx}")
            out.write(frame)
            frame_idx += 1
            continue

        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
        bbox = annotation["bbox"]

        # Assign ID
        person_id, prev_centroids = assign_id(bbox, prev_centroids)

        # Normalize keypoints (exclude confidence)
        norm_keypoints = normalize_keypoints(keypoints.flatten(), bbox)
        sequence_buffer.append(norm_keypoints.flatten())

        action_label = "Unknown"
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            input_data = np.array(sequence_buffer).reshape(1, SEQUENCE_LENGTH, NUM_KEYPOINTS * 2)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                prediction = model(input_tensor)
                predicted_class = torch.argmax(prediction, dim=1).item()
                action_xx = sorted(list(set(label_map.values())))[predicted_class]
                action_label = action_to_name.get(action_xx, "Unknown")
            sequence_buffer.pop(0)

        # Draw skeleton, bounding box, ID, and action label
        frame = draw_skeleton(frame, keypoints, SKELETON_CONNECTIONS, bbox, person_id, action_label)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Saved output video with predictions to {OUTPUT_VIDEO_PATH}")