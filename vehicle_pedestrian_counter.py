from ultralytics import YOLO
import cv2, cvzone, numpy as np, argparse
from sort import Sort

# Parse arguments for input video, output CSV, YOLO weights
parser = argparse.ArgumentParser(description='Vehicle and Pedestrian Counter')
parser.add_argument('--source', type=str, default='videos/demo.mp4', help='Path to video file')
parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Path to YOLO weights')
parser.add_argument('--output', type=str, default='counts.csv', help='Output CSV file')
args = parser.parse_args()

# Load YOLOv8 model
model = YOLO(args.weights)

# Classes to detect and count
classes_of_interest = ["car", "truck", "bus", "motorbike", "person"]
display_names = {"car":"Cars","truck":"Trucks","bus":"Buses","motorbike":"Motorbikes","person":"People"}

# Video capture (file or webcam)
cap = cv2.VideoCapture(args.source)
mask = cv2.imread("mask.png")  # region-of-interest mask

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counting line (drawn on image)
line_pts = [405, 297, 673, 297]

# Data structures for ID-class mapping and counts
id_to_class = {}
counted_ids = set()
class_counts = {cls:0 for cls in classes_of_interest}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply mask, run YOLOv8 detection
    roi = cv2.bitwise_and(frame, mask)
    results = model(roi, stream=True)

    # Prepare detections array for SORT and record raw detection info
    detections = np.empty((0,5))
    raw_dets = []
    for res in results:
        for box in res.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            if name in classes_of_interest and conf > 0.3:
                raw_dets.append((x1, y1, x2, y2, conf, name))
                detections = np.vstack((detections, np.array([x1,y1,x2,y2,conf])))

    # Update tracker with current frame detections
    tracked = tracker.update(detections)

    # Draw counting line
    cv2.line(frame, (line_pts[0], line_pts[1]), (line_pts[2], line_pts[3]), (0,0,255), 5)

    # Process each tracked object
    for x1,y1,x2,y2,track_id in tracked:
        x1,y1,x2,y2,tid = map(int, (x1,y1,x2,y2,track_id))
        w,h = x2-x1, y2-y1

        # Draw bounding box + ID
        cvzone.cornerRect(frame, (x1,y1,w,h))
        cvzone.putTextRect(frame, f'ID {tid}', (x1, y1-10), scale=1, thickness=2)

        # Center point of box
        cx, cy = x1 + w//2, y1 + h//2
        cv2.circle(frame, (cx, cy), 5, (255,0,255), cv2.FILLED)

        # Assign class to this ID if unknown
        if tid not in id_to_class:
            best_iou = 0; best_class = None
            for dx1,dy1,dx2,dy2,dconf, dcls in raw_dets:
                # compute IoU...
                ix1, iy1 = max(x1,dx1), max(y1,dy1)
                ix2, iy2 = min(x2,dx2), min(y2,dy2)
                iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
                if iw>0 and ih>0:
                    inter = iw * ih
                    union = (w*h) + ((dx2-dx1)*(dy2-dy1)) - inter
                    iou = inter / union if union>0 else 0
                else:
                    iou = 0
                if iou > best_iou:
                    best_iou = iou; best_class = dcls
            if best_class:
                id_to_class[tid] = best_class

        # Check line crossing for counting
        if line_pts[0] < cx < line_pts[2] and abs(cy - line_pts[1]) < 20:
            if tid not in counted_ids:
                counted_ids.add(tid)
                cls_name = id_to_class.get(tid)
                if cls_name:
                    class_counts[cls_name] += 1
                # change line color to green as feedback
                cv2.line(frame, (line_pts[0], line_pts[1]), (line_pts[2], line_pts[3]), (0,255,0), 5)

    # Overlay counts on frame
    start_y, dy = 50, 30
    for i, cls in enumerate(classes_of_interest):
        text = f"{display_names[cls]}: {class_counts[cls]}"
        cvzone.putTextRect(frame, text, (50, start_y + i*dy), scale=1, thickness=2)

    cv2.imshow("Vehicle and Pedestrian Counter", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

# After processing, write counts to CSV
import csv
with open(args.output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Class','Count'])
    for cls in classes_of_interest:
        writer.writerow([cls, class_counts[cls]])
cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"Counting complete. Results saved to {args.output}")
# This script counts vehicles and pedestrians in a video using YOLOv8 and SORT tracking.