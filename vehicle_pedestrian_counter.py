import os
import sys
import csv
import argparse
import numpy as np
import cv2
import cvzone
from ultralytics import YOLO
from sort import Sort

class DynamicVehiclePedestrianCounter:
    def __init__(self, source, weights, output, mask_path="mask.png"):
        """
        Initialize the vehicle and pedestrian counter with dynamic line positioning
        """
        self.source = source
        self.weights = weights
        self.output = output
        self.mask_path = mask_path

        # Classes to detect and count
        self.classes_of_interest = ["car", "truck", "bus", "motorbike", "person"]
        self.display_names = {
            "car": "Cars", "truck": "Trucks", "bus": "Buses",
            "motorbike": "Motorbikes", "person": "People"
        }

        # Initialize counters and tracking data
        self.id_to_class = {}
        self.counted_ids = set()
        self.class_counts = {cls: 0 for cls in self.classes_of_interest}

        # Line positioning variables
        self.line_pts = None
        self.line_set = False
        self.drawing_line = False
        self.temp_point = None
        self.setup_frame = None

        # Initialize components
        self._initialize_model()
        self._initialize_video_capture()
        self._load_mask()
        self._initialize_tracker()

    def _initialize_model(self):
        """Load YOLOv8 model with error handling"""
        try:
            if not os.path.exists(self.weights):
                print(f"Warning: Weights file not found at {self.weights}")
                print("Attempting to download YOLOv8 weights...")
                self.weights = "yolov8n.pt"

            self.model = YOLO(self.weights)
            print(f"YOLOv8 model loaded successfully from {self.weights}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)

    def _initialize_video_capture(self):
        """Initialize video capture with comprehensive error handling"""
        if isinstance(self.source, str) and not self.source.isdigit():
            possible_paths = [
                self.source,
                os.path.join("sample_videos", os.path.basename(self.source)),
                os.path.join("videos", os.path.basename(self.source)),
                os.path.join("Videos", os.path.basename(self.source))
            ]

            video_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.source = path
                    video_found = True
                    break

            if not video_found:
                print("Video file not found. Searched in:")
                for path in possible_paths:
                    print(f"  - {path}")
                sys.exit(1)

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
            sys.exit(1)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video loaded: {self.source}")
        print(f"  - Dimensions: {self.frame_width}x{self.frame_height}")
        print(f"  - FPS: {self.fps}")
        print(f"  - Total frames: {self.total_frames}")

    def _load_mask(self):
        """Load region of interest mask with fallback options"""
        try:
            if os.path.exists(self.mask_path):
                self.mask = cv2.imread(self.mask_path)
                if self.mask is None:
                    raise ValueError("Mask file is corrupted or invalid format")

                if self.mask.shape[:2] != (self.frame_height, self.frame_width):
                    self.mask = cv2.resize(self.mask, (self.frame_width, self.frame_height))
                    print("Mask resized to match video dimensions")

                print(f"Mask loaded from {self.mask_path}")
            else:
                print(f"Mask file not found: {self.mask_path}")
                print("Creating default full-frame mask...")
                self.mask = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 255
        except Exception as e:
            print(f"Error loading mask: {e}")
            print("Using full-frame mask as fallback...")
            self.mask = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 255

    def _initialize_tracker(self):
        """Initialize SORT tracker with optimized parameters"""
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        print("SORT tracker initialized")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing counting line"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_line = True
            self.line_pts = [x, y, x, y]
            print(f"Started drawing line at: ({x}, {y})")

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_line:
            # Update the end point of the line
            self.line_pts[2] = x
            self.line_pts[3] = y
            
            # Create a temporary image to show the line being drawn
            temp_frame = self.setup_frame.copy()
            cv2.line(temp_frame, (self.line_pts[0], self.line_pts[1]), 
                    (self.line_pts[2], self.line_pts[3]), (0, 0, 255), 3)
            
            # Add instruction text
            cv2.putText(temp_frame, "Draw counting line - Release to confirm", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(temp_frame, "Press 'r' to reset, 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Setup Counting Line", temp_frame)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing_line:
            self.drawing_line = False
            self.line_set = True
            print(f"Line set from ({self.line_pts[0]}, {self.line_pts[1]}) to ({self.line_pts[2]}, {self.line_pts[3]})")
            
            # Show final line
            temp_frame = self.setup_frame.copy()
            cv2.line(temp_frame, (self.line_pts[0], self.line_pts[1]), 
                    (self.line_pts[2], self.line_pts[3]), (0, 0, 255), 3)
            cv2.putText(temp_frame, "Line set! Press SPACE to continue or 'r' to reset", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Setup Counting Line", temp_frame)

    def setup_counting_line(self):
        """Interactive setup for counting line position"""
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return False

        self.setup_frame = frame.copy()
        
        # Create setup window
        cv2.namedWindow("Setup Counting Line", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Setup Counting Line", self.mouse_callback)
        
        # Show initial frame with instructions
        display_frame = self.setup_frame.copy()
        cv2.putText(display_frame, "Click and drag to draw counting line", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'r' to reset, 'q' to quit", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Setup Counting Line", display_frame)

        print("\nSetup Instructions:")
        print("1. Click and drag to draw the counting line")
        print("2. Press SPACE to confirm and start counting")
        print("3. Press 'r' to reset the line")
        print("4. Press 'q' to quit setup")

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Setup cancelled by user")
                cv2.destroyWindow("Setup Counting Line")
                return False
            elif key == ord('r'):
                # Reset line
                self.line_pts = None
                self.line_set = False
                self.drawing_line = False
                display_frame = self.setup_frame.copy()
                cv2.putText(display_frame, "Click and drag to draw counting line", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'r' to reset, 'q' to quit", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Setup Counting Line", display_frame)
                print("Line reset - draw a new line")
            elif key == ord(' ') and self.line_set:
                # Confirm line and start processing
                cv2.destroyWindow("Setup Counting Line")
                print(f"Counting line confirmed: {self.line_pts}")
                
                # Reset video to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return True

    def _check_line_crossing(self, track_id, cx, cy, frame):
        """Check if object crosses the counting line using line equation"""
        if not self.line_pts:
            return

        x1, y1, x2, y2 = self.line_pts
        
        # Calculate distance from point to line
        # Line equation: (y2-y1)x - (x2-x1)y + x2*y1 - y2*x1 = 0
        # Distance = |ax + by + c| / sqrt(a² + b²)
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - y2 * x1
        
        distance = abs(a * cx + b * cy + c) / np.sqrt(a*a + b*b)
        
        # Check if point is close to line (within threshold)
        threshold = 15
        if distance < threshold:
            # Check if point is within line segment bounds
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            # Add some tolerance for line segment bounds
            tolerance = 20
            if (min_x - tolerance <= cx <= max_x + tolerance and 
                min_y - tolerance <= cy <= max_y + tolerance):
                
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    cls_name = self.id_to_class.get(track_id)

                    if cls_name:
                        self.class_counts[cls_name] += 1
                        print(f"Counted {cls_name} (ID: {track_id}) - Total: {self.class_counts[cls_name]}")

                    # Visual feedback - change line color temporarily
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 8)

    def _detect_objects(self, roi):
        """Detect objects using YOLOv8 and filter relevant classes"""
        results = self.model(roi, stream=True, verbose=False)
        detections = np.empty((0, 5))
        self.raw_detections = []

        for res in results:
            if res.boxes is not None:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]

                    if class_name in self.classes_of_interest and conf > 0.3:
                        self.raw_detections.append((x1, y1, x2, y2, conf, class_name))
                        detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

        return detections

    def _assign_class_to_track(self, track_id, x1, y1, x2, y2):
        """Assign object class to track ID using IoU matching"""
        best_iou = 0
        best_class = None

        for dx1, dy1, dx2, dy2, dconf, dcls in self.raw_detections:
            iou = self._calculate_iou(x1, y1, x2, y2, dx1, dy1, dx2, dy2)
            if iou > best_iou:
                best_iou = iou
                best_class = dcls

        if best_class and best_iou > 0.3:
            self.id_to_class[track_id] = best_class

    def _calculate_iou(self, x1, y1, x2, y2, dx1, dy1, dx2, dy2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        ix1, iy1 = max(x1, dx1), max(y1, dy1)
        ix2, iy2 = min(x2, dx2), min(y2, dy2)

        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (dx2 - dx1) * (dy2 - dy1)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _process_tracked_objects(self, frame, tracked_objects, detections):
        """Process tracked objects and handle counting logic"""
        for x1, y1, x2, y2, track_id in tracked_objects:
            x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, track_id))
            w, h = x2 - x1, y2 - y1

            # Draw bounding box and ID
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2)
            cvzone.putTextRect(frame, f'ID {tid}', (x1, y1 - 10), scale=1, thickness=2)

            # Calculate center point
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Assign class to track ID if not already assigned
            if tid not in self.id_to_class:
                self._assign_class_to_track(tid, x1, y1, x2, y2)

            # Check for line crossing
            self._check_line_crossing(tid, cx, cy, frame)

    def _draw_ui(self, frame):
        """Draw user interface elements on frame"""
        if self.line_pts:
            # Draw counting line
            cv2.line(frame, (self.line_pts[0], self.line_pts[1]),
                     (self.line_pts[2], self.line_pts[3]), (0, 0, 255), 3)

        # Draw count overlay
        start_y, dy = 50, 35
        for i, cls in enumerate(self.classes_of_interest):
            text = f"{self.display_names[cls]}: {self.class_counts[cls]}"
            cvzone.putTextRect(frame, text, (50, start_y + i * dy),
                               scale=1.2, thickness=2, colorT=(255, 255, 255),
                               colorR=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX)

    def process_video(self):
        """Main video processing loop with dynamic line setup"""
        print("\nStarting dynamic line setup...")
        
        # First setup the counting line
        if not self.setup_counting_line():
            return

        print("\nStarting video processing...")
        cv2.namedWindow("Vehicle and Pedestrian Counter", cv2.WINDOW_NORMAL)
        frame_count = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"\nVideo processing complete. Processed {frame_count} frames.")
                    break

                frame_count += 1

                if frame_count % 30 == 0:
                    progress = (frame_count / self.total_frames) * 100 if self.total_frames > 0 else 0
                    print(f"Processing frame {frame_count}... ({progress:.1f}%)")

                # Apply mask and run detection
                roi = cv2.bitwise_and(frame, self.mask)
                detections = self._detect_objects(roi)

                # Update tracker
                tracked_objects = self.tracker.update(detections)

                # Process tracked objects
                self._process_tracked_objects(frame, tracked_objects, detections)

                # Draw UI elements
                self._draw_ui(frame)

                # Display frame
                cv2.imshow("Vehicle and Pedestrian Counter", frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    print("\nProcessing interrupted by user")
                    break

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        except Exception as e:
            print(f"\nError during video processing: {e}")
        finally:
            self._cleanup()

    def _save_results(self):
        """Save counting results to CSV file"""
        try:
            with open(self.output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Class', 'Count'])
                for cls in self.classes_of_interest:
                    writer.writerow([cls, self.class_counts[cls]])

            print(f"\nResults saved to {self.output}")
            print("\nFinal Count Summary:")
            print("-" * 30)
            total_count = 0
            for cls in self.classes_of_interest:
                count = self.class_counts[cls]
                print(f"{self.display_names[cls]:12}: {count:4d}")
                total_count += count
            print("-" * 30)
            print(f"{'Total':12}: {total_count:4d}")

        except Exception as e:
            print(f"Error saving results: {e}")

    def _cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self._save_results()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Dynamic Vehicle and Pedestrian Counter')
    parser.add_argument('--source', type=str, default='sample_videos/demo.mp4',
                        help='Path to video file or webcam index')
    parser.add_argument('--weights', type=str, default='Yolo-Weights/yolov8n.pt',
                        help='Path to YOLO weights file')
    parser.add_argument('--output', type=str, default='counts.csv',
                        help='Output CSV file for results')
    parser.add_argument('--mask', type=str, default='masks/mask.png',
                        help='Path to region of interest mask')

    args = parser.parse_args()

    print("Dynamic Vehicle and Pedestrian Counter")
    print("=" * 50)

    # Initialize and run counter
    counter = DynamicVehiclePedestrianCounter(
        source=args.source,
        weights=args.weights,
        output=args.output,
        mask_path=args.mask
    )

    counter.process_video()

if __name__ == "__main__":
    main()
