import os
import sys
import csv
import argparse
import numpy as np
import cv2
import cvzone
from ultralytics import YOLO
from sort import Sort

class VehiclePedestrianCounter:
    def __init__(self, source, weights, output, mask_path="mask.png"):
        """
        Initialize the vehicle and pedestrian counter with proper error handling
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
        
        # Counting line (will be adjusted based on video dimensions)
        self.line_pts = [405, 297, 673, 297]
        
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
                self.weights = "yolov8n.pt"  # Use default weights
            
            self.model = YOLO(self.weights)
            print(f"✓ YOLOv8 model loaded successfully from {self.weights}")
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            sys.exit(1)
    
    def _initialize_video_capture(self):
        """Initialize video capture with comprehensive error handling"""
        # Check if source is a file path
        if isinstance(self.source, str) and not self.source.isdigit():
            # Handle different possible video paths
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
                print(f"✗ Video file not found. Searched in:")
                for path in possible_paths:
                    print(f"  - {path}")
                print("\nPlease ensure your video file exists in one of these locations.")
                sys.exit(1)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"✗ Error: Could not open video source: {self.source}")
            sys.exit(1)
        
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✓ Video loaded: {self.source}")
        print(f"  - Dimensions: {self.frame_width}x{self.frame_height}")
        print(f"  - FPS: {self.fps}")
        print(f"  - Total frames: {self.total_frames}")
        
        # Adjust counting line based on video dimensions
        self._adjust_counting_line()
    
    def _adjust_counting_line(self):
        """Adjust counting line coordinates based on video dimensions"""
        # Place line at 1/3 from bottom, spanning most of the width
        margin = self.frame_width // 10
        y_pos = int(self.frame_height * 0.7)  # 70% down from top
        self.line_pts = [margin, y_pos, self.frame_width - margin, y_pos]
        print(f"✓ Counting line adjusted to: {self.line_pts}")
    
    def _load_mask(self):
        """Load region of interest mask with fallback options"""
        try:
            if os.path.exists(self.mask_path):
                self.mask = cv2.imread(self.mask_path)
                if self.mask is None:
                    raise ValueError("Mask file is corrupted or invalid format")
                
                # Resize mask to match video dimensions if necessary
                if self.mask.shape[:2] != (self.frame_height, self.frame_width):
                    self.mask = cv2.resize(self.mask, (self.frame_width, self.frame_height))
                    print(f"✓ Mask resized to match video dimensions")
                
                print(f"✓ Mask loaded from {self.mask_path}")
            else:
                print(f"⚠ Mask file not found: {self.mask_path}")
                print("Creating default full-frame mask...")
                self.mask = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 255
                print("✓ Using full-frame mask (no region restriction)")
                
        except Exception as e:
            print(f"⚠ Error loading mask: {e}")
            print("Using full-frame mask as fallback...")
            self.mask = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 255
    
    def _initialize_tracker(self):
        """Initialize SORT tracker with optimized parameters"""
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        print("✓ SORT tracker initialized")
    
    def process_video(self):
        """Main video processing loop with comprehensive error handling"""
        print("\n🎬 Starting video processing...")
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"\n✓ Video processing complete. Processed {frame_count} frames.")
                    break
                
                frame_count += 1
                
                # Show progress
                if frame_count % 30 == 0:  # Every 30 frames
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
                
                # Display frame (optional - can be disabled for faster processing)
                cv2.imshow("Vehicle and Pedestrian Counter", frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    print("\n⚠ Processing interrupted by user")
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠ Processing interrupted by user")
        except Exception as e:
            print(f"\n✗ Error during video processing: {e}")
        finally:
            self._cleanup()
    
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
        # Calculate intersection coordinates
        ix1, iy1 = max(x1, dx1), max(y1, dy1)
        ix2, iy2 = min(x2, dx2), min(y2, dy2)
        
        # Calculate intersection area
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        
        # Calculate union area
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (dx2 - dx1) * (dy2 - dy1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _check_line_crossing(self, track_id, cx, cy, frame):
        """Check if object crosses the counting line and update count"""
        line_y = self.line_pts[1]
        line_x1, line_x2 = self.line_pts[0], self.line_pts[2]
        
        # Check if object center is on the counting line
        if (line_x1 < cx < line_x2 and abs(cy - line_y) < 15):
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                cls_name = self.id_to_class.get(track_id)
                
                if cls_name:
                    self.class_counts[cls_name] += 1
                    print(f"✓ Counted {cls_name} (ID: {track_id}) - Total: {self.class_counts[cls_name]}")
                
                # Visual feedback - change line color temporarily
                cv2.line(frame, (self.line_pts[0], self.line_pts[1]), 
                        (self.line_pts[2], self.line_pts[3]), (0, 255, 0), 8)
    
    def _draw_ui(self, frame):
        """Draw user interface elements on frame"""
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
    
    def _save_results(self):
        """Save counting results to CSV file"""
        try:
            with open(self.output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Class', 'Count'])
                for cls in self.classes_of_interest:
                    writer.writerow([cls, self.class_counts[cls]])
            
            print(f"\n✓ Results saved to {self.output}")
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
            print(f"✗ Error saving results: {e}")
    
    def _cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self._save_results()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Vehicle and Pedestrian Counter')
    parser.add_argument('--source', type=str, default='sample_videos/demo.mp4', 
                       help='Path to video file or webcam index')
    parser.add_argument('--weights', type=str, default='Yolo-Weights/yolov8n.pt', 
                       help='Path to YOLO weights file')
    parser.add_argument('--output', type=str, default='counts.csv', 
                       help='Output CSV file for results')
    parser.add_argument('--mask', type=str, default='masks/mask.png', 
                       help='Path to region of interest mask')
    
    args = parser.parse_args()
    
    print("🚗🚶 Enhanced Vehicle and Pedestrian Counter")
    print("=" * 50)
    
    # Initialize and run counter
    counter = VehiclePedestrianCounter(
        source=args.source,
        weights=args.weights, 
        output=args.output,
        mask_path=args.mask
    )
    
    counter.process_video()

if __name__ == "__main__":
    main()
