"""
# Create mask for a video file
python create_mask.py --video sample_videos/demo.mp4

# Specify custom output directory
python create_mask.py --video sample_videos/demo.mp4 --output custom_masks/

# Start from a specific frame
python create_mask.py --video sample_videos/demo.mp4 --frame 100

"""
import cv2
import numpy as np
import os
import argparse
from pathlib import Path

class InteractiveMaskCreator:
    def __init__(self, video_path, output_dir="masks"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.points = []
        self.mask = None
        self.frame = None
        self.original_frame = None
        self.drawing = False
        self.polygon_complete = False
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Colors for visualization
        self.point_color = (0, 255, 0)      # Green for points
        self.line_color = (0, 255, 0)       # Green for lines
        self.polygon_color = (0, 255, 255)  # Yellow for completed polygon
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon creation"""
        if event == cv2.EVENT_LBUTTONDOWN and not self.polygon_complete:
            # Add new point
            self.points.append((x, y))
            self.update_display()
            print(f"Point {len(self.points)} added at ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) > 0:
            # Remove last point
            self.points.pop()
            self.update_display()
            print(f"Last point removed. Total points: {len(self.points)}")
            
        elif event == cv2.EVENT_MOUSEMOVE and not self.polygon_complete:
            # Show preview line to current mouse position
            if len(self.points) > 0:
                temp_frame = self.frame.copy()
                cv2.line(temp_frame, self.points[-1], (x, y), (128, 128, 128), 2)
                cv2.imshow("Mask Creator", temp_frame)
    
    def update_display(self):
        """Update the display with current points and lines"""
        self.frame = self.original_frame.copy()
        
        # Draw all points
        for i, point in enumerate(self.points):
            cv2.circle(self.frame, point, 5, self.point_color, -1)
            cv2.putText(
                self.frame, str(i+1), (point[0]+10, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.point_color, 1
            )
        
        # Draw lines between consecutive points
        for i in range(len(self.points) - 1):
            cv2.line(self.frame, self.points[i], self.points[i+1], self.line_color, 2)
        
        # If polygon is complete, draw closing line and fill
        if self.polygon_complete and len(self.points) >= 3:
            cv2.line(self.frame, self.points[-1], self.points[0], self.polygon_color, 3)
            overlay = self.frame.copy()
            pts = np.array(self.points, np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
        
        cv2.imshow("Mask Creator", self.frame)
    
    def create_mask_from_polygon(self):
        """Create binary mask from polygon points"""
        if len(self.points) < 3:
            print("Need at least 3 points to create a mask")
            return None
        
        mask = np.zeros(self.original_frame.shape[:2], dtype=np.uint8)
        pts = np.array(self.points, np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def save_mask(self, filename=None):
        """Save the created mask to file"""
        if self.mask is None:
            print("No mask to save")
            return False
        
        if filename is None:
            video_name = Path(self.video_path).stem
            filename = f"{video_name}_mask.png"
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            cv2.imwrite(output_path, self.mask)
            print(f"Mask saved to: {output_path}")
            
            # Also save a preview image showing the mask overlay
            preview_path = os.path.join(
                self.output_dir, f"{Path(filename).stem}_preview.png"
            )
            preview = self.original_frame.copy()
            mask_colored = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 1] = self.mask  # Green channel
            overlay = cv2.addWeighted(preview, 0.7, mask_colored, 0.3, 0)
            cv2.imwrite(preview_path, overlay)
            print(f"Preview saved to: {preview_path}")
            
            return True
        except Exception as e:
            print(f"Error saving mask: {e}")
            return False
    
    def load_video_frame(self, frame_number=0):
        """Load a specific frame from the video"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video: {self.video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video info: {total_frames} frames, {fps} FPS")
        
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read frame from video")
            return False
        
        self.original_frame = frame
        self.frame = frame.copy()
        return True
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("Interactive Mask Creator")
        print("="*60)
        print("INSTRUCTIONS:")
        print("• Left click: Add polygon point")
        print("• Right click: Remove last point")
        print("• 'c' or SPACE: Complete polygon and create mask")
        print("• 's': Save mask to file")
        print("• 'r': Reset and start over")
        print("• 'n': Load next frame (advance by 30 frames)")
        print("• 'p': Load previous frame (go back by 30 frames)")
        print("• 'f': Choose specific frame number")
        print("• 'q' or ESC: Quit without saving")
        print("• 'h': Show this help")
        print("="*60)
    
    def run(self):
        """Main execution loop"""
        if not self.load_video_frame():
            return
        
        self.print_instructions()
        
        cv2.namedWindow("Mask Creator", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Mask Creator", self.mouse_callback)
        
        self.update_display()
        current_frame = 0
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord(' '):
                if len(self.points) >= 3:
                    self.polygon_complete = True
                    self.mask = self.create_mask_from_polygon()
                    self.update_display()
                    print(f"Polygon completed with {len(self.points)} points")
                    print("Press 's' to save the mask")
                else:
                    print("Need at least 3 points to complete polygon")
            
            elif key == ord('s'):
                if self.mask is not None:
                    self.save_mask()
                else:
                    print("No mask to save. Complete polygon first with 'c'")
            
            elif key == ord('r'):
                self.points = []
                self.polygon_complete = False
                self.mask = None
                self.update_display()
                print("Reset complete")
            
            elif key == ord('n'):
                current_frame += 30
                if self.load_video_frame(current_frame):
                    self.points = []
                    self.polygon_complete = False
                    self.mask = None
                    self.update_display()
                    print(f"Loaded frame {current_frame}")
            
            elif key == ord('p'):
                current_frame = max(0, current_frame - 30)
                if self.load_video_frame(current_frame):
                    self.points = []
                    self.polygon_complete = False
                    self.mask = None
                    self.update_display()
                    print(f"Loaded frame {current_frame}")
            
            elif key == ord('f'):
                try:
                    frame_num = int(input("Enter frame number: "))
                    if self.load_video_frame(frame_num):
                        current_frame = frame_num
                        self.points = []
                        self.polygon_complete = False
                        self.mask = None
                        self.update_display()
                        print(f"Loaded frame {frame_num}")
                except ValueError:
                    print("Invalid frame number")
            
            elif key == ord('h'):
                self.print_instructions()
            
            elif key == ord('q') or key == 27:
                print("Exiting mask creator")
                break
        
        cv2.destroyAllWindows()

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Interactive Mask Creator for Vehicle Counter')
    parser.add_argument('--video', type=str, required=True, 
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default='masks', 
                       help='Output directory for masks (default: masks/)')
    parser.add_argument('--frame', type=int, default=0, 
                       help='Starting frame number (default: 0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    creator = InteractiveMaskCreator(args.video, args.output)
    creator.run()

if __name__ == "__main__":
    main()
