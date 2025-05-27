import cv2
import numpy as np
import os
import argparse
from pathlib import Path

class InteractiveMaskCreator:
    def __init__(self, video_path, output_dir="masks", max_display_size=(1200, 800)):
        self.video_path = video_path
        self.output_dir = output_dir
        self.max_display_size = max_display_size
        self.points = []
        self.mask = None
        self.frame = None
        self.original_frame = None
        self.display_frame = None
        self.scale_factor = 1.0
        self.drawing = False
        self.polygon_complete = False
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Colors for visualization
        self.point_color = (0, 255, 0)  # Green for points
        self.line_color = (0, 255, 0)   # Green for lines
        self.polygon_color = (0, 255, 255)  # Yellow for completed polygon
        
    def calculate_display_scale(self, frame_width, frame_height):
        """Calculate optimal scale factor to fit frame within display limits"""
        max_width, max_height = self.max_display_size
        
        # Calculate scale factors for both dimensions
        width_scale = max_width / frame_width
        height_scale = max_height / frame_height
        
        # Use the smaller scale factor to ensure both dimensions fit
        scale_factor = min(width_scale, height_scale, 1.0)  # Don't upscale
        
        return scale_factor
    
    def scale_point_to_original(self, display_point):
        """Convert display coordinates to original frame coordinates"""
        x, y = display_point
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        return (orig_x, orig_y)
    
    def scale_point_to_display(self, original_point):
        """Convert original coordinates to display coordinates"""
        x, y = original_point
        disp_x = int(x * self.scale_factor)
        disp_y = int(y * self.scale_factor)
        return (disp_x, disp_y)
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon creation"""
        if event == cv2.EVENT_LBUTTONDOWN and not self.polygon_complete:
            # Convert display coordinates to original frame coordinates
            original_point = self.scale_point_to_original((x, y))
            self.points.append(original_point)
            self.update_display()
            print(f"Point {len(self.points)} added at original coords: {original_point}")
            
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) > 0:
            # Remove last point
            self.points.pop()
            self.update_display()
            print(f"Last point removed. Total points: {len(self.points)}")
            
        elif event == cv2.EVENT_MOUSEMOVE and not self.polygon_complete:
            # Show preview line to current mouse position
            if len(self.points) > 0:
                temp_frame = self.display_frame.copy()
                last_display_point = self.scale_point_to_display(self.points[-1])
                cv2.line(temp_frame, last_display_point, (x, y), (128, 128, 128), 2)
                cv2.imshow("Mask Creator", temp_frame)
    
    def update_display(self):
        """Update the display with current points and lines"""
        # Create display frame (scaled version)
        self.display_frame = cv2.resize(self.original_frame, 
                                       (int(self.original_frame.shape[1] * self.scale_factor),
                                        int(self.original_frame.shape[0] * self.scale_factor)))
        
        # Convert all points to display coordinates and draw
        display_points = [self.scale_point_to_display(p) for p in self.points]
        
        # Draw all points
        for i, point in enumerate(display_points):
            cv2.circle(self.display_frame, point, 5, self.point_color, -1)
            cv2.putText(self.display_frame, str(i+1), (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.point_color, 1)
        
        # Draw lines between consecutive points
        for i in range(len(display_points) - 1):
            cv2.line(self.display_frame, display_points[i], display_points[i+1], self.line_color, 2)
        
        # If polygon is complete, draw closing line and fill
        if self.polygon_complete and len(display_points) >= 3:
            cv2.line(self.display_frame, display_points[-1], display_points[0], self.polygon_color, 3)
            # Create semi-transparent overlay
            overlay = self.display_frame.copy()
            pts = np.array(display_points, np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
        
        # Add scale information to display
        scale_text = f"Scale: {self.scale_factor:.2f}x | Original: {self.original_frame.shape[1]}x{self.original_frame.shape[0]}"
        cv2.putText(self.display_frame, scale_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Mask Creator", self.display_frame)
    
    def create_mask_from_polygon(self):
        """Create binary mask from polygon points using original coordinates"""
        if len(self.points) < 3:
            print("Need at least 3 points to create a mask")
            return None
        
        # Create binary mask using original frame dimensions
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
            # Generate filename from video name
            video_name = Path(self.video_path).stem
            filename = f"{video_name}_mask.png"
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            cv2.imwrite(output_path, self.mask)
            print(f"✓ Mask saved to: {output_path}")
            print(f"  Mask dimensions: {self.mask.shape}")
            print(f"  Original video dimensions: {self.original_frame.shape[:2]}")
            
            # Also save a preview image showing the mask overlay
            preview_path = os.path.join(self.output_dir, f"{Path(filename).stem}_preview.png")
            preview = self.original_frame.copy()
            mask_colored = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:,:,1] = self.mask  # Green channel
            overlay = cv2.addWeighted(preview, 0.7, mask_colored, 0.3, 0)
            cv2.imwrite(preview_path, overlay)
            print(f"✓ Preview saved to: {preview_path}")
            
            return True
        except Exception as e:
            print(f"✗ Error saving mask: {e}")
            return False
    
    def load_video_frame(self, frame_number=0):
        """Load a specific frame from the video and calculate display scaling"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open video: {self.video_path}")
            return False
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frames} frames, {fps} FPS, {frame_width}x{frame_height}")
        
        # Calculate display scale factor
        self.scale_factor = self.calculate_display_scale(frame_width, frame_height)
        display_width = int(frame_width * self.scale_factor)
        display_height = int(frame_height * self.scale_factor)
        
        print(f"Display scaling: {self.scale_factor:.2f}x -> {display_width}x{display_height}")
        
        # Set frame position
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ Error: Could not read frame from video")
            return False
        
        self.original_frame = frame
        return True
    
    def setup_window(self):
        """Setup OpenCV window with proper configuration"""
        window_name = "Mask Creator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        
        # Set initial window size based on scaled dimensions
        display_width = int(self.original_frame.shape[1] * self.scale_factor)
        display_height = int(self.original_frame.shape[0] * self.scale_factor)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        return window_name
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("Interactive Mask Creator")
        print("="*60)
        print("WINDOW CONTROLS:")
        print("• Window is automatically scaled to fit your screen")
        print("• You can manually resize the window by dragging corners")
        print("• All coordinates are automatically converted to original resolution")
        print("\nANNOTATION CONTROLS:")
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
        # Load initial frame
        if not self.load_video_frame():
            return
        
        self.print_instructions()
        
        window_name = self.setup_window()
        self.update_display()
        
        current_frame = 0
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord(' '):  # Complete polygon
                if len(self.points) >= 3:
                    self.polygon_complete = True
                    self.mask = self.create_mask_from_polygon()
                    self.update_display()
                    print(f"✓ Polygon completed with {len(self.points)} points")
                    print("Press 's' to save the mask")
                else:
                    print("⚠ Need at least 3 points to complete polygon")
            
            elif key == ord('s'):  # Save mask
                if self.mask is not None:
                    self.save_mask()
                else:
                    print("⚠ No mask to save. Complete polygon first with 'c'")
            
            elif key == ord('r'):  # Reset
                self.points = []
                self.polygon_complete = False
                self.mask = None
                self.update_display()
                print("Reset complete")
            
            elif key == ord('n'):  # Next frame
                current_frame += 30
                if self.load_video_frame(current_frame):
                    self.points = []
                    self.polygon_complete = False
                    self.mask = None
                    cv2.resizeWindow(window_name, 
                                   int(self.original_frame.shape[1] * self.scale_factor),
                                   int(self.original_frame.shape[0] * self.scale_factor))
                    self.update_display()
                    print(f"Loaded frame {current_frame}")
            
            elif key == ord('p'):  # Previous frame
                current_frame = max(0, current_frame - 30)
                if self.load_video_frame(current_frame):
                    self.points = []
                    self.polygon_complete = False
                    self.mask = None
                    cv2.resizeWindow(window_name, 
                                   int(self.original_frame.shape[1] * self.scale_factor),
                                   int(self.original_frame.shape[0] * self.scale_factor))
                    self.update_display()
                    print(f"Loaded frame {current_frame}")
            
            elif key == ord('f'):  # Choose specific frame
                try:
                    frame_num = int(input("Enter frame number: "))
                    if self.load_video_frame(frame_num):
                        current_frame = frame_num
                        self.points = []
                        self.polygon_complete = False
                        self.mask = None
                        cv2.resizeWindow(window_name, 
                                       int(self.original_frame.shape[1] * self.scale_factor),
                                       int(self.original_frame.shape[0] * self.scale_factor))
                        self.update_display()
                        print(f"Loaded frame {frame_num}")
                except ValueError:
                    print("⚠ Invalid frame number")
            
            elif key == ord('h'):  # Help
                self.print_instructions()
            
            elif key == ord('q') or key == 27:  # Quit
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
    parser.add_argument('--max-size', type=int, nargs=2, default=[1200, 800],
                       help='Maximum display size as width height (default: 1200 800)')
    
    args = parser.parse_args()
    
    # Validate video file exists
    if not os.path.exists(args.video):
        print(f"✗ Error: Video file not found: {args.video}")
        return
    
    # Create mask creator and run
    creator = InteractiveMaskCreator(args.video, args.output, tuple(args.max_size))
    creator.run()

if __name__ == "__main__":
    main()
