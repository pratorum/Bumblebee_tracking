#!/usr/bin/env python3
"""
Simple demo to practice click and drag technique
"""

import cv2
import numpy as np

def click_drag_demo():
    """Simple demo to practice click and drag"""
    
    # Create a simple colored image
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    image[:] = (50, 50, 50)  # Dark gray background
    
    # Add instructions
    cv2.putText(image, "CLICK AND DRAG DEMO", (150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, "1. Click and HOLD mouse button", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, "2. While holding, DRAG mouse", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, "3. Release mouse button", (50, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, "4. Press SPACEBAR to confirm", (50, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, "Press ESC to exit", (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    drawing = False
    start_point = None
    end_point = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            print(f"✓ CLICKED at ({x}, {y}) - Now DRAG!")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                # Draw rectangle as you drag
                temp_image = image.copy()
                cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)
                cv2.imshow("Click and Drag Demo", temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            print(f"✓ RELEASED at ({x}, {y}) - Rectangle complete!")
    
    cv2.setMouseCallback("Click and Drag Demo", mouse_callback)
    cv2.imshow("Click and Drag Demo", image)
    
    print("Click and Drag Demo started!")
    print("Try clicking and dragging on the image...")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Spacebar
            if start_point and end_point:
                x1 = min(start_point[0], end_point[0])
                y1 = min(start_point[1], end_point[1])
                w = abs(end_point[0] - start_point[0])
                h = abs(end_point[1] - start_point[1])
                print(f"🎉 SUCCESS! Rectangle: x={x1}, y={y1}, w={w}, h={h}")
                break
            else:
                print("❌ No rectangle drawn yet!")
        elif key == 27:  # ESC
            print("Exiting...")
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    click_drag_demo()
