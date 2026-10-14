import cv2
import torch

def run_inference(source=0, weights='weights/best.pt'):
    # Load TS-YOLO model
    model = torch.hub.load('.', 'custom', path=weights, source='local')
    
    # Open video feed or webcam stream
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error opening video source: {source}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run TS-YOLO inference
        results = model(frame)

        # Draw flame/smoke bounding boxes and render output
        rendered_frame = results.render()[0]

        # Display output window
        cv2.imshow('TS-YOLO Flame/Smoke Detection', rendered_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()
