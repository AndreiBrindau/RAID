import cv2
import sys
from ultralytics import YOLO

def main():
    
    model_path = "yolov8n.pt" 
    
    print(f"[INFO] Initializing test framework using model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access your PC's webcam.")
        sys.exit(1)

    print("\n=======================================================")
    print("RAID AI LOCAL TEST ONLINE: Point your webcam at objects!")
    print("Press the 'q' key in the video window to stop.")
    print("=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame from webcam.")
            continue

        # Run the base AI model
        results = model(frame, conf=0.25, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf) * 100
                label_name = model.names[class_id].upper()
                display_text = f"{label_name} ({confidence:.1f}%)"

                box_color = (0, 0, 255) 

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                pad_x = 8
                pad_y = 6
                label_x1 = max(x1, 0)
                label_y1 = max(y1 - text_h - pad_y * 2, 0)
                label_x2 = min(label_x1 + text_w + pad_x * 2, frame.shape[1])
                label_y2 = min(label_y1 + text_h + pad_y * 2, frame.shape[0])

                cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), box_color, -1)
                cv2.putText(frame, display_text, (label_x1 + pad_x, label_y2 - pad_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the video feed window right on your laptop desktop
        cv2.imshow("RAID AI LOCAL PC TEST", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
