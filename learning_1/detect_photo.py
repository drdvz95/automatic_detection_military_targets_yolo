from ultralytics import YOLO
import cv2
import os

def detect_on_image(image_path):
    model = YOLO('best.pt')

    img = cv2.imread(image_path)
    if img is None:
        print(f"fail: {image_path}")
        return

    results = model(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            label = f'{class_name} {confidence:.2f}'

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLO Military Vehicles Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.makedirs('test_images', exist_ok=True)
    detect_on_image('test_images/counter.jpg')