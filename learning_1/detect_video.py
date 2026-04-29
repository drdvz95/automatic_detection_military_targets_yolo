from ultralytics import YOLO
import cv2
import os
import argparse


def detect_on_video(video_path, output_path=None, conf_threshold=0.5):
    """
    Детектирует военную технику на видео с помощью обученной модели YOLO.

    Args:
        video_path (str): Путь к исходному видеофайлу.
        output_path (str, optional): Путь для сохранения обработанного видео.
                                     Если не указан, результат будет просто показан на экране.
        conf_threshold (float): Порог уверенности для отображения рамки (от 0 до 1).
    """
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        print(f"fal: {model_path}")
        return
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"fail cant open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Результат будет сохранен в: {output_path}")

    print(f"Начинаю обработку видео: {video_path}")
    print(f"Разрешение: {frame_width}x{frame_height}, FPS: {fps}")
    print("Нажми 'q' для выхода из режима просмотра.")

    frame_count = 0

#цикл обработки видео
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Видео обработано полностью или произошла ошибка чтения.")
            break

        frame_count += 1
        # Детекция на текущем кадре
        results = model(frame, conf=conf_threshold, verbose=False)

        #результаты детекции
        for result in results:
            for box in result.boxes:
                #Координаты
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                #Класс и уверенность
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])

                #подпись
                label = f'{class_name} {confidence:.2f}'

                #прямоугольник и подпись
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #сохранение в выходной файл
        if out:
            out.write(frame)

        #вывод в реальном времени
        cv2.imshow('YOLO Military Vehicles Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Обработка прервана пользователем.")
            break

    #очистка
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"Обработка завершена. Всего кадров: {frame_count}")


if __name__ == '__main__':
    #парсер аргументов для гибкости
    parser = argparse.ArgumentParser(description='Детекция военной техники на видео.')
    parser.add_argument('--video', type=str, required=True, help='test_videos/Test6.mp4')
    parser.add_argument('--output', type=str, default=None,
                        help='Путь для сохранения обработанного видео (необязательно).')
    parser.add_argument('--conf', type=float, default=0.8, help='Порог уверенности для детекции (по умолчанию 0.5).')

    args = parser.parse_args()

    detect_on_video(args.video, args.output, args.conf)