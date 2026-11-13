from flask import Flask, Response
import cv2

# Экземпляр Flask-приложения
app = Flask(__name__)

# Глобальная переменная, в которую будет прокидывать кадры main.py
latest_frame = None  # type: ignore


def _mjpeg_generator():
    """Бесконечно генерирует MJPEG-стрим из global latest_frame."""
    global latest_frame
    while True:
        if latest_frame is None:
            # Небольшая задержка, чтобы не нагружать CPU, если кадра нет
            continue
        # Кодируем кадр в JPEG
        ret, buf = cv2.imencode('.jpg', latest_frame)
        if not ret:
            continue
        frame = buf.tobytes()
        # Отдаём кадр в формате multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video_feed():
    """HTTP-эндпоинт для просмотра видео: http://<host>:8000/video"""
    return Response(_mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run(host: str = '0.0.0.0', port: int = 8000):
    """Запуск Flask-сервера (вызывается в отдельном потоке)."""
    # threaded=True позволяет одновременно обслуживать поток и получать кадры
    app.run(host=host, port=port, threaded=True, debug=False) 