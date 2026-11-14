from flask import Flask, Response
import cv2

# Flash application copy
app = Flask(__name__)

# Global variable to which main.py images will be leaked
latest_frame = None  # type: ignore


def _mjpeg_generator():
    'Infinitely generates MJPEG-stream from global latest_frame.'
    global latest_frame
    while True:
        if latest_frame is None:
            # A little delay to avoid loading the CPU if the frame is missing
            continue
        # Code the frame in JPEG
        ret, buf = cv2.imencode('.jpg', latest_frame)
        if not ret:
            continue
        frame = buf.tobytes()
        # Give the frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video_feed():
    'HTTP-endpoint to view video: http://<host>:8000/video'
    return Response(_mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run(host: str = '0.0.0.0', port: int = 8000):
    'Flask server start-up (in separate stream).'
    # threaded=True Allows both flow and human resources to be maintained
    app.run(host=host, port=port, threaded=True, debug=False) 