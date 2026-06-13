#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro


NOTA
----
O seguinte comando é para exportar o modelo YOLOv8s para ONNX.
Execute no terminal:

uv run yolo export model=yolov8s.pt format=onnx
"""
import io
import math
import os
import struct
import time
import urllib.error
import urllib.request
import wave

from torch import device

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import cv2
import cvzone
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_NAME = "yolov8s.onnx"  #yolov8s.onnx    yolo26x.pt  "yolov8s.pt"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DETECTOR_MODEL = os.path.join(BASE_DIR, "models", "blaze_face_short_range.tflite")
FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
FACE_DETECTOR_MIN_BYTES = 100_000
CAMERA_INDEXES = [0, 1]
FRAME_WIDTH = 1500  # 1280
FRAME_HEIGHT = 920  # 720
PERSON_CONFIDENCE_THRESHOLD = 0.50
FACE_DETECTION_CONFIDENCE = 0.70
WARNING_TRIGGER_SECONDS = 1.5
DANGER_TRIGGER_SECONDS = 4.0
CLEAR_ALERT_SECONDS = 1.0
MAX_CONSECUTIVE_READ_FAILURES = 10
ALARM_FILENAMES = ["alarm.mp3", "alarm.wav", "alarm.ogg", "alarm.mpeg"]
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]

SAFE_STATE = "safe"
WARNING_STATE = "warning"
DANGER_STATE = "danger"


def open_camera(camera_indexes):
    for camera_index in camera_indexes:
        device_path = f"/dev/video{camera_index}"
        if not os.path.exists(device_path):
            continue

        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            continue

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return camera, camera_index

    return None, None


def get_alert_state(elapsed_time):
    if elapsed_time >= DANGER_TRIGGER_SECONDS:
        return DANGER_STATE
    if elapsed_time >= WARNING_TRIGGER_SECONDS:
        return WARNING_STATE
    return SAFE_STATE


def download_face_detector_model(model_path, model_url):
    model_dir = os.path.dirname(model_path)
    temp_path = f"{model_path}.tmp"
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    print("MediaPipe Face Detector model not found. Downloading model...")
    print(f"Source: {model_url}")
    print(f"Destination: {model_path}")

    try:
        with urllib.request.urlopen(model_url, timeout=60) as response:
            with open(temp_path, "wb") as model_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    model_file.write(chunk)

        downloaded_size = os.path.getsize(temp_path)
        if downloaded_size < FACE_DETECTOR_MIN_BYTES:
            raise OSError(
                "Downloaded model is smaller than expected "
                f"({downloaded_size} bytes)."
            )

        os.replace(temp_path, model_path)
        print("MediaPipe Face Detector model downloaded successfully.")
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise SystemExit(
            "Unable to download the MediaPipe Face Detector model.\n"
            f"Expected file: {model_path}\n"
            f"Download URL: {model_url}\n"
            "Manual fallback:\n"
            f"  mkdir -p {model_dir}\n"
            f'  curl -L "{model_url}" -o "{model_path}"\n'
            f"Original error: {exc}"
        ) from exc


def load_face_detector(model_path):
    if not os.path.exists(model_path):
        download_face_detector_model(model_path, FACE_DETECTOR_MODEL_URL)

    if not os.path.exists(model_path):
        raise SystemExit(
            "MediaPipe Face Detector model not found. " f"Expected file: {model_path}"
        )

    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=FACE_DETECTION_CONFIDENCE,
    )
    return vision.FaceDetector.create_from_options(options)


def find_alarm_file():
    search_directories = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)

    for directory in (current_dir, project_dir, os.getcwd()):
        if directory not in search_directories:
            search_directories.append(directory)

    for directory in search_directories:
        for filename in ALARM_FILENAMES:
            candidate = os.path.join(directory, filename)
            if os.path.exists(candidate):
                return candidate

    return None


def create_fallback_alarm_sound():
    sample_rate = 22050
    duration_seconds = 0.45
    amplitude = 12000
    frequencies = (880, 660)
    total_samples = int(sample_rate * duration_seconds)

    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            frames = bytearray()
            for sample_index in range(total_samples):
                time_position = sample_index / sample_rate
                frequency = frequencies[
                    (sample_index // (sample_rate // 6)) % len(frequencies)
                ]
                sample_value = int(
                    amplitude * math.sin(2 * math.pi * frequency * time_position)
                )
                frames.extend(struct.pack("<h", sample_value))

            wav_file.writeframes(bytes(frames))

        wav_buffer.seek(0)
        return pygame.mixer.Sound(file=wav_buffer)


def load_pillow_font(size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_font_candidates = [
        os.path.join(current_dir, "assets", "fonts", "DejaVuSans-Bold.ttf"),
        os.path.join(current_dir, "assets", "fonts", "DejaVuSans.ttf"),
    ]

    for font_path in [*local_font_candidates, *FONT_CANDIDATES]:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue

    return ImageFont.load_default()


def get_text_size(font, text):
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def clamp_point(point, frame_width, frame_height):
    x, y = point
    return max(0, min(x, frame_width - 1)), max(0, min(y, frame_height - 1))


def add_text_overlay(
    overlays,
    text,
    position,
    font,
    text_color,
    background_color=None,
    padding=(12, 8),
):
    overlays.append(
        {
            "text": text,
            "position": position,
            "font": font,
            "text_color": text_color,
            "background_color": background_color,
            "padding": padding,
        }
    )


def draw_label_background(img, overlay):
    text = overlay["text"]
    font = overlay["font"]
    position = overlay["position"]
    background_color = overlay["background_color"]
    padding_x, padding_y = overlay["padding"]
    text_width, text_height = get_text_size(font, text)
    frame_height, frame_width = img.shape[:2]

    x, y = clamp_point(position, frame_width, frame_height)
    x2 = min(frame_width - 1, x + text_width + (padding_x * 2))
    y2 = min(frame_height - 1, y + text_height + (padding_y * 2))

    cv2.rectangle(img, (x, y), (x2, y2), background_color, -1)
    overlay["text_position"] = (x + padding_x, y + padding_y)


def render_text_overlays(img, overlays):
    if not overlays:
        return img

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)
    frame_height, frame_width = img.shape[:2]

    for overlay in overlays:
        x, y = clamp_point(overlay["position"], frame_width, frame_height)
        text_position = overlay.get("text_position", (x, y))
        draw.text(
            text_position,
            overlay["text"],
            font=overlay["font"],
            fill=overlay["text_color"],
        )

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


cap, active_camera_index = open_camera(CAMERA_INDEXES)
if cap is None:
    checked_indexes = ", ".join(str(index) for index in CAMERA_INDEXES)
    raise SystemExit(
        f"Unable to open any configured camera. Checked indexes: {checked_indexes}."
    )

model = YOLO(MODEL_NAME, task="detect")
LABEL_FONT = load_pillow_font(size=28)
BANNER_FONT = load_pillow_font(size=30)

# --- DETECTORS ---
face_detector = load_face_detector(FACE_DETECTOR_MODEL)

# --- AUDIO SETUP ---
audio_available = False
alarm_file_path = find_alarm_file()
alarm_sound_mode = None
fallback_alarm_sound = None
try:
    pygame.mixer.init()
    if alarm_file_path:
        pygame.mixer.music.load(alarm_file_path)
        audio_available = True
        alarm_sound_mode = "music"
    else:
        fallback_alarm_sound = create_fallback_alarm_sound()
        audio_available = True
        alarm_sound_mode = "fallback"
except pygame.error as exc:
    if alarm_file_path:
        try:
            fallback_alarm_sound = create_fallback_alarm_sound()
            audio_available = True
            alarm_sound_mode = "fallback"
        except pygame.error:
            print(f"Audio alarm unavailable: {exc}")
    else:
        print(f"Audio alarm unavailable: {exc}")

# --- ALERT STATE ---
anomaly_start_time = None
anomaly_clear_start_time = None
alert_state = SAFE_STATE
held_alert_box = None
consecutive_read_failures = 0
face_detection_timestamp_ms = 0

print(
    "Starting Facial Surveillance System "
    f"(camera {active_camera_index}, warning={WARNING_TRIGGER_SECONDS:.1f}s, "
    f"danger={DANGER_TRIGGER_SECONDS:.1f}s, clear={CLEAR_ALERT_SECONDS:.1f}s)..."
)

while True:
    success, img = cap.read()
    if not success:
        consecutive_read_failures += 1
        if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
            print(
                "Camera read failed repeatedly. "
                "Check the selected device or try another camera index."
            )
            break
        continue

    consecutive_read_failures = 0
    now = time.time()
    results = model(img,
                         stream=True,
                         classes=[0],
                         verbose=False,
                         device="0" # cpu
                         )

    current_frame_has_anomaly = False
    current_frame_box = None
    current_anomaly_box = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3]),
            )
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf <= PERSON_CONFIDENCE_THRESHOLD:
                continue

            if current_frame_box is None:
                current_frame_box = (x1, y1, w, h)

            y1_c, y2_c = max(0, y1), min(img.shape[0], y2)
            x1_c, x2_c = max(0, x1), min(img.shape[1], x2)
            person_roi = img[y1_c:y2_c, x1_c:x2_c]

            if person_roi.size == 0:
                continue

            rgb_person_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            mp_person_roi = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_person_roi,
            )
            face_detection_timestamp_ms = max(
                int(time.perf_counter() * 1000),
                face_detection_timestamp_ms + 1,
            )
            face_results = face_detector.detect_for_video(
                mp_person_roi,
                face_detection_timestamp_ms,
            )
            is_face_visible = bool(face_results.detections)

            if not is_face_visible:
                current_frame_has_anomaly = True
                if current_anomaly_box is None:
                    current_anomaly_box = (x1, y1, w, h)

    if current_frame_has_anomaly:
        if anomaly_start_time is None:
            anomaly_start_time = now
        anomaly_clear_start_time = None
        held_alert_box = current_anomaly_box or held_alert_box
        elapsed_time = now - anomaly_start_time
    elif anomaly_start_time is not None:
        if anomaly_clear_start_time is None:
            anomaly_clear_start_time = now

        elapsed_time = anomaly_clear_start_time - anomaly_start_time
        clear_elapsed = now - anomaly_clear_start_time

        if clear_elapsed >= CLEAR_ALERT_SECONDS:
            anomaly_start_time = None
            anomaly_clear_start_time = None
            held_alert_box = None
            elapsed_time = 0.0
    else:
        elapsed_time = 0.0

    alert_state = get_alert_state(elapsed_time)
    alarm_active = alert_state == DANGER_STATE
    text_overlays = []

    if alert_state == DANGER_STATE:
        target_box = current_anomaly_box or held_alert_box
        if target_box:
            x1, y1, w, h = target_box
            cvzone.cornerRect(
                img,
                (x1, y1, w, h),
                l=30,
                t=5,
                rt=1,
                colorR=(0, 0, 255),
                colorC=(0, 0, 255),
            )
            add_text_overlay(
                text_overlays,
                text="SECURITY VIOLATION DETECTED",
                position=(max(0, x1), max(35, y1)),
                font=LABEL_FONT,
                text_color=(255, 255, 255),
                background_color=(0, 0, 255),
                padding=(12, 8),
            )
            draw_label_background(img, text_overlays[-1])

        cv2.rectangle(img, (0, 0), (FRAME_WIDTH, 50), (0, 0, 255), -1)
        add_text_overlay(
            text_overlays,
            text="ALARM: FACE COVERAGE DETECTED",
            position=(300, 10),
            font=BANNER_FONT,
            text_color=(255, 255, 255),
        )
    elif alert_state == WARNING_STATE:
        target_box = current_anomaly_box or held_alert_box
        time_left = max(0, math.ceil(DANGER_TRIGGER_SECONDS - elapsed_time))

        if target_box:
            x1, y1, w, h = target_box
            cvzone.cornerRect(
                img,
                (x1, y1, w, h),
                l=30,
                t=5,
                rt=1,
                colorR=(0, 255, 255),
                colorC=(0, 255, 255),
            )
            add_text_overlay(
                text_overlays,
                text=f"WARNING: SHOW YOUR FACE ({time_left}s)",
                position=(max(0, x1), max(35, y1)),
                font=LABEL_FONT,
                text_color=(0, 0, 0),
                background_color=(0, 255, 255),
                padding=(12, 8),
            )
            draw_label_background(img, text_overlays[-1])

        cv2.rectangle(img, (0, 0), (FRAME_WIDTH, 50), (0, 255, 255), -1)
        add_text_overlay(
            text_overlays,
            text=f"WARNING: SHOW YOUR FACE ({time_left}s)",
            position=(300, 10),
            font=BANNER_FONT,
            text_color=(0, 0, 0),
        )
    else:
        if current_frame_box:
            x1, y1, w, h = current_frame_box
            cvzone.cornerRect(
                img,
                (x1, y1, w, h),
                l=30,
                t=5,
                rt=1,
                colorR=(0, 255, 0),
                colorC=(0, 255, 0),
            )
            add_text_overlay(
                text_overlays,
                text="SAFE: FACE VERIFIED",
                position=(max(0, x1), max(35, y1)),
                font=LABEL_FONT,
                text_color=(255, 255, 255),
                background_color=(0, 255, 0),
                padding=(12, 8),
            )
            draw_label_background(img, text_overlays[-1])

    if audio_available:
        if alarm_active:
            if alarm_sound_mode == "music":
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
                    print("Alarm sounding.")
            elif alarm_sound_mode == "fallback" and fallback_alarm_sound is not None:
                if pygame.mixer.get_busy() is False:
                    fallback_alarm_sound.play(loops=-1)
                    print("Alarm sounding.")
        elif alarm_sound_mode == "music":
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        elif alarm_sound_mode == "fallback":
            pygame.mixer.stop()

    img = render_text_overlays(img, text_overlays)
    cv2.imshow("Facial Surveillance - Temporal Logic", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

face_detector.close()
cap.release()
cv2.destroyAllWindows()
