import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import numpy as np
from collections import deque
import pyautogui
import math
from threading import Thread, Lock
import argparse
import collections
import json
import os
import threading
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    tk = None
    messagebox = None
try:
    import keyboard
except Exception:
    keyboard = None


# Open the webcam:
cam = cv2.VideoCapture(2)

# --- Performance / behaviour tunables ---
# Target capture resolution (lower -> faster inference)
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
# Process one of every N frames (1 = every frame)
FRAME_SKIP = 2
# Disable drawing for highest throughput
DRAW = True

# Gesture tuning defaults (can be changed via options menu)
MIN_HAND_SIZE = 0.05
PINCH_REL_THRESH = 0.20
EXT_REL_THRESH = 0.10
GESTURE_DEBOUNCE = 4         # consecutive frames required
GESTURE_COOLDOWN_MS = 500    # ms between triggers

# config file
CONFIG_PATH = 'gesture_config.json'

# Apply target resolution to capture device (best-effort)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

# Get the width and height of the frames (after setting)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define Codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Import necessary MediaPipe classes:
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a hand landmarker instance (synchronous IMAGE mode):
def draw_landmarks_on_image(frame_np: np.ndarray, result: HandLandmarkerResult):
    annotated_image = frame_np.copy()
    if result is None:
        return annotated_image
    if not result.hand_landmarks:
        return annotated_image

    for hand_landmarks in result.hand_landmarks:
        # Support both: objects with .landmark and plain lists of landmarks
        points = getattr(hand_landmarks, "landmark", hand_landmarks)

        for lm in points:
            # lm can be an object with .x/.y, a dict, or a (x,y,...) sequence
            if hasattr(lm, "x") and hasattr(lm, "y"):
                lx, ly = lm.x, lm.y
            elif isinstance(lm, dict) and "x" in lm and "y" in lm:
                lx, ly = lm["x"], lm["y"]
            elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
                lx, ly = lm[0], lm[1]
            else:
                # unknown landmark format; skip
                continue

            x = int(lx * frame_width)
            y = int(ly * frame_height)
            if DRAW:
                cv2.circle(annotated_image, (x, y), 3, (128, 0, 255), -1)

    return annotated_image

def landmark_point(lm):
    if hasattr(lm, "x") and hasattr(lm, "y"):
        return lm.x, lm.y
    if isinstance(lm, (list, tuple)) and len(lm) >= 2:
        return lm[0], lm[1]
    if isinstance(lm, dict) and "x" in lm and "y" in lm:
        return lm["x"], lm["y"]
    return None

def classify_gesture(pts):
    # pts: list-like of 21 normalized landmarks
    if not pts or len(pts) < 21:
        return None

    # build Nx2 float array of (x,y); tolerate objects/dicts/tuples
    coords = []
    for p in pts:
        lp = landmark_point(p)
        if lp is None:
            return None
        coords.append(lp)
    arr = np.asarray(coords, dtype=np.float32)

    wrist = arr[0]

    xs = arr[:, 0]
    ys = arr[:, 1]
    # bounding-box diagonal (normalized)
    bbox_diag = np.hypot(xs.max() - xs.min(), ys.max() - ys.min())
    middle_mcp = arr[9]
    wrist_mid = np.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
    hand_size = max(float(bbox_diag), float(wrist_mid))

    if hand_size < MIN_HAND_SIZE:
        return None

    # fingertip indices
    tips_idx = (4, 8, 12, 16, 20)
    tips = arr[list(tips_idx)]

    # pinch check (thumb-index)
    thumb_index_dist = np.hypot(tips[0, 0] - tips[1, 0], tips[0, 1] - tips[1, 1])
    if (thumb_index_dist / hand_size) < PINCH_REL_THRESH:
        return "pinch"

    # extension test: compare tip vs pip distances to wrist
    pip_indices = (6, 10, 14, 18)
    tip_dist = np.linalg.norm(arr[[8, 12, 16, 20]] - wrist, axis=1)
    pip_dist = np.linalg.norm(arr[list(pip_indices)] - wrist, axis=1)
    diff_rel = (tip_dist - pip_dist) / hand_size
    extended = int((diff_rel > EXT_REL_THRESH).sum())

    if extended >= 3:
        return "open"
    if extended == 0:
        return "fist"
    return None


# simple key mapping (edge-trigger: send once when gesture changes)
gesture_to_key = {"pinch": "space", "open": "up", "fist": "down"}
last_triggered = None

# --- cursor control settings ---
import pyautogui as _pga  # local alias to avoid name clash
screen_w, screen_h = _pga.size()
_pga.FAILSAFE = False  # optional: disable corner-escape if you find it annoying

cursor_x = None
cursor_y = None
SMOOTHING = 1       # 0..1, higher = smoother (slower)
CLICK_BY_PINCH = True   # hold mouse when "pinch" detected
click_down = False

latest_result = [None]
result_lock = Lock()
latest_result_frame = [None]

# simple backpressure: allow only one in-flight inference at a time
pending_inference = [False]

# buffer for frames keyed by timestamp (ms) so callback can attach results to correct frame
frames_buffer = {}
frame_ts_queue = collections.deque()
frames_lock = Lock()
MAX_BUFFER = 6

# config lock for GUI thread-safe updates
config_lock = Lock()

# gestures paused flag and inference pause
gestures_paused = [False]
pause_inference = [False]
request_open_console_menu = [False]
request_quit = [False]
settings_window = {'root': None, 'thread': None}
request_toggle_settings = [False]
 
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            return cfg
        except Exception:
            pass
    # defaults
    return {
        'gesture_to_key': {'pinch': 'space', 'open': 'up', 'fist': 'down'},
        'pinch_rel_thresh': PINCH_REL_THRESH,
        'ext_rel_thresh': EXT_REL_THRESH,
        'min_hand_size': MIN_HAND_SIZE,
        'gesture_debounce': GESTURE_DEBOUNCE,
        'gesture_cooldown_ms': GESTURE_COOLDOWN_MS,
    }

def save_config(cfg):
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception:
        return False

# load configuration
_cfg = load_config()
gesture_to_key = _cfg.get('gesture_to_key', {'pinch':'space','open':'up','fist':'down'})
PINCH_REL_THRESH = float(_cfg.get('pinch_rel_thresh', PINCH_REL_THRESH))
EXT_REL_THRESH = float(_cfg.get('ext_rel_thresh', EXT_REL_THRESH))
MIN_HAND_SIZE = float(_cfg.get('min_hand_size', MIN_HAND_SIZE))
GESTURE_DEBOUNCE = int(_cfg.get('gesture_debounce', GESTURE_DEBOUNCE))
GESTURE_COOLDOWN_MS = int(_cfg.get('gesture_cooldown_ms', GESTURE_COOLDOWN_MS))

def open_settings_gui():
    if tk is None:
        print('Tkinter not available; cannot open settings GUI')
        return
    # create a settings window in this thread; return root so caller can control it
    root = tk.Tk()
    root.title('Gesture Settings')

    row = 0
    tk.Label(root, text='Gesture mappings').grid(row=row, column=0, columnspan=2)
    row += 1
    entries = {}
    for g in ['pinch','open','fist']:
        tk.Label(root, text=g).grid(row=row, column=0, sticky='w')
        e = tk.Entry(root)
        e.insert(0, gesture_to_key.get(g, ''))
        e.grid(row=row, column=1)
        entries[g] = e
        row += 1

    tk.Label(root, text='PINCH_REL_THRESH').grid(row=row, column=0, sticky='w'); e_pin = tk.Entry(root); e_pin.insert(0, str(PINCH_REL_THRESH)); e_pin.grid(row=row, column=1); row += 1
    tk.Label(root, text='EXT_REL_THRESH').grid(row=row, column=0, sticky='w'); e_ext = tk.Entry(root); e_ext.insert(0, str(EXT_REL_THRESH)); e_ext.grid(row=row, column=1); row += 1
    tk.Label(root, text='MIN_HAND_SIZE').grid(row=row, column=0, sticky='w'); e_min = tk.Entry(root); e_min.insert(0, str(MIN_HAND_SIZE)); e_min.grid(row=row, column=1); row += 1
    tk.Label(root, text='GESTURE_DEBOUNCE').grid(row=row, column=0, sticky='w'); e_deb = tk.Entry(root); e_deb.insert(0, str(GESTURE_DEBOUNCE)); e_deb.grid(row=row, column=1); row += 1
    tk.Label(root, text='GESTURE_COOLDOWN_MS').grid(row=row, column=0, sticky='w'); e_cd = tk.Entry(root); e_cd.insert(0, str(GESTURE_COOLDOWN_MS)); e_cd.grid(row=row, column=1); row += 1

    pause_var = tk.BooleanVar(value=pause_inference[0])
    def toggle_pause():
        pause_inference[0] = pause_var.get()
        # persist pause state
        _cfg['pause_inference'] = bool(pause_inference[0])
        save_config(_cfg)
    tk.Checkbutton(root, text='Pause inference', variable=pause_var, command=toggle_pause).grid(row=row, column=0, columnspan=2)
    row += 1

    def apply_settings():
        try:
            with config_lock:
                for g in ['pinch','open','fist']:
                    gesture_to_key[g] = entries[g].get().strip()
                _cfg['gesture_to_key'] = gesture_to_key
                _cfg['pinch_rel_thresh'] = float(e_pin.get().strip())
                _cfg['ext_rel_thresh'] = float(e_ext.get().strip())
                _cfg['min_hand_size'] = float(e_min.get().strip())
                _cfg['gesture_debounce'] = int(e_deb.get().strip())
                _cfg['gesture_cooldown_ms'] = int(e_cd.get().strip())
                globals()['PINCH_REL_THRESH'] = float(_cfg['pinch_rel_thresh'])
                globals()['EXT_REL_THRESH'] = float(_cfg['ext_rel_thresh'])
                globals()['MIN_HAND_SIZE'] = float(_cfg['min_hand_size'])
                globals()['GESTURE_DEBOUNCE'] = int(_cfg['gesture_debounce'])
                globals()['GESTURE_COOLDOWN_MS'] = int(_cfg['gesture_cooldown_ms'])
                save_config(_cfg)
            messagebox.showinfo('Settings', 'Applied')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def save_and_close():
        apply_settings()
        root.destroy()

    tk.Button(root, text='Apply', command=apply_settings).grid(row=row, column=0)
    tk.Button(root, text='Save & Close', command=save_and_close).grid(row=row, column=1)

    return root

# (Tkinter settings window removed - reverted per user request)

def _result_callback(result, output_image, timestamp_ms):
    # called from MediaPipe worker thread in LIVE_STREAM mode
    # attach the result and, if we have the matching frame, store it for immediate rendering
    with frames_lock:
        frame = frames_buffer.pop(timestamp_ms, None)
        try:
            frame_ts_queue.remove(timestamp_ms)
        except Exception:
            pass
    with result_lock:
        latest_result[0] = result
    if frame is not None:
        latest_result_frame[0] = frame
    # mark inference complete
    pending_inference[0] = False

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r'hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=_result_callback,
)
def _capture_thread(latest, lock, stop_flag):
    while not stop_flag[0]:
        ret, frame = cam.read()
        if not ret:
            continue
        with lock:
            latest[0] = frame


def main_loop(landmarker, args):
    global click_down, cursor_x, cursor_y, last_triggered
    latest_frame = [None]
    stop_flag = [False]
    cap_lock = Lock()
    t = Thread(target=_capture_thread, args=(latest_frame, cap_lock, stop_flag), daemon=True)
    t.start()

    frame_counter = 0
    # gesture debounce state
    candidate_gesture = None
    candidate_count = 0
    last_trigger_time = 0

    def open_options_menu():
        nonlocal candidate_gesture, candidate_count
        print('\n=== Gesture Options Menu ===')
        print('Current mappings:')
        for g, k in gesture_to_key.items():
            print(f' - {g}: {k}')
        print('\nThresholds:')
        print(f' PINCH_REL_THRESH: {PINCH_REL_THRESH}')
        print(f' EXT_REL_THRESH: {EXT_REL_THRESH}')
        print(f' MIN_HAND_SIZE: {MIN_HAND_SIZE}')
        print(f' GESTURE_DEBOUNCE: {GESTURE_DEBOUNCE}')
        print(f' GESTURE_COOLDOWN_MS: {GESTURE_COOLDOWN_MS}')
        print('\nCommands:')
        print(' 1) Remap gesture key')
        print(' 2) Change thresholds')
        print(' 3) Save config')
        print(' 4) Reset defaults')
        print(' 0) Cancel')
        try:
            choice = input('Select option: ').strip()
        except Exception:
            return
        if choice == '1':
            g = input('Gesture name (pinch/open/fist): ').strip()
            if g not in gesture_to_key:
                print('Unknown gesture')
            else:
                k = input('Key to send (pyautogui key name): ').strip()
                gesture_to_key[g] = k
                _cfg['gesture_to_key'] = gesture_to_key
                print('Updated')
        elif choice == '2':
            try:
                p = float(input(f'PINCH_REL_THRESH (current {PINCH_REL_THRESH}): ').strip())
                e = float(input(f'EXT_REL_THRESH (current {EXT_REL_THRESH}): ').strip())
                m = float(input(f'MIN_HAND_SIZE (current {MIN_HAND_SIZE}): ').strip())
                d = int(input(f'GESTURE_DEBOUNCE (current {GESTURE_DEBOUNCE}): ').strip())
                c = int(input(f'GESTURE_COOLDOWN_MS (current {GESTURE_COOLDOWN_MS}): ').strip())
                _cfg['pinch_rel_thresh'] = p
                _cfg['ext_rel_thresh'] = e
                _cfg['min_hand_size'] = m
                _cfg['gesture_debounce'] = d
                _cfg['gesture_cooldown_ms'] = c
                # apply
                globals()['PINCH_REL_THRESH'] = p
                globals()['EXT_REL_THRESH'] = e
                globals()['MIN_HAND_SIZE'] = m
                globals()['GESTURE_DEBOUNCE'] = d
                globals()['GESTURE_COOLDOWN_MS'] = c
                print('Updated thresholds')
            except Exception as ex:
                print('Invalid input', ex)
        elif choice == '3':
            _cfg['gesture_to_key'] = gesture_to_key
            save_config(_cfg)
            print('Saved to', CONFIG_PATH)
        elif choice == '4':
            if os.path.exists(CONFIG_PATH):
                os.remove(CONFIG_PATH)
            print('Config reset; restart to load defaults')
        else:
            print('Cancelled')

    try:
        while True:
            frame_counter += 1
            with cap_lock:
                frame = None if latest_frame[0] is None else latest_frame[0].copy()
            if frame is None:
                time.sleep(0.005)
                continue

            # Skip processing some frames to reduce CPU/model load
            if FRAME_SKIP > 1 and (frame_counter % FRAME_SKIP) != 0:
                # still show preview but skip inference/draw (cheap)
                if DRAW:
                    small = cv2.resize(frame, (frame_width, frame_height))
                    cv2.imshow('Webcam', small)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # OpenCV gives BGR; convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # enqueue frame for async processing (LIVE_STREAM)
            ts_ms = int(time.time() * 1000)
            # If inference is paused, or an inference is already pending, drop this frame (backpressure)
            if pause_inference[0] or pending_inference[0]:
                # show a small preview for responsiveness and skip enqueue
                if DRAW:
                    small = cv2.resize(frame, (frame_width, frame_height))
                    cv2.imshow('Webcam', small)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # store a copy of the frame associated with this timestamp so the callback can render
            with frames_lock:
                frames_buffer[ts_ms] = frame.copy()
                frame_ts_queue.append(ts_ms)
                # keep bounded buffer size
                while len(frame_ts_queue) > MAX_BUFFER:
                    old = frame_ts_queue.popleft()
                    frames_buffer.pop(old, None)

            try:
                pending_inference[0] = True
                landmarker.detect_async(mp_image, ts_ms)
            except Exception:
                # fall back to synchronous video detect if async not available
                res = landmarker.detect_for_video(mp_image, ts_ms)
                with result_lock:
                    latest_result[0] = res
                # synchronous fallback finished immediately
                pending_inference[0] = False

            # draw the most recent result delivered by the callback; prefer the matched frame
            with result_lock:
                snapshot = latest_result[0]
                snapshot_frame = latest_result_frame[0]
            render_frame = snapshot_frame if snapshot_frame is not None else frame
            annotation = draw_landmarks_on_image(render_frame, snapshot)
            if DRAW:
                cv2.imshow('Webcam', annotation)
            k = cv2.waitKey(1) & 0xFF
            # local key handling (window must be focused) still supported
            if k == ord('q'):
                break
            if k == ord('o') or k == ord('m'):
                request_open_console_menu[0] = True
            if k == ord('s'):
                # toggle settings window thread-safe
                if settings_window['root'] is None:
                    # start a thread to create and run the Tk window
                    def _start_settings():
                        try:
                            root = open_settings_gui()
                            settings_window['root'] = root
                            settings_window['thread'] = threading.current_thread()
                            if root is not None:
                                root.mainloop()
                        finally:
                            settings_window['root'] = None
                            settings_window['thread'] = None
                    th = Thread(target=_start_settings, daemon=True)
                    th.start()
                else:
                    # close existing window
                    try:
                        settings_window['root'].destroy()
                    except Exception:
                        pass
            # handle requests from global hotkey listener
            if request_toggle_settings[0]:
                request_toggle_settings[0] = False
                if settings_window['root'] is None:
                    def _start_settings():
                        try:
                            root = open_settings_gui()
                            settings_window['root'] = root
                            settings_window['thread'] = threading.current_thread()
                            if root is not None:
                                root.mainloop()
                        finally:
                            settings_window['root'] = None
                            settings_window['thread'] = None
                    th = Thread(target=_start_settings, daemon=True)
                    th.start()
                else:
                    try:
                        settings_window['root'].destroy()
                    except Exception:
                        pass
            if k == ord('p'):
                pause_inference[0] = not pause_inference[0]
                _cfg['pause_inference'] = bool(pause_inference[0])
                save_config(_cfg)
                print('Inference paused' if pause_inference[0] else 'Inference resumed')

            # handle requests set by background keyboard listener
            if request_open_console_menu[0]:
                request_open_console_menu[0] = False
                open_options_menu()
            if request_quit[0]:
                break

            # process hand(s) using the latest async callback snapshot
            if snapshot and getattr(snapshot, "hand_landmarks", None):
                hands = snapshot.hand_landmarks
                if len(hands) > 0:
                    pts = getattr(hands[0], "landmark", hands[0])
                    # compute gesture once
                    gesture = classify_gesture(pts)

                    # debounce: require consecutive frames of same gesture
                    if gesture == candidate_gesture:
                        candidate_count += 1
                    else:
                        candidate_gesture = gesture
                        candidate_count = 1

                    now_ms = int(time.time() * 1000)
                    if candidate_gesture and candidate_count >= GESTURE_DEBOUNCE and (now_ms - last_trigger_time) > GESTURE_COOLDOWN_MS:
                        if not gestures_paused[0] and candidate_gesture in gesture_to_key and candidate_gesture != last_triggered:
                            print('gesture:', candidate_gesture, '->', gesture_to_key[candidate_gesture])
                            pyautogui.press(gesture_to_key[candidate_gesture])
                            last_triggered = candidate_gesture
                            last_trigger_time = now_ms

                    # --- cursor: use index fingertip (landmark 8) ---
                    coords = [landmark_point(p) for p in pts]
                    idx_tip = coords[8] if len(coords) > 8 else None
                    if idx_tip is not None:
                        target_x = idx_tip[0] * screen_w
                        target_y = idx_tip[1] * screen_h
                        if cursor_x is None:
                            cursor_x, cursor_y = target_x, target_y
                        else:
                            cursor_x = cursor_x * (1 - SMOOTHING) + target_x * SMOOTHING
                            cursor_y = cursor_y * (1 - SMOOTHING) + target_y * SMOOTHING
                        _pga.moveTo(int(cursor_x), int(cursor_y))

                    # --- click by pinch: hold mouse while pinch active (use debounced gesture) ---
                    if CLICK_BY_PINCH:
                        if candidate_gesture == 'pinch' and not click_down and candidate_count >= 1:
                            _pga.mouseDown()
                            click_down = True
                        elif candidate_gesture != 'pinch' and click_down:
                            _pga.mouseUp()
                            click_down = False

            # if no hand detected, optionally release click and/or keep last_triggered
            else:
                # release click if hand disappears while holding
                if click_down:
                    _pga.mouseUp()
                    click_down = False
    finally:
        stop_flag[0] = True
        t.join(timeout=0.5)
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true', help='enable cProfile and print stats')
    args = parser.parse_args()

    with HandLandmarker.create_from_options(options) as landmarker:
        if args.profile:
            import cProfile, pstats, io
            pr = cProfile.Profile()
            pr.enable()
            main_loop(landmarker, args)
            pr.disable()
            pr.dump_stats('stats.prof')
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(40)
            print(s.getvalue())
        else:
            # run main_loop in the main thread (keeps keyboard handling local),
            # and allow the user to toggle settings window with 's'.
            main_loop(landmarker, args)