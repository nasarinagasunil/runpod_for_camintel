
import cv2
import numpy as np
import torch
from pathlib import Path
from scipy.signal import savgol_filter
from collections import defaultdict, deque
from ultralytics import YOLO
from boxmot.trackers.botsort.botsort import BotSort
from app.ml.person_identification import slow_lane_pipeline

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Detection parameters
    YOLO_MODEL = 'yolov8n-pose.pt'
    PERSON_CONF_THRESHOLD = 0.4
    KEYPOINT_CONF_THRESHOLD = 0.3
    MIN_VISIBLE_KEYPOINTS = 5

    # Bbox filtering
    MIN_BBOX_AREA_RATIO = 0.002
    MAX_BBOX_AREA_RATIO = 0.8
    MIN_ASPECT_RATIO = 0.5
    MAX_ASPECT_RATIO = 5.0

    # Tracking parameters
    TRACK_HIGH_THRESH = 0.5
    TRACK_LOW_THRESH = 0.1
    NEW_TRACK_THRESH = 0.6
    TRACK_BUFFER = 30
    MATCH_THRESH = 0.8

    # Smoothing parameters
    SMOOTHING_WINDOW = 7
    SMOOTHING_POLYORDER = 2
    SMOOTHING_HISTORY_SIZE = 30

# ============================================================================
# Helper Functions
# ============================================================================
def extract_keypoints(result):
    try:
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return np.empty((0, 17, 3))
        kps = result.keypoints.data.cpu().numpy()
        if kps.ndim != 3 or kps.shape[1] != 17 or kps.shape[2] != 3:
            return np.empty((0, 17, 3))
        return kps
    except Exception as e:
        print(f"⚠️ Keypoint extraction failed: {e}")
        return np.empty((0, 17, 3))

# ============================================================================
# Visualization
# ============================================================================
KEYPOINT_EDGES = [
    (0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), (7,9), (6,8), (8,10),
    (5,6), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
]

def draw_keypoints(frame, keypoints_xyc, conf_threshold, color, thickness=4):
    for i in range(min(17, keypoints_xyc.shape[0])):
        x, y, c = keypoints_xyc[i]
        if c > conf_threshold:
            cv2.circle(frame, (int(x), int(y)), thickness, color, -1)

def draw_skeleton(frame, keypoints_xyc, conf_threshold, color, thickness=2):
    for a, b in KEYPOINT_EDGES:
        if a >= len(keypoints_xyc) or b >= len(keypoints_xyc):
            continue
        x1, y1, c1 = keypoints_xyc[a]
        x2, y2, c2 = keypoints_xyc[b]
        if c1 > conf_threshold and c2 > conf_threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def draw_bbox(frame, x1, y1, x2, y2, color, label=None, thickness=2):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if label:
        text_y = max(int(y1) - 10, 20)
        cv2.putText(frame, label, (int(x1), text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ============================================================================
# Keypoint Smoothing
# ============================================================================
class KeypointSmoother:
    def __init__(self, window_length=7, polyorder=2, history_size=30):
        self.window_length = window_length
        self.polyorder = polyorder
        self.history_size = history_size
        self.keypoint_histories = defaultdict(lambda: deque(maxlen=history_size))

    def add_keypoints(self, track_id, keypoints_xyc):
        self.keypoint_histories[track_id].append(keypoints_xyc.copy())

    def get_smoothed_keypoints(self, track_id):
        history = self.keypoint_histories[track_id]
        if len(history) == 0: return None
        if len(history) < 3: return history[-1]

        history_array = np.array(list(history))
        history_len = len(history)
        window = min(self.window_length, history_len)
        if window % 2 == 0: window -= 1
        window = max(3, window)

        smoothed_kps = np.zeros((17, 3))
        for kp_idx in range(17):
            trajectory = history_array[:, kp_idx, :]
            recent_confs = trajectory[-window:, 2]
            if np.sum(recent_confs > Config.KEYPOINT_CONF_THRESHOLD) < window // 2:
                smoothed_kps[kp_idx] = trajectory[-1]
                continue
            try:
                smoothed_x = savgol_filter(trajectory[:, 0], window, self.polyorder)
                smoothed_y = savgol_filter(trajectory[:, 1], window, self.polyorder)
                smoothed_kps[kp_idx, 0] = smoothed_x[-1]
                smoothed_kps[kp_idx, 1] = smoothed_y[-1]
                smoothed_kps[kp_idx, 2] = trajectory[-1, 2]
            except:
                smoothed_kps[kp_idx] = trajectory[-1]
        return smoothed_kps

    def cleanup_old_tracks(self, active_track_ids):
        all_ids = list(self.keypoint_histories.keys())
        for track_id in all_ids:
            if track_id not in active_track_ids:
                del self.keypoint_histories[track_id]

# ============================================================================
# Main Tracker Class
# ============================================================================
class BotSortTracker:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = self.device == 'cuda'
        
        # Load YOLO model
        self.model = YOLO(Config.YOLO_MODEL)
        
        # Initialize BoTSORT
        self.tracker = BotSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device=self.device,
            half=self.half,
            track_high_thresh=Config.TRACK_HIGH_THRESH,
            track_low_thresh=Config.TRACK_LOW_THRESH,
            new_track_thresh=Config.NEW_TRACK_THRESH,
            track_buffer=Config.TRACK_BUFFER,
            match_thresh=Config.MATCH_THRESH,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method='sof',
            frame_rate=30, # Default, will be updated per video
            fuse_first_associate=False,
            with_reid=True
        )
        
        self.smoother = KeypointSmoother(
            window_length=Config.SMOOTHING_WINDOW,
            polyorder=Config.SMOOTHING_POLYORDER,
            history_size=Config.SMOOTHING_HISTORY_SIZE
        )
        
        # Initialize Activity Predictor
        from app.ml.activity_recognition import ActivityPredictor
        self.activity_predictor = ActivityPredictor(
            model_path='/app/best_activity_model.pth', # Path in Docker container
            device=self.device
        )
        
        # Initialize Accident Predictor
        from app.ml.accident_detector import AccidentPredictor
        self.accident_predictor = AccidentPredictor(
            model_path='/app/best_accident_model.pth',
            device=self.device
        )
        
        self.colors = (np.random.randint(50, 255, size=(256, 3))).tolist()
        
        # Slow Lane State
        self.track_identities = {} # track_id -> name
        
        # Async Executor for Face Recognition
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.face_future = None

    def process_video(self, input_path: str, output_path: str, progress_callback=None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_area = width * height
        
        # Update tracker FPS
        self.tracker.frame_rate = int(fps)

        if output_path.endswith('.webm'):
            # VP8 codec for WebM container - widely supported in browsers
            fourcc = cv2.VideoWriter_fourcc(*'vp80')
        else:
            # Fallback to mp4v if not webm (avc1 might fail without openh264)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        unique_tracks = {}
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            if progress_callback and total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                progress_callback(progress)

            results = self.model(frame, verbose=False)
            if not results:
                out.write(frame)
                continue
                
            result = results[0]
            detections, keypoints_dict = self._process_detections_frame(result, width, height, frame_area)

            tracks = self.tracker.update(detections, frame)

            # ---------------------------------------------------------
            # SLOW LANE TRIGGER (Face Recognition) - Every ~1 sec (30 frames)
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # SLOW LANE TRIGGER (Face Recognition) - Non-Blocking
            # ---------------------------------------------------------
            
            # Check for completed face tasks
            if self.face_future is not None and self.face_future.done():
                try:
                    face_results = self.face_future.result()
                    # Update Identity Cache
                    if 'identities' in face_results:
                        for ident in face_results['identities']:
                            if ident.get('matched') and ident.get('track_id'):
                                # Store confirmed name
                                self.track_identities[ident['track_id']] = ident.get('name', 'Unknown')
                except Exception as e:
                    print(f"Face recognition task failed: {e}")
                finally:
                    self.face_future = None

            # Submit new task if time is right AND no task is running
            if frame_count % 30 == 0 and self.face_future is None:
                active_tracks_for_face = []
                if tracks.size > 0:
                    for trk in tracks:
                        tx1, ty1, tx2, ty2 = trk[:4]
                        tid = int(trk[4])
                        # Pass bbox as (x, y, w, h)
                        active_tracks_for_face.append({
                            'bbox': (tx1, ty1, tx2-tx1, ty2-ty1),
                            'track_id': tid
                        })
                
                # Run in background
                # We copy frame to ensure thread safety (though read-only is usually fine)
                frame_copy = frame.copy() 
                self.face_future = self.executor.submit(
                    slow_lane_pipeline.process_frame,
                    frame_copy, 
                    frame_count, 
                    fast_lane_tracks=active_tracks_for_face
                )

            active_ids = set()

            if tracks.size > 0:
                for track in tracks:
                    x1, y1, x2, y2 = track[:4]
                    track_id = int(track[4])
                    det_idx = int(track[7]) if len(track) > 7 else -1
                    active_ids.add(track_id)

                    if det_idx >= 0 and det_idx in keypoints_dict:
                        kps = keypoints_dict[det_idx]
                        self.smoother.add_keypoints(track_id, kps)
                        smoothed = self.smoother.get_smoothed_keypoints(track_id)

                        if smoothed is not None:
                            color = tuple(map(int, self.colors[track_id % len(self.colors)]))
                            draw_skeleton(frame, smoothed, Config.KEYPOINT_CONF_THRESHOLD, color)
                            draw_keypoints(frame, smoothed, Config.KEYPOINT_CONF_THRESHOLD, color)
                            if track_id not in unique_tracks:
                                unique_tracks[track_id] = {
                                    'track_id': track_id,
                                    'start_frame': frame_count,
                                    'end_frame': frame_count,
                                    'activity_history': []  # List of (frame_idx, activity_label)
                                }
                            unique_tracks[track_id]['end_frame'] = frame_count
                            
                            # Update activity predictor
                            self.activity_predictor.update_pose(track_id, smoothed)
                            activity, conf = self.activity_predictor.predict_activity(track_id)
                            
                            if activity:
                                unique_tracks[track_id]['activity_history'].append((frame_count, activity))
                                # Visualize activity + Name if known
                                name_label = self.track_identities.get(track_id, f"ID:{track_id}")
                                label = f"{name_label} {activity}"
                                draw_bbox(frame, x1, y1, x2, y2, color, label)
                            else:
                                name_label = self.track_identities.get(track_id, f"ID:{track_id}")
                                draw_bbox(frame, x1, y1, x2, y2, color, name_label)

                            # Accident Detection (Fall)
                            # Use RAW keypoints for detection (better for sudden impact)
                            # But visualization relies on smoothed to look nice.
                            
                            # The 'kps' variable holds the raw keypoints from this frame
                            self.accident_predictor.update_pose(track_id, kps)
                            is_accident, acc_conf = self.accident_predictor.predict_accident(track_id)
                            
                            if is_accident:
                                # Flag as incident
                                if 'accidents' not in unique_tracks[track_id]:
                                    unique_tracks[track_id]['accidents'] = 0
                                unique_tracks[track_id]['accidents'] += 1
                                
                                # RED ALERT visualisation
                                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 0, 255), 4)
                                cv2.putText(frame, "FALL DETECTED!", (ix1, iy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.smoother.cleanup_old_tracks(active_ids)
            self.activity_predictor.cleanup_old_tracks(active_ids)
            self.accident_predictor.cleanup_old_tracks(active_ids)
            out.write(frame)

        cap.release()
        out.release()
        

        
        return output_path, list(unique_tracks.values())

    def _process_detections_frame(self, result, w, h, area):
        detections = []
        keypoints_dict = {}
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kps_all = extract_keypoints(result)

        for i in range(len(boxes)):
            if confs[i] < Config.PERSON_CONF_THRESHOLD: continue
            
            kps = kps_all[i]
            visible = sum(1 for j in range(17) if kps[j][2] > Config.KEYPOINT_CONF_THRESHOLD)
            if visible < Config.MIN_VISIBLE_KEYPOINTS: continue

            x1, y1, x2, y2 = boxes[i]
            x1, x2 = max(0, min(w-1, int(x1))), max(0, min(w-1, int(x2)))
            y1, y2 = max(0, min(h-1, int(y1))), max(0, min(h-1, int(y2)))
            
            if x2 <= x1 or y2 <= y1: continue
            
            bbox_area = (x2-x1) * (y2-y1)
            if not (Config.MIN_BBOX_AREA_RATIO * area < bbox_area < Config.MAX_BBOX_AREA_RATIO * area):
                continue

            det_idx = len(detections)
            keypoints_dict[det_idx] = kps
            detections.append([x1, y1, x2, y2, confs[i], 0])

        return np.array(detections) if detections else np.empty((0, 6)), keypoints_dict


# Alias for handler.py compatibility
class MLTracker(BotSortTracker):
    """Alias for BotSortTracker for use in handler.py"""
    pass
