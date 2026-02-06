import torch
import torch.nn as nn
import numpy as np
from collections import deque
from scipy.signal import savgol_filter
import os

# ============================================================================
# Feature Extractor
# ============================================================================
class PoseFeatureExtractor:
    """Extract meaningful features from pose keypoint sequences"""

    def __init__(self):
        # COCO keypoint connections (bones)
        self.bones = [
            (0, 1), (0, 2),      # Head connections
            (1, 3), (2, 4),      # Face to shoulders
            (0, 5), (0, 6),      # Neck to shoulders
            (5, 7), (7, 9),      # Left arm
            (6, 8), (8, 10),     # Right arm
            (5, 11), (6, 12),    # Torso
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]

    def normalize_pose(self, pose_seq):
        """
        Normalize pose to be translation & scale invariant
        """
        normalized = pose_seq.copy()

        for t in range(len(pose_seq)):
            pose = pose_seq[t]

            # Center around torso midpoint (between shoulders)
            if pose[5].sum() != 0 and pose[6].sum() != 0:
                center = (pose[5] + pose[6]) / 2  # Midpoint of shoulders
                normalized[t] = pose - center

                # Scale by torso height (makes size invariant)
                torso_height = np.linalg.norm(pose[5] - pose[11]) + \
                               np.linalg.norm(pose[6] - pose[12])
                if torso_height > 0:
                    normalized[t] = normalized[t] / (torso_height + 1e-6)

        return normalized

    def compute_bone_angles(self, pose_seq):
        """Compute angles of each bone over time"""
        T = len(pose_seq)
        angles = np.zeros((T, len(self.bones)))

        for t in range(T):
            for i, (j1, j2) in enumerate(self.bones):
                v = pose_seq[t, j2] - pose_seq[t, j1] # Vector from joint1 to joint2
                angle = np.arctan2(v[1], v[0]) # Angle of this vector
                angles[t, i] = angle

        return angles

    def compute_velocities(self, pose_seq):
        """Compute how fast each joint is moving"""
        velocities = np.diff(pose_seq, axis=0)  # (T-1, 17, 2)
        # Pad to maintain same length
        velocities = np.vstack([velocities, velocities[-1:]])
        return velocities

    def extract_features(self, pose_seq):
        """
        Extract ALL features from a pose sequence
        Input: (30, 17, 2)
        Output: (30, 84)
        """
        # 1. Normalize pose
        norm_pose = self.normalize_pose(pose_seq)

        # 2. Flatten normalized positions
        positions = norm_pose.reshape(len(pose_seq), -1)

        # 3. Compute velocities
        velocities = self.compute_velocities(norm_pose)
        velocities_flat = velocities.reshape(len(velocities), -1)

        # 4. Compute bone angles
        angles = self.compute_bone_angles(norm_pose)

        # Concatenate all features
        features = np.concatenate([positions, velocities_flat, angles], axis=1)
        return features


# ============================================================================
# LSTM Model
# ============================================================================
class ActivityClassifierLSTM(nn.Module):
    """LSTM-based activity classifier"""

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.3):
        super(ActivityClassifierLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,      # 84 features per frame
            hidden_size=hidden_size,    # 128 neurons
            num_layers=num_layers,      # 2 stacked LSTM layers
            batch_first=True,           # Input shape: (batch, sequence, features)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True          # Read sequence forward AND backward
        )

        # Fully connected layers (Sequential to match state dict keys: fc.0, fc.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # LSTM processes the sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size*2)

        # Pass through fully connected layers
        out = self.fc(last_output)

        return out


# ============================================================================
# Activity Predictor
# ============================================================================
class ActivityPredictor:
    """Real-time activity prediction from pose sequences"""

    def __init__(self, model_path, device=None, sequence_length=30):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.sequence_length = sequence_length
        self.extractor = PoseFeatureExtractor()

        # Activity labels
        self.activities = [
            'walking',
            'standing_idle',
            'sitting'
        ]

        # Initialize model
        self.model = ActivityClassifierLSTM(
            input_size=84,
            hidden_size=128,
            num_layers=2,
            num_classes=len(self.activities),
            dropout=0.3
        ).to(self.device)

        # Load trained model if available
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"✅ Activity model loaded from {model_path}")
            except Exception as e:
                print(f"❌ Failed to load activity model: {e}")
        else:
            print(f"⚠️ Activity model not found at {model_path}. Predictor will not work correctly until trained.")

        # Track pose sequences per person
        self.pose_history = {}  # {track_id: deque of poses}

    def update_pose(self, track_id, keypoints_xyc):
        """Update pose history for a tracked person"""
        if track_id not in self.pose_history:
            self.pose_history[track_id] = deque(maxlen=self.sequence_length)

        # Store just x,y coordinates (drop confidence)
        pose = keypoints_xyc[:, :2]  # (17, 2)
        self.pose_history[track_id].append(pose)

    def predict_activity(self, track_id, confidence_threshold=0.5):
        """Predict activity for a tracked person"""
        if track_id not in self.pose_history:
            return None, 0.0

        history = self.pose_history[track_id]

        # Relaxed check: Allow if we have at least 1 frame
        if len(history) < 1:
            return "warming_up", 0.0

        # Convert to numpy array
        pose_seq = np.array(list(history))  # (N, 17, 2)
        
        # Pad if needed
        if len(pose_seq) < self.sequence_length:
             # Pad with last frame
            pad_count = self.sequence_length - len(pose_seq)
            last_frame = pose_seq[-1]
            padding = np.tile(last_frame, (pad_count, 1, 1))
            pose_seq = np.vstack([pose_seq, padding])

        # Extract features
        try:
            features = self.extractor.extract_features(pose_seq)
        except Exception as e:
            print(f"Feature extraction failed for ID {track_id}: {e}")
            return None, 0.0

        # Predict
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

            activity = self.activities[predicted.item()]
            conf = confidence.item()
            
            # DEBUG: Print all probs to see what's happening
            print(f"DEBUG: Track {track_id} | Activity: {activity} ({conf:.2f})")
            # print(f"Track {track_id}: {activity} ({conf:.2f}) | Probs: {probs.cpu().numpy()}")

        # --- Velocity Heuristic Override ---
        # The LSTM model is biased towards "walking". We use a simple heuristic:
        # If the person is not moving (low velocity), force "standing_idle".
        
        # Velocity features are indices 34 to 67 (34 values)
        # features is (Seq, 84).
        velocity_features = features[:, 34:68]
        avg_velocity = np.mean(np.abs(velocity_features))
        
        # Threshold: 0.005 (normalized units). Reduced to allow slow walking.
        VELOCITY_THRESHOLD = 0.005
        
        print(f"DEBUG: ID {track_id} Avg Velocity: {avg_velocity:.4f}")
        
        if avg_velocity < VELOCITY_THRESHOLD:
            # Force standing_idle
            if activity == 'walking': # Only override walking, maybe sitting is valid low velocity
                 # print(f"DEBUG: Overriding Walking -> Standing (Vel {avg_velocity:.4f} < {VELOCITY_THRESHOLD})")
                 activity = 'standing_idle'
                 conf = 0.95 # High confidence override

        # Only return if confidence is high enough
        if conf < confidence_threshold:
            # print(f"DEBUG: Track {track_id} ignored {activity} (Conf: {conf:.2f} < {confidence_threshold})")
            return None, conf

        return activity, conf

    def cleanup_old_tracks(self, active_track_ids):
        """Remove inactive tracks to save memory"""
        all_ids = list(self.pose_history.keys())
        for track_id in all_ids:
            if track_id not in active_track_ids:
                del self.pose_history[track_id]


def get_activity_recognition_model():
    """Factory function to get the activity recognition model.
    Currently used by ActivityMonitorService but usage is commented out.
    """
    return ActivityPredictor(model_path='/app/models/best_activity_model.pth')
