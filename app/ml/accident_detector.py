import torch
import torch.nn as nn
import numpy as np
from collections import deque
import os

# ============================================================================
# Model Architecture (from User)
# ============================================================================
class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=128, num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()

        self.bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn * lstm_out, dim=1)

        return self.classifier(context)


# ============================================================================
# Accident Predictor
# ============================================================================
class AccidentPredictor:
    """Real-time accident (fall) prediction from pose sequences"""

    def __init__(self, model_path, device=None, sequence_length=32):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.sequence_length = sequence_length
        # Input size is 51 (17 keypoints * 3 values [x, y, conf])
        self.input_size = 51 

        # Initialize model
        self.model = FallDetectionLSTM(
            input_size=self.input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=2,
            dropout=0.5
        ).to(self.device)

        # Load trained model if available
        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
                print(f"✅ Accident model loaded from {model_path}")
            except Exception as e:
                print(f"❌ Failed to load accident model: {e}")
        else:
            print(f"⚠️ Accident model not found at {model_path}. Accident prediction DISABLED.")

        # Track pose sequences per person
        # We need raw keypoints (x, y, conf) flattened
        self.pose_history = {}  # {track_id: deque of flat_keypoints}

    def _normalize_pose(self, keypoints):
        """
        Normalize a single pose (17, 3) to be translation & scale invariant.
        Returns: (17, 3) normalized pose
        """
        # Copy to avoid modifying original
        norm_kp = keypoints.copy()
        
        # COCO Keypoints: 5=L_Shoulder, 6=R_Shoulder, 11=L_Hip, 12=R_Hip
        # Check if we have necessary keypoints (conf > 0)
        # We use the confidence channel (idx 2) to check visibility
        if (norm_kp[5, 2] > 0 and norm_kp[6, 2] > 0 and 
            norm_kp[11, 2] > 0 and norm_kp[12, 2] > 0):
            
            # 1. Center around shoulder midpoint
            # We only normalize x,y (indices 0,1). Leave conf (index 2) alone.
            center_x = (norm_kp[5, 0] + norm_kp[6, 0]) / 2
            center_y = (norm_kp[5, 1] + norm_kp[6, 1]) / 2
            
            norm_kp[:, 0] -= center_x
            norm_kp[:, 1] -= center_y
            
            # 2. Scale by torso height
            # Dist between shoulder and hip
            h1 = np.linalg.norm(norm_kp[5, :2] - norm_kp[11, :2])
            h2 = np.linalg.norm(norm_kp[6, :2] - norm_kp[12, :2])
            scale = (h1 + h2) / 2
            
            if scale > 0:
                norm_kp[:, :2] /= scale
                
        return norm_kp

    def update_pose(self, track_id, keypoints):
        """
        Update pose history for a tracked person.
        keypoints: (17, 3) numpy array [x, y, conf]
        """
        if track_id not in self.pose_history:
            self.pose_history[track_id] = deque(maxlen=self.sequence_length)

        # Normalize BEFORE flattening
        norm_kp = self._normalize_pose(keypoints)

        # Flatten features: (17, 3) -> (51,)
        flat_kp = norm_kp.flatten()
        self.pose_history[track_id].append(flat_kp)

    def predict_accident(self, track_id, confidence_threshold=0.5):
        """
        Predict if an accident (fall) has occurred.
        Returns: (is_accident: bool, confidence: float)
        """
        if track_id not in self.pose_history:
            return False, 0.0

        history = self.pose_history[track_id]
        # Relaxed check: Allow if we have at least 10 frames (approx 0.3s)
        # 1 frame is too noisy and causes false positives when padded
        if len(history) < 10:
            return False, 0.0

        # Create sequence array
        pose_seq = np.array(list(history))  # (N, 51)
        
        # Pad if needed (replicate last frame to match training logic)
        if len(pose_seq) < self.sequence_length:
            # print(f"DEBUG: Padding track {track_id} from {len(pose_seq)} to {self.sequence_length}")
            # Calculate how many frames to pad
            pad_count = self.sequence_length - len(pose_seq)
            # Create padding by repeating the last frame
            last_frame = pose_seq[-1]
            padding = np.tile(last_frame, (pad_count, 1))
            pose_seq = np.vstack([pose_seq, padding])

        # Predict
        if not self.model_loaded:
             return False, 0.0

        with torch.no_grad():
            input_tensor = torch.FloatTensor(pose_seq).unsqueeze(0).to(self.device) # (1, 32, 51)
            
            # DEBUG: Check input range
            if len(history) % 30 == 0: # Log rarely to avoid spam
                print(f"DEBUG: ID {track_id} Input Stats - Min: {input_tensor.min():.2f}, Max: {input_tensor.max():.2f}, Mean: {input_tensor.mean():.2f}")
                
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Class 1 is Fall, Class 0 is No Fall (usually)
            fall_prob = probs[0, 1].item()
            
        # Log ALL probabilities for debugging
        print(f"DEBUG: Accident Prob for ID {track_id}: {fall_prob:.6f}")

        # ---------------------------------------------------------
        # HEURISTIC FALLBACK (For robustness)
        # ---------------------------------------------------------
        # Standard Fall Check: Is the person horizontal?
        # Check current frame (last in sequence)
        current_pose = pose_seq[-1] # (51,)
        # Reshape to (17, 3)
        kp = current_pose.reshape(17, 3)
        
        # Check if we have shoulders (5,6) and hips (11,12)
        if (kp[5, 2] > 0 and kp[6, 2] > 0 and kp[11, 2] > 0 and kp[12, 2] > 0):
            # Midpoints
            shoulder_mid = (kp[5, :2] + kp[6, :2]) / 2
            hip_mid = (kp[11, :2] + kp[12, :2]) / 2
            
            diff = shoulder_mid - hip_mid
            dy = abs(diff[1])
            dx = abs(diff[0])
            
            # If horizontal distance > vertical distance * 1.5 -> Likely Lying Down
            if dx > dy * 1.5:
                print(f"DEBUG: Heuristic Fall Triggered (Horizontal Torso) for ID {track_id}")
                # Boost probability if logic agrees
                if fall_prob < 0.5:
                    fall_prob = 0.85

        if fall_prob > 0.4:
            print(f"!!! ACCIDENT DETECTED for ID {track_id} (conf={fall_prob:.4f}) !!!")
            return True, fall_prob
            
        return False, fall_prob

    def cleanup_old_tracks(self, active_track_ids):
        """Remove inactive tracks to save memory"""
        all_ids = list(self.pose_history.keys())
        for track_id in all_ids:
            if track_id not in active_track_ids:
                del self.pose_history[track_id]


def get_accident_detector_model():
    """Factory function to get the accident detection model."""
    return AccidentPredictor(model_path='/app/models/best_accident_model.pth')
