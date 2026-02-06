
import numpy as np
import cv2
import time
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from app.core.config import settings
from app.services.employee_db import EmployeeDatabase

logger = logging.getLogger(__name__)

# Try to import DeepFace, handle failure gracefully
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DeepFace library not found. Slow Lane Pipeline will be disabled. Error: {e}")
    DEEPFACE_AVAILABLE = False
    
    # Mock class to prevent NameErrors in type hints or unused imports
    class DeepFace:
        pass

class SlowLanePipeline:
    def __init__(self):
        self.config = settings
        self.database = EmployeeDatabase()
        
        if not DEEPFACE_AVAILABLE:
            logger.warning("Slow Lane Pipeline initialized in MOCK mode (DeepFace missing)")
            return

        logger.info("Slow Lane Face Recognition Pipeline initialized")
        
        # Warm up ArcFace model
        try:
            logger.info("Warming up ArcFace model...")
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            # We wrap this to catch initial download/load errors
            DeepFace.represent(
                img_path=dummy,
                model_name=self.config.FACE_RECOGNITION_MODEL,
                detector_backend="skip",
                enforce_detection=False
            )
            logger.info("ArcFace model ready")
        except Exception as e:
            logger.warning(f"Model warm-up failed (might be first run or missing weights): {e}")

    def process_frame(self, frame: np.ndarray, frame_id: int, 
                      fast_lane_tracks: List[Dict] = None,
                      rfid_readings: Dict[int, str] = None) -> Dict:
        """
        Process one frame through the complete pipeline
        """
        start_time = time.time()
        timing = {}
        results = []
        
        if not DEEPFACE_AVAILABLE:
            return {
                'frame_id': frame_id,
                'timestamp': time.time(),
                'faces_detected': 0,
                'identities': [],
                'timing': {},
                'total_ms': 0,
                'error': 'DeepFace not available'
            }
        
        try:
            # Step 1: Face Detection
            t0 = time.time()
            faces = self._detect_faces(frame)
            timing['detection_ms'] = (time.time() - t0) * 1000
            
            for face in faces:
                identity_result = {}
                
                # Step 2: Alignment
                t0 = time.time()
                # Use Smart Alignment result if available, else fallback
                if 'aligned_img' in face and face['aligned_img'] is not None:
                     aligned = face['aligned_img']
                else:
                     aligned = self._align_face(frame, face['bbox'])
                
                timing['alignment_ms'] = timing.get('alignment_ms', 0) + (time.time() - t0) * 1000
                
                # Step 3: Embedding
                t0 = time.time()
                embedding = self._extract_embedding(aligned)
                timing['embedding_ms'] = timing.get('embedding_ms', 0) + (time.time() - t0) * 1000
                
                if embedding is not None:
                    # Step 4: Identity Match
                    t0 = time.time()
                    identity_result = self._find_identity(embedding)
                    timing['matching_ms'] = timing.get('matching_ms', 0) + (time.time() - t0) * 1000
                
                # Step 5: RFID Check (if available)
                track_id = self._match_to_track(face['bbox'], fast_lane_tracks)
                if rfid_readings and track_id and track_id in rfid_readings:
                    t0 = time.time()
                    identity_result = self._verify_with_rfid(identity_result, rfid_readings[track_id])
                    timing['rfid_ms'] = timing.get('rfid_ms', 0) + (time.time() - t0) * 1000
                
                # Combine results
                final_result = {
                    'bbox': face['bbox'],
                    'face_confidence': face['confidence'],
                    'track_id': track_id,
                    **identity_result
                }
                results.append(final_result)
                
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
        
        total_ms = (time.time() - start_time) * 1000
        
        return {
            'frame_id': frame_id,
            'timestamp': time.time(),
            'faces_detected': len(faces),
            'identities': results,
            'timing': timing,
            'total_ms': total_ms
        }

    def _detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using configured backend (RetinaFace/MTCNN)"""
        faces = []
        try:
            # DeepFace extract_faces returns list of dicts
            results = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.config.FACE_DETECTOR_BACKEND,
                enforce_detection=False,
                align=True  # [IMPROVEMENT] Enable smart alignment for rotation correction
            )
            
            for result in results:
                # Filter by confidence if available (DeepFace result structure varies by version)
                confidence = result.get('confidence', 1.0)
                
                # DeepFace area keys: x, y, w, h
                area = result.get('facial_area', {})
                x, y, w, h = area.get('x', 0), area.get('y', 0), area.get('w', 0), area.get('h', 0)
                
                # Get the ALIGNED face image directly from DeepFace
                # It is usually normalized to [0,1], we might need to cast to uint8 for consistency if needed
                aligned_face_img = result.get('face') 
                if aligned_face_img is not None:
                     # DeepFace returns float [0,1], convert back to [0,255] uint8 for opencv/storage
                     if aligned_face_img.max() <= 1.0:
                         aligned_face_img = (aligned_face_img * 255).astype(np.uint8)
                     # Ensure BGR (DeepFace usually returns RGB)
                     aligned_face_img = cv2.cvtColor(aligned_face_img, cv2.COLOR_RGB2BGR)

                if w >= self.config.MIN_FACE_SIZE and h >= self.config.MIN_FACE_SIZE:
                    faces.append({
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'aligned_img': aligned_face_img # Store aligned image
                    })
        except Exception as e:
            # 'No face detected' sometimes raises generic exception in DeepFace
            # logger.debug(f"Face detection info: {e}")
            pass
            
        return faces

    def _align_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop and resize face with padding"""
        x, y, w, h = bbox
        
        # Add 20% padding
        pad = int(0.2 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        face_crop = frame[y1:y2, x1:x2]
        
        # Resize to 224x224 (standard for ArcFace)
        if face_crop.size == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)
            
        aligned = cv2.resize(face_crop, (224, 224))
        return aligned

    def _extract_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """Extract 512D ArcFace embedding"""
        try:
            result = DeepFace.represent(
                img_path=aligned_face,
                model_name=self.config.FACE_RECOGNITION_MODEL,
                detector_backend="skip",
                enforce_detection=False
            )
            
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
        return None

    def _find_identity(self, query_embedding: np.ndarray) -> Dict:
        """Search database for matching identity using cosine similarity"""
        emb_matrix, emp_ids = self.database.get_all_embeddings()
        
        if len(emb_matrix) == 0:
            return {"matched": False, "reason": "empty_database"}
        
        # Cosine Similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_normalized = emb_matrix / (emb_norms + 1e-10)
        
        similarities = np.dot(emb_normalized, query_norm)
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_distance = 1 - best_similarity
        
        best_emp_id = emp_ids[best_idx]
        emp_data = self.database.employees.get(best_emp_id, {})
        
        # Threshold Check
        if best_distance <= self.config.FACE_MATCH_THRESHOLD:
            return {
                "matched": True,
                "verified": True,
                "employee_id": best_emp_id,
                "name": emp_data.get('name', 'Unknown'),
                "role": emp_data.get('role', 'Unknown'),
                "confidence": float(best_similarity),
                "distance": float(best_distance)
            }
        
        # --- Auto-Enrollment for Unknown Faces MATCH_THRESHOLD Check ---
        # If we reach here, it means no existing face matched closely enough.
        
        # [MODIFIED] Auto-enrollment disabled to prevent database bloat.
        # Unknown faces will remain unidentified.
        
        return {
            "matched": False,
            "verified": False,
            "employee_id": None,
            "name": "Unknown",
            "role": "Visitor",
            "confidence": 0.0,
            "distance": float(best_distance),
            "reason": "low_confidence"
        }

    def _match_to_track(self, face_bbox, tracks):
        """Match face to Fast Lane track by bbox overlap"""
        if not tracks:
            return None
        
        fx, fy, fw, fh = face_bbox
        face_center = (fx + fw/2, fy + fh/2)
        
        for track in tracks:
            # Assuming track['bbox'] format is compatible
            tx, ty, tw, th = track.get('bbox', [0,0,0,0])
            if tx <= face_center[0] <= tx + tw and ty <= face_center[1] <= ty + th:
                return track.get('track_id')
        return None

    def _verify_with_rfid(self, identity: Dict, rfid: str) -> Dict:
        """Cross-check with RFID"""
        rfid_employee = self.database.get_by_rfid(rfid)
        
        if rfid_employee:
            if identity.get('employee_id') == rfid_employee['employee_id']:
                # RFID confirms face
                identity['rfid_confirmed'] = True
                identity['confidence'] = min(1.0, identity.get('confidence', 0) + 0.2)
            else:
                # RFID overrides face
                identity = {
                    "matched": True,
                    "verified": True,
                    "employee_id": rfid_employee['employee_id'],
                    "name": rfid_employee['name'],
                    "role": rfid_employee['role'],
                    "confidence": 0.95,
                    "rfid_confirmed": True,
                    "method": "rfid_override"
                }
        return identity

    def enroll_from_image(self, image_path: str, name: str, role: str) -> bool:
        """
        Enroll a person from an image file.
        Returns True if successful, False otherwise.
        """
        try:
            # 1. Detect and represent
            # We use the same parameters as the pipeline to ensure compatibility
            embeddings = DeepFace.represent(
                img_path=image_path,
                model_name=self.config.FACE_RECOGNITION_MODEL,
                detector_backend=self.config.FACE_DETECTOR_BACKEND,
                enforce_detection=True
            )
            
            if not embeddings:
                logger.warning(f"No face detected in {image_path}")
                return False
                
            # 2. Extract embedding vector
            # DeepFace.represent returns a list of dicts. We take the first one (assuming single face).
            # Each dict has 'embedding' key.
            embedding_vector = np.array(embeddings[0]['embedding'])
            
            # 3. Generate ID
            import uuid
            emp_id = str(uuid.uuid4())
            
            # 4. Save to database
            # The database expects a LIST of embeddings (to support multiple angles in future)
            self.database.enroll(
                emp_id=emp_id,
                name=name,
                role=role,
                embeddings=[embedding_vector]
            )
            logger.info(f"Successfully enrolled {name} (ID: {emp_id})")
            return True
            
        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            return False

# Instantiate globally or per-request as needed
slow_lane_pipeline = SlowLanePipeline()
