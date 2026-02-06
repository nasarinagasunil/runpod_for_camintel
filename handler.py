"""
RunPod Serverless Handler for Camintel Video Processing

This handler receives video processing jobs, runs the ML pipeline,
and returns results via webhook to the Cloudflare Workers backend.

The ML modules are copied exactly from backend/app/ml/ without modification.
"""

import runpod
import os
import tempfile
import logging
import requests
import boto3
from botocore.config import Config
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models")

# B2/S3 Configuration
B2_ENDPOINT_URL = os.environ.get("B2_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")


def get_b2_client(credentials: dict = None):
    """Create B2 S3-compatible client."""
    key_id = credentials.get("key_id") if credentials else os.environ.get("B2_APPLICATION_KEY_ID")
    key = credentials.get("key") if credentials else os.environ.get("B2_APPLICATION_KEY")
    
    return boto3.client(
        's3',
        endpoint_url=B2_ENDPOINT_URL,
        aws_access_key_id=key_id,
        aws_secret_access_key=key,
        config=Config(signature_version='s3v4')
    )


def download_video(s3_client, bucket: str, s3_key: str, local_path: str):
    """Download video from B2."""
    logger.info(f"Downloading video from s3://{bucket}/{s3_key}")
    s3_client.download_file(bucket, s3_key, local_path)
    logger.info(f"Downloaded to {local_path}")


def upload_file(s3_client, local_path: str, bucket: str, s3_key: str):
    """Upload file to B2."""
    logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
    s3_client.upload_file(local_path, bucket, s3_key)
    logger.info("Upload complete")


def generate_presigned_url(s3_client, bucket: str, s3_key: str, expires_in: int = 3600):
    """Generate presigned URL for accessing uploaded file."""
    return s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': s3_key},
        ExpiresIn=expires_in
    )


def process_video_with_ml(video_path: str, camera_id: int):
    """
    Process video using the ML pipeline.
    
    This imports and uses the exact modules from backend/app/ml/
    """
    import cv2
    import numpy as np
    
    # Import ML modules (copied from backend/app/ml/)
    from app.ml.tracking import MLTracker
    from app.ml.accident_detector import get_accident_detector_model
    from app.ml.activity_recognition import get_activity_recognition_model
    
    logger.info(f"Processing video: {video_path}")
    
    # Initialize models
    tracker = MLTracker()
    accident_model = get_accident_detector_model()
    activity_model = get_activity_recognition_model()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")
    
    # Output video setup
    output_path = video_path.replace('.mp4', '_processed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    tracks_data = {}
    frame_id = 0
    start_time = datetime.utcnow()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run tracking and ML inference
        results = tracker.process_frame(frame, frame_id)
        
        # Draw results on frame
        annotated_frame = tracker.draw_results(frame, results)
        out.write(annotated_frame)
        
        # Collect track data for each person
        for track in results.get("tracks", []):
            track_id = track.get("track_id")
            if track_id not in tracks_data:
                tracks_data[track_id] = {
                    "person_id": f"person_{track_id}",
                    "frames": [],
                    "activities": [],
                    "accidents": [],
                    "first_frame": frame_id,
                    "last_frame": frame_id
                }
            
            tracks_data[track_id]["last_frame"] = frame_id
            tracks_data[track_id]["frames"].append(frame_id)
            
            # Activity prediction
            if activity_model and track.get("keypoints") is not None:
                activity_model.update_pose(track_id, track["keypoints"])
                activity_pred = activity_model.predict_activity(track_id)
                if activity_pred[0]:  # activity detected
                    tracks_data[track_id]["activities"].append({
                        "frame": frame_id,
                        "activity": activity_pred[0],
                        "confidence": activity_pred[1]
                    })
            
            # Accident detection
            if accident_model and track.get("keypoints") is not None:
                accident_model.update_pose(track_id, track["keypoints"])
                is_accident, confidence = accident_model.predict_accident(track_id)
                if is_accident:
                    tracks_data[track_id]["accidents"].append({
                        "frame": frame_id,
                        "confidence": confidence
                    })
        
        frame_id += 1
        
        # Log progress every 100 frames
        if frame_id % 100 == 0:
            logger.info(f"Processed {frame_id}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    # Generate summary for each track
    end_time = datetime.utcnow()
    tracks_summary = []
    
    for track_id, data in tracks_data.items():
        # Calculate duration
        duration_frames = data["last_frame"] - data["first_frame"]
        duration_seconds = duration_frames / fps
        
        # Determine dominant activity
        activity_counts = {}
        for act in data["activities"]:
            activity_counts[act["activity"]] = activity_counts.get(act["activity"], 0) + 1
        
        dominant_activity = max(activity_counts, key=activity_counts.get) if activity_counts else "UNKNOWN"
        avg_confidence = sum(a["confidence"] for a in data["activities"]) / len(data["activities"]) if data["activities"] else 0
        
        # Check for accidents
        has_accident = len(data["accidents"]) > 0
        accident_confidence = max(a["confidence"] for a in data["accidents"]) if has_accident else 0
        
        track_start = start_time + timedelta(seconds=data["first_frame"] / fps)
        track_end = start_time + timedelta(seconds=data["last_frame"] / fps)
        
        tracks_summary.append({
            "person_id": data["person_id"],
            "activity": dominant_activity,
            "start_time": track_start.isoformat(),
            "end_time": track_end.isoformat(),
            "duration": round(duration_seconds, 2),
            "confidence": round(avg_confidence, 3),
            "is_accident": has_accident,
            "accident_confidence": round(accident_confidence, 3) if has_accident else None,
            "accident_type": "FALL" if has_accident else None,
            "incident_time": data["accidents"][0]["frame"] / fps if has_accident else None
        })
    
    logger.info(f"Processing complete. {len(tracks_summary)} tracks detected.")
    
    return {
        "output_path": output_path,
        "tracks": tracks_summary,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
        "processing_time_seconds": (end_time - start_time).total_seconds()
    }


def call_webhook(callback_url: str, payload: dict):
    """Send results to Cloudflare Workers webhook."""
    if not callback_url:
        logger.warning("No callback URL provided, skipping webhook")
        return
    
    try:
        logger.info(f"Calling webhook: {callback_url}")
        response = requests.post(
            callback_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Webhook response: {response.status_code}")
    except Exception as e:
        logger.error(f"Webhook call failed: {e}")


def handler(job):
    """
    RunPod serverless handler.
    
    Expected input:
    {
        "action": "process_video",
        "video_id": 123,
        "camera_id": 1,
        "s3_key": "cameras/1/video.mp4",
        "bucket": "camintel-videos",
        "b2_credentials": {
            "key_id": "...",
            "key": "..."
        },
        "callback_url": "https://api.camintel.com/api/v1/videos/webhook"
    }
    """
    try:
        job_input = job["input"]
        action = job_input.get("action", "process_video")
        
        if action != "process_video":
            return {"error": f"Unknown action: {action}"}
        
        video_id = job_input["video_id"]
        camera_id = job_input.get("camera_id", 1)
        s3_key = job_input["s3_key"]
        bucket = job_input["bucket"]
        b2_credentials = job_input.get("b2_credentials", {})
        callback_url = job_input.get("callback_url")
        
        logger.info(f"Processing job: video_id={video_id}, s3_key={s3_key}")
        
        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Determine file extension
            ext = os.path.splitext(s3_key)[1] or ".mp4"
            local_video_path = os.path.join(temp_dir, f"input{ext}")
            
            # Download video from B2
            s3_client = get_b2_client(b2_credentials)
            download_video(s3_client, bucket, s3_key, local_video_path)
            
            # Process video with ML pipeline
            results = process_video_with_ml(local_video_path, camera_id)
            
            # Upload processed video
            processed_key = f"processed/{video_id}_processed.mp4"
            upload_file(s3_client, results["output_path"], bucket, processed_key)
            
            # Generate presigned URL for processed video
            processed_url = generate_presigned_url(s3_client, bucket, processed_key, expires_in=86400)
            
            # Generate thumbnail (first frame)
            thumbnail_key = None
            thumbnail_url = None
            # TODO: Extract and upload thumbnail
            
            # Prepare webhook payload
            webhook_payload = {
                "video_id": video_id,
                "camera_id": camera_id,
                "status": "completed",
                "processed_video_url": processed_url,
                "thumbnail_url": thumbnail_url,
                "tracks": results["tracks"],
                "summary": {
                    "total_frames": results["total_frames"],
                    "duration_seconds": results["duration_seconds"],
                    "processing_time_seconds": results["processing_time_seconds"],
                    "tracks_count": len(results["tracks"]),
                    "incidents_count": sum(1 for t in results["tracks"] if t["is_accident"])
                }
            }
            
            # Call webhook
            call_webhook(callback_url, webhook_payload)
            
            return {
                "status": "completed",
                "video_id": video_id,
                "tracks_count": len(results["tracks"]),
                "processing_time_seconds": results["processing_time_seconds"]
            }
            
    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)
        
        # Try to notify webhook of failure
        try:
            if job_input.get("callback_url"):
                call_webhook(job_input["callback_url"], {
                    "video_id": job_input.get("video_id"),
                    "status": "failed",
                    "error": str(e)
                })
        except:
            pass
        
        return {"error": str(e)}


# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
