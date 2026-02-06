#!/usr/bin/env python3
"""
Test script for RunPod handler.py
Run this locally to verify the ML pipeline works before deploying to RunPod.

Usage:
    cd runpod-ml-service
    pip install -r requirements.txt
    python test_handler.py --video /path/to/test_video.mp4
"""

import os
import sys
import json
import argparse
import tempfile

# Set up environment variables for testing
os.environ.setdefault("B2_APPLICATION_KEY_ID", "005ab1f0c3bf7570000000004")
os.environ.setdefault("B2_APPLICATION_KEY", "K005Nvr6DKb6no71mTNPCneZjC3b78s")
os.environ.setdefault("B2_BUCKET_NAME", "camintel")
os.environ.setdefault("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")


def test_ml_pipeline_only(video_path: str):
    """
    Test just the ML pipeline without B2 upload/download.
    This is the quickest way to verify the ML code works.
    """
    print("=" * 60)
    print("Testing ML Pipeline Only (no B2)")
    print("=" * 60)
    
    # Import ML modules
    try:
        from app.ml.tracking import BotSortTracker, MLTracker
        print("✅ Imported tracking module")
    except Exception as e:
        print(f"❌ Failed to import tracking: {e}")
        return False
    
    try:
        from app.ml.accident_detector import AccidentPredictor
        print("✅ Imported accident_detector module")
    except Exception as e:
        print(f"❌ Failed to import accident_detector: {e}")
        return False
    
    try:
        from app.ml.activity_recognition import ActivityPredictor
        print("✅ Imported activity_recognition module")
    except Exception as e:
        print(f"❌ Failed to import activity_recognition: {e}")
        return False
    
    try:
        from app.ml.person_identification import SlowLanePipeline
        print("✅ Imported person_identification module")
    except Exception as e:
        print(f"⚠️ person_identification import warning (DeepFace may not be installed): {e}")
    
    # Test video processing
    if video_path and os.path.exists(video_path):
        print(f"\n📹 Processing video: {video_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.mp4")
            
            try:
                tracker = MLTracker()
                print("✅ Created MLTracker instance")
                
                result_path, tracks = tracker.process_video(video_path, output_path)
                print(f"✅ Processed video successfully!")
                print(f"   Output: {result_path}")
                print(f"   Tracks found: {len(tracks)}")
                
                for i, track in enumerate(tracks[:5]):  # Show first 5 tracks
                    print(f"   - Track {track['track_id']}: frames {track['start_frame']}-{track['end_frame']}")
                
                return True
            except Exception as e:
                print(f"❌ Video processing failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    else:
        print("\n⚠️ No video path provided, skipping video processing test")
        print("   Run with: python test_handler.py --video /path/to/video.mp4")
        return True


def test_handler_mock():
    """
    Test the handler function with a mock job (no actual B2 operations).
    """
    print("\n" + "=" * 60)
    print("Testing Handler with Mock Job")
    print("=" * 60)
    
    try:
        from handler import handler
        print("✅ Imported handler function")
    except Exception as e:
        print(f"❌ Failed to import handler: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create mock job
    mock_job = {
        "id": "test-job-123",
        "input": {
            "action": "process_video",
            "video_id": 999,
            "camera_id": 1,
            "s3_key": "test/video.mp4",
            "bucket": "camintel",
            "b2_credentials": {
                "key_id": os.environ.get("B2_APPLICATION_KEY_ID"),
                "key": os.environ.get("B2_APPLICATION_KEY")
            },
            "callback_url": None  # Disable webhook for testing
        }
    }
    
    print(f"Mock job: {json.dumps(mock_job, indent=2)}")
    print("\n⚠️ Note: This will fail at video download since we're not using a real video.")
    print("   This test verifies the handler structure and imports work correctly.\n")
    
    try:
        result = handler(mock_job)
        print(f"Handler result: {json.dumps(result, indent=2)}")
        return "error" in result  # Expected to fail at download
    except Exception as e:
        print(f"Handler error (expected): {e}")
        return True


def test_b2_connection():
    """
    Test B2 connection and list bucket contents.
    """
    print("\n" + "=" * 60)
    print("Testing B2 Connection")
    print("=" * 60)
    
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("❌ boto3 not installed. Run: pip install boto3")
        return False
    
    key_id = os.environ.get("B2_APPLICATION_KEY_ID")
    key = os.environ.get("B2_APPLICATION_KEY")
    bucket = os.environ.get("B2_BUCKET_NAME", "camintel")
    endpoint = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    
    if not key_id or not key:
        print("❌ B2 credentials not set in environment")
        return False
    
    print(f"Connecting to B2...")
    print(f"  Endpoint: {endpoint}")
    print(f"  Bucket: {bucket}")
    print(f"  Key ID: {key_id[:10]}...")
    
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=key,
            config=Config(signature_version='s3v4')
        )
        
        # List objects in bucket
        response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=5)
        
        print(f"✅ B2 connection successful!")
        
        if 'Contents' in response:
            print(f"\nFirst 5 objects in bucket:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print(f"\n  Bucket is empty or no access")
        
        return True
        
    except Exception as e:
        print(f"❌ B2 connection failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RunPod ML Handler")
    parser.add_argument("--video", help="Path to test video file")
    parser.add_argument("--b2", action="store_true", help="Test B2 connection")
    parser.add_argument("--handler", action="store_true", help="Test handler with mock job")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.all or (not args.b2 and not args.handler):
        # Run ML pipeline test by default
        success = test_ml_pipeline_only(args.video)
        
    if args.b2 or args.all:
        test_b2_connection()
        
    if args.handler or args.all:
        test_handler_mock()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
