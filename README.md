# RunPod ML Service

RunPod serverless endpoint for Camintel ML video processing.

## Directory Structure

```
runpod-ml-service/
├── Dockerfile              # PyTorch + CUDA container
├── handler.py              # RunPod serverless entry point
├── requirements.txt        # Python dependencies
├── models/                 # ML model weights (not in git)
│   ├── yolov8n-pose.pt
│   ├── osnet_x0_25_msmt17.pt
│   ├── best_activity_model.pth
│   └── best_accident_model.pth
└── app/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   └── config.py          # Minimal config for RunPod
    ├── services/
    │   ├── __init__.py
    │   └── employee_db.py     # File-based employee DB for face recognition
    └── ml/                    # ML modules (copied from backend)
        ├── __init__.py
        ├── tracking.py              # YOLOv8 + BoTSORT person tracking
        ├── accident_detector.py     # Fall detection LSTM
        ├── activity_recognition.py  # Activity classifier (walk/stand/sit)
        ├── person_identification.py # DeepFace face recognition
        ├── theft_detector.py        # Theft detection (boilerplate)
        ├── model_manager.py         # Model loading utilities
        ├── error_handler.py         # ML error handling
        ├── data_validator.py        # Input validation
        ├── confidence_manager.py    # Quality scoring
        └── performance_monitor.py   # Performance metrics
```

## Deployment Steps

1. Build Docker image:
```bash
docker build -t yourusername/camintel-ml:latest .
docker push yourusername/camintel-ml:latest
```

2. Create RunPod Endpoint:
   - Go to runpod.io → Serverless → New Endpoint
   - Container Image: `yourusername/camintel-ml:latest`
   - GPU: 16GB+ (RTX 4000 recommended)
   - Max Workers: 3

3. Copy Endpoint ID and API Key to Cloudflare Workers secrets

## Environment Variables (set in RunPod)

- `B2_APPLICATION_KEY_ID`
- `B2_APPLICATION_KEY`
- `B2_BUCKET_NAME`
- `B2_ENDPOINT_URL`
