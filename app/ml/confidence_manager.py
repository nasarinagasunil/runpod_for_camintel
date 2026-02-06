"""
Confidence Calibration and Prediction Quality Management
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

class ConfidenceCalibrator:
    """Calibrate model confidence scores to match actual accuracy"""
    
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrators = {}
        self.calibration_data = defaultdict(list)
        self.is_fitted = {}
    
    def add_prediction(self, model_name: str, confidence: float, 
                      is_correct: bool, prediction_class: str = None):
        """Add prediction result for calibration"""
        key = f"{model_name}_{prediction_class}" if prediction_class else model_name
        self.calibration_data[key].append((confidence, is_correct))
    
    def fit_calibrator(self, model_name: str, prediction_class: str = None) -> bool:
        """Fit calibrator for a specific model/class"""
        key = f"{model_name}_{prediction_class}" if prediction_class else model_name
        
        if len(self.calibration_data[key]) < 50:  # Need minimum data
            logger.warning(f"Insufficient data for calibration: {key}")
            return False
        
        confidences, correctness = zip(*self.calibration_data[key])
        confidences = np.array(confidences)
        correctness = np.array(correctness, dtype=int)
        
        try:
            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(confidences, correctness)
                self.calibrators[key] = calibrator
                self.is_fitted[key] = True
                
                logger.info(f"Fitted isotonic calibrator for {key}")
                return True
                
        except Exception as e:
            logger.error(f"Calibration fitting failed for {key}: {e}")
            return False
    
    def calibrate_confidence(self, model_name: str, confidence: float, 
                           prediction_class: str = None) -> float:
        """Apply calibration to confidence score"""
        key = f"{model_name}_{prediction_class}" if prediction_class else model_name
        
        if key not in self.calibrators or not self.is_fitted.get(key, False):
            return confidence  # Return uncalibrated if no calibrator
        
        try:
            calibrated = self.calibrators[key].predict([confidence])[0]
            return float(np.clip(calibrated, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Calibration failed for {key}: {e}")
            return confidence
    
    def get_calibration_metrics(self, model_name: str, 
                              prediction_class: str = None) -> Dict:
        """Get calibration quality metrics"""
        key = f"{model_name}_{prediction_class}" if prediction_class else model_name
        
        if len(self.calibration_data[key]) < 10:
            return {"error": "Insufficient data"}
        
        confidences, correctness = zip(*self.calibration_data[key])
        confidences = np.array(confidences)
        correctness = np.array(correctness, dtype=int)
        
        # Reliability diagram data
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                correctness, confidences, n_bins=10
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = correctness[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return {
                "expected_calibration_error": float(ece),
                "reliability_diagram": {
                    "fraction_of_positives": fraction_of_positives.tolist(),
                    "mean_predicted_value": mean_predicted_value.tolist()
                },
                "total_predictions": len(confidences),
                "average_confidence": float(np.mean(confidences)),
                "average_accuracy": float(np.mean(correctness))
            }
            
        except Exception as e:
            logger.error(f"Calibration metrics calculation failed: {e}")
            return {"error": str(e)}

class UncertaintyQuantifier:
    """Quantify prediction uncertainty using multiple methods"""
    
    def __init__(self):
        self.ensemble_predictions = defaultdict(list)
        self.mc_dropout_samples = 10
    
    def monte_carlo_dropout(self, model: torch.nn.Module, input_tensor: torch.Tensor, 
                           n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using Monte Carlo Dropout
        Returns: (mean_prediction, uncertainty)
        """
        model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(input_tensor)
                predictions.append(F.softmax(pred, dim=-1))
        
        model.eval()  # Restore eval mode
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1)  # Average variance across classes
        
        return mean_pred, uncertainty
    
    def ensemble_uncertainty(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate uncertainty from ensemble predictions
        """
        if len(predictions) < 2:
            return predictions[0], torch.zeros(predictions[0].shape[0])
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1)
        
        return mean_pred, uncertainty
    
    def epistemic_uncertainty(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
        """
        Estimate epistemic (model) uncertainty
        """
        # Use gradient magnitude as proxy for epistemic uncertainty
        model.eval()
        input_tensor.requires_grad_(True)
        
        output = model(input_tensor)
        loss = F.cross_entropy(output, torch.argmax(output, dim=-1))
        
        gradients = torch.autograd.grad(loss, input_tensor, create_graph=True)[0]
        epistemic_uncertainty = torch.norm(gradients, dim=-1).mean().item()
        
        return epistemic_uncertainty
    
    def aleatoric_uncertainty(self, model_variance: torch.Tensor) -> torch.Tensor:
        """
        Calculate aleatoric (data) uncertainty from model variance output
        """
        return torch.sqrt(model_variance)

class ModelDriftDetector:
    """Detect model performance drift over time"""
    
    def __init__(self, window_size: int = 1000, drift_threshold: float = 0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.performance_history = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_performance = {}
        self.drift_alerts = []
    
    def add_performance_metric(self, model_name: str, metric_name: str, 
                             value: float, timestamp: datetime = None):
        """Add performance metric for drift detection"""
        if timestamp is None:
            timestamp = datetime.now()
        
        key = f"{model_name}_{metric_name}"
        self.performance_history[key].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Set baseline if not exists
        if key not in self.baseline_performance and len(self.performance_history[key]) >= 100:
            baseline_values = [p['value'] for p in list(self.performance_history[key])[:100]]
            self.baseline_performance[key] = {
                'mean': np.mean(baseline_values),
                'std': np.std(baseline_values)
            }
    
    def detect_drift(self, model_name: str, metric_name: str) -> Dict:
        """Detect if model performance has drifted"""
        key = f"{model_name}_{metric_name}"
        
        if key not in self.baseline_performance:
            return {"status": "insufficient_baseline_data"}
        
        if len(self.performance_history[key]) < 50:
            return {"status": "insufficient_recent_data"}
        
        # Get recent performance
        recent_values = [p['value'] for p in list(self.performance_history[key])[-50:]]
        recent_mean = np.mean(recent_values)
        
        baseline = self.baseline_performance[key]
        
        # Statistical tests for drift
        # 1. Mean shift test
        mean_shift = abs(recent_mean - baseline['mean']) / (baseline['std'] + 1e-8)
        
        # 2. Kolmogorov-Smirnov test
        baseline_values = [p['value'] for p in list(self.performance_history[key])[:100]]
        ks_statistic, ks_p_value = stats.ks_2samp(baseline_values, recent_values)
        
        # 3. Mann-Whitney U test
        mw_statistic, mw_p_value = stats.mannwhitneyu(baseline_values, recent_values, 
                                                      alternative='two-sided')
        
        # Determine drift
        drift_detected = (
            mean_shift > self.drift_threshold or 
            ks_p_value < 0.05 or 
            mw_p_value < 0.05
        )
        
        if drift_detected:
            alert = {
                'model_name': model_name,
                'metric_name': metric_name,
                'timestamp': datetime.now(),
                'severity': 'high' if mean_shift > 2 * self.drift_threshold else 'medium',
                'details': {
                    'mean_shift': float(mean_shift),
                    'ks_p_value': float(ks_p_value),
                    'mw_p_value': float(mw_p_value),
                    'baseline_mean': baseline['mean'],
                    'recent_mean': recent_mean
                }
            }
            self.drift_alerts.append(alert)
            logger.warning(f"Model drift detected: {model_name} - {metric_name}")
        
        return {
            "status": "drift_detected" if drift_detected else "no_drift",
            "mean_shift": float(mean_shift),
            "ks_p_value": float(ks_p_value),
            "mw_p_value": float(mw_p_value),
            "drift_threshold": self.drift_threshold,
            "baseline_performance": baseline,
            "recent_performance": {
                'mean': recent_mean,
                'std': np.std(recent_values)
            }
        }
    
    def get_drift_report(self, hours_back: int = 24) -> Dict:
        """Get drift detection report"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_alerts = [
            alert for alert in self.drift_alerts 
            if alert['timestamp'] > cutoff_time
        ]
        
        return {
            "total_alerts": len(recent_alerts),
            "high_severity_alerts": len([a for a in recent_alerts if a['severity'] == 'high']),
            "alerts": recent_alerts,
            "monitored_models": list(set(
                key.split('_')[0] for key in self.performance_history.keys()
            ))
        }

class PredictionQualityManager:
    """Comprehensive prediction quality management"""
    
    def __init__(self):
        self.calibrator = ConfidenceCalibrator()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.drift_detector = ModelDriftDetector()
        self.quality_metrics = defaultdict(lambda: deque(maxlen=1000))
    
    def evaluate_prediction_quality(self, model_name: str, prediction: Dict, 
                                  ground_truth: Optional[Dict] = None) -> Dict:
        """Comprehensive prediction quality evaluation"""
        quality_assessment = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'quality_score': 0.0,
            'confidence_calibrated': False,
            'uncertainty_estimated': False,
            'issues': []
        }
        
        # 1. Confidence calibration
        if 'confidence' in prediction:
            original_conf = prediction['confidence']
            calibrated_conf = self.calibrator.calibrate_confidence(
                model_name, original_conf, prediction.get('class')
            )
            quality_assessment['calibrated_confidence'] = calibrated_conf
            quality_assessment['confidence_calibrated'] = True
        
        # 2. Quality scoring
        quality_score = self._calculate_quality_score(prediction)
        quality_assessment['quality_score'] = quality_score
        
        # 3. Issue detection
        issues = self._detect_prediction_issues(prediction)
        quality_assessment['issues'] = issues
        
        # 4. Record for drift detection
        if ground_truth:
            accuracy = self._calculate_accuracy(prediction, ground_truth)
            self.drift_detector.add_performance_metric(model_name, 'accuracy', accuracy)
        
        return quality_assessment
    
    def _calculate_quality_score(self, prediction: Dict) -> float:
        """Calculate overall prediction quality score (0-1)"""
        score = 0.0
        factors = 0
        
        # Confidence factor
        if 'confidence' in prediction:
            conf = prediction['confidence']
            # Penalize very low or suspiciously high confidence
            if 0.3 <= conf <= 0.9:
                score += 1.0
            elif 0.1 <= conf < 0.3 or 0.9 < conf <= 0.95:
                score += 0.5
            factors += 1
        
        # Consistency factor (if multiple predictions)
        if 'ensemble_predictions' in prediction:
            predictions = prediction['ensemble_predictions']
            if len(predictions) > 1:
                consistency = 1.0 - np.std([p.get('confidence', 0) for p in predictions])
                score += max(0, consistency)
                factors += 1
        
        # Uncertainty factor
        if 'uncertainty' in prediction:
            uncertainty = prediction['uncertainty']
            # Lower uncertainty is better
            score += max(0, 1.0 - uncertainty)
            factors += 1
        
        return score / max(factors, 1)
    
    def _detect_prediction_issues(self, prediction: Dict) -> List[str]:
        """Detect potential issues with prediction"""
        issues = []
        
        # Low confidence
        if prediction.get('confidence', 1.0) < 0.3:
            issues.append("low_confidence")
        
        # Suspiciously high confidence
        if prediction.get('confidence', 0.0) > 0.99:
            issues.append("overconfident")
        
        # High uncertainty
        if prediction.get('uncertainty', 0.0) > 0.5:
            issues.append("high_uncertainty")
        
        # Missing required fields
        required_fields = ['confidence', 'class']
        for field in required_fields:
            if field not in prediction:
                issues.append(f"missing_{field}")
        
        return issues
    
    def _calculate_accuracy(self, prediction: Dict, ground_truth: Dict) -> float:
        """Calculate prediction accuracy"""
        pred_class = prediction.get('class')
        true_class = ground_truth.get('class')
        
        if pred_class is None or true_class is None:
            return 0.0
        
        return 1.0 if pred_class == true_class else 0.0

# Global instances
confidence_calibrator = ConfidenceCalibrator()
uncertainty_quantifier = UncertaintyQuantifier()
drift_detector = ModelDriftDetector()
quality_manager = PredictionQualityManager()