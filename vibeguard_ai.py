# VibeGuard AI - Edge Processing Implementation
# Technical guide for anomaly detection on vibration data

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

# Configuration for edge deployment
@dataclass
class EdgeConfig:
    sampling_rate: int = 100  # Hz
    window_size: int = 256    # samples
    overlap: float = 0.5      # 50% overlap
    anomaly_threshold: float = 3.0  # standard deviations
    learning_rate: float = 0.01
    memory_size: int = 1000   # historical windows to keep

class VibrationFeatureExtractor:
    """Extract features from vibration data for anomaly detection"""
    
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate
        
    def extract_features(self, data: np.ndarray) -> dict:
        """Extract time and frequency domain features"""
        features = {}
        
        # Time domain features
        features['rms'] = np.sqrt(np.mean(data**2))
        features['peak'] = np.max(np.abs(data))
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
        features['kurtosis'] = self._kurtosis(data)
        features['zero_crossings'] = self._zero_crossing_rate(data)
        
        # Frequency domain features
        freqs, magnitude = self._compute_fft(data)
        features['dominant_freq'] = freqs[np.argmax(magnitude)]
        features['spectral_centroid'] = self._spectral_centroid(freqs, magnitude)
        features['spectral_spread'] = self._spectral_spread(freqs, magnitude, features['spectral_centroid'])
        
        # Energy in frequency bands
        features['energy_low'] = self._band_energy(magnitude, freqs, 0, 10)
        features['energy_mid'] = self._band_energy(magnitude, freqs, 10, 30)
        features['energy_high'] = self._band_energy(magnitude, freqs, 30, 50)
        
        return features
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis (tailedness of distribution)"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _zero_crossing_rate(self, data: np.ndarray) -> int:
        """Count zero crossings in the signal"""
        return np.sum(np.diff(np.sign(data)) != 0)
    
    def _compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT and return frequencies and magnitudes"""
        n = len(data)
        fft_vals = fft(data)
        freqs = fftfreq(n, 1/self.sampling_rate)
        magnitude = np.abs(fft_vals[:n//2])
        freqs = freqs[:n//2]
        return freqs, magnitude
    
    def _spectral_centroid(self, freqs: np.ndarray, magnitude: np.ndarray) -> float:
        """Calculate spectral centroid (center of mass of spectrum)"""
        return np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
    
    def _spectral_spread(self, freqs: np.ndarray, magnitude: np.ndarray, centroid: float) -> float:
        """Calculate spectral spread (bandwidth)"""
        return np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)) if np.sum(magnitude) > 0 else 0
    
    def _band_energy(self, magnitude: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
        """Calculate energy in a frequency band"""
        mask = (freqs >= low) & (freqs < high)
        return np.sum(magnitude[mask] ** 2)

class OnlineAnomalyDetector:
    """Real-time anomaly detection using adaptive thresholding"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.feature_extractor = VibrationFeatureExtractor(config.sampling_rate)
        self.feature_memory = deque(maxlen=config.memory_size)
        self.baseline_stats = {}
        self.is_learning = True
        self.min_samples = 100  # Minimum samples before switching to detection mode
        
    def process_window(self, data: np.ndarray) -> dict:
        """Process a window of vibration data"""
        # Extract features
        features = self.feature_extractor.extract_features(data)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(features)
        
        # Update baseline if in learning mode or low anomaly
        if self.is_learning or anomaly_score < self.config.anomaly_threshold:
            self._update_baseline(features)
        
        # Switch to detection mode after enough samples
        if self.is_learning and len(self.feature_memory) >= self.min_samples:
            self.is_learning = False
        
        return {
            'features': features,
            'anomaly_score': anomaly_score,
            'is_anomaly': anomaly_score > self.config.anomaly_threshold,
            'threshold': self.config.anomaly_threshold,
            'mode': 'learning' if self.is_learning else 'detection'
        }
    
    def _calculate_anomaly_score(self, features: dict) -> float:
        """Calculate anomaly score using Mahalanobis distance"""
        if not self.baseline_stats:
            return 0.0
        
        scores = []
        for feature_name, value in features.items():
            if feature_name in self.baseline_stats:
                stats = self.baseline_stats[feature_name]
                if stats['std'] > 0:
                    z_score = abs(value - stats['mean']) / stats['std']
                    scores.append(z_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _update_baseline(self, features: dict):
        """Update baseline statistics with exponential moving average"""
        self.feature_memory.append(features)
        
        # Calculate statistics for each feature
        for feature_name in features:
            values = [f[feature_name] for f in self.feature_memory if feature_name in f]
            
            if values:
                new_mean = np.mean(values)
                new_std = np.std(values)
                
                if feature_name not in self.baseline_stats:
                    self.baseline_stats[feature_name] = {'mean': new_mean, 'std': new_std}
                else:
                    # Exponential moving average update
                    old_stats = self.baseline_stats[feature_name]
                    alpha = self.config.learning_rate
                    old_stats['mean'] = (1 - alpha) * old_stats['mean'] + alpha * new_mean
                    old_stats['std'] = (1 - alpha) * old_stats['std'] + alpha * new_std

class EdgeProcessor:
    """Main edge processing class for deployment on embedded devices"""
    
    def __init__(self, sensor_id: str, config: EdgeConfig):
        self.sensor_id = sensor_id
        self.config = config
        self.detector = OnlineAnomalyDetector(config)
        self.buffer = deque(maxlen=config.window_size)
        self.alert_cooldown = 0
        self.alert_cooldown_period = 50  # windows
        
    def add_sample(self, value: float) -> Optional[dict]:
        """Add a single sample and process if window is full"""
        self.buffer.append(value)
        
        if len(self.buffer) == self.config.window_size:
            return self._process_buffer()
        return None
    
    def _process_buffer(self) -> dict:
        """Process the current buffer"""
        data = np.array(self.buffer)
        
        # Apply preprocessing
        data = self._preprocess(data)
        
        # Run anomaly detection
        result = self.detector.process_window(data)
        
        # Add sensor metadata
        result['sensor_id'] = self.sensor_id
        result['timestamp'] = self._get_timestamp()
        
        # Handle alerting logic
        if result['is_anomaly'] and self.alert_cooldown == 0:
            result['alert'] = True
            self.alert_cooldown = self.alert_cooldown_period
        else:
            result['alert'] = False
            if self.alert_cooldown > 0:
                self.alert_cooldown -= 1
        
        # Slide window with overlap
        slide_amount = int(self.config.window_size * (1 - self.config.overlap))
        for _ in range(slide_amount):
            self.buffer.popleft()
        
        return result
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to raw data"""
        # Remove DC offset
        data = data - np.mean(data)
        
        # Apply high-pass filter to remove low-frequency drift
        b, a = signal.butter(4, 0.5, 'high', fs=self.config.sampling_rate)
        data = signal.filtfilt(b, a, data)
        
        return data
    
    def _get_timestamp(self) -> int:
        """Get current timestamp (would use RTC on embedded device)"""
        import time
        return int(time.time() * 1000)  # milliseconds
    
    def get_status(self) -> dict:
        """Get current processor status"""
        return {
            'sensor_id': self.sensor_id,
            'mode': self.detector.is_learning,
            'samples_processed': len(self.detector.feature_memory),
            'baseline_features': len(self.detector.baseline_stats),
            'alert_cooldown': self.alert_cooldown
        }
    
    def export_model(self) -> str:
        """Export model parameters for persistence"""
        model_data = {
            'sensor_id': self.sensor_id,
            'baseline_stats': self.detector.baseline_stats,
            'config': {
                'sampling_rate': self.config.sampling_rate,
                'window_size': self.config.window_size,
                'anomaly_threshold': self.config.anomaly_threshold
            }
        }
        return json.dumps(model_data)
    
    def load_model(self, model_json: str):
        """Load previously trained model"""
        model_data = json.loads(model_json)
        self.detector.baseline_stats = model_data['baseline_stats']
        self.detector.is_learning = False

# Example usage for embedded deployment
def simulate_edge_deployment():
    """Simulate edge deployment with synthetic data"""
    
    # Initialize edge processor
    config = EdgeConfig(
        sampling_rate=100,
        window_size=256,
        anomaly_threshold=3.0
    )
    processor = EdgeProcessor("VG-001", config)
    
    # Simulate data stream
    t = np.linspace(0, 10, 1000)
    normal_vibration = 2.0 + 0.5 * np.sin(2 * np.pi * 5 * t) + 0.3 * np.random.randn(len(t))
    
    # Add some anomalies
    anomaly_indices = [300, 600, 850]
    for idx in anomaly_indices:
        normal_vibration[idx:idx+20] += 5.0 * np.sin(2 * np.pi * 25 * t[idx:idx+20])
    
    # Process stream
    results = []
    for i, sample in enumerate(normal_vibration):
        result = processor.add_sample(sample)
        if result:
            results.append(result)
            if result['alert']:
                print(f"ALERT at sample {i}: Anomaly Score = {result['anomaly_score']:.2f}")
    
    # Export trained model
    model_json = processor.export_model()
    print(f"\nTrained model size: {len(model_json)} bytes")
    
    return results

# Communication protocol for integration with main system
class EdgeCommunicator:
    """Handle communication between edge device and central system"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.message_queue = deque(maxlen=100)
        
    def create_message(self, processor_result: dict) -> dict:
        """Create message for transmission"""
        message = {
            'device_id': self.device_id,
            'sensor_id': processor_result['sensor_id'],
            'timestamp': processor_result['timestamp'],
            'anomaly_score': round(processor_result['anomaly_score'], 3),
            'is_anomaly': processor_result['is_anomaly'],
            'alert': processor_result['alert'],
            'features': {k: round(v, 3) for k, v in processor_result['features'].items()}
        }
        
        # Add to queue for batch transmission
        self.message_queue.append(message)
        
        return message
    
    def get_batch_message(self) -> dict:
        """Get batch of messages for efficient transmission"""
        messages = list(self.message_queue)
        self.message_queue.clear()
        
        return {
            'device_id': self.device_id,
            'message_count': len(messages),
            'messages': messages
        }

# Model optimization for MCU deployment
def optimize_for_mcu(model_params: dict) -> dict:
    """Optimize model parameters for microcontroller deployment"""
    
    # Quantize floating point values to fixed point
    optimized = {}
    
    for feature, stats in model_params.items():
        optimized[feature] = {
            'mean': int(stats['mean'] * 1000) / 1000,  # 3 decimal places
            'std': int(stats['std'] * 1000) / 1000
        }
    
    return optimized

if __name__ == "__main__":
    # Run simulation
    print("VibeGuard AI - Edge Processing Simulation")
    print("=========================================")
    results = simulate_edge_deployment()
    print(f"\nProcessed {len(results)} windows")
    anomalies = sum(1 for r in results if r['is_anomaly'])
    print(f"Detected {anomalies} anomalous windows")