import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from vibeguard_ai import EdgeProcessor, EdgeConfig

def visualize_vibeguard_demo():
    """Run VibeGuard AI with visualization"""
    
    # Initialize edge processor
    config = EdgeConfig(
        sampling_rate=100,
        window_size=256,
        anomaly_threshold=3.0
    )
    processor = EdgeProcessor("VG-001", config)
    
    # Generate synthetic vibration data
    duration = 10  # seconds
    t = np.linspace(0, duration, duration * config.sampling_rate)
    
    # Normal vibration: baseline + harmonic + noise
    baseline = 2.0
    normal_vibration = (
        baseline + 
        0.5 * np.sin(2 * np.pi * 5 * t) +  # 5 Hz component
        0.3 * np.sin(2 * np.pi * 12 * t) +  # 12 Hz component
        0.3 * np.random.randn(len(t))  # Random noise
    )
    
    # Add anomalies (simulating tampering)
    anomaly_regions = [
        (300, 320, "Cut attempt"),
        (600, 650, "Drilling"),
        (850, 870, "Impact")
    ]
    
    for start, end, label in anomaly_regions:
        # Add high-frequency vibration burst
        anomaly_signal = 5.0 * np.sin(2 * np.pi * 25 * t[start:end])
        normal_vibration[start:end] += anomaly_signal
    
    # Process the data stream
    results = []
    anomaly_scores = []
    timestamps = []
    
    for i, sample in enumerate(normal_vibration):
        result = processor.add_sample(sample)
        if result:
            results.append(result)
            anomaly_scores.append(result['anomaly_score'])
            timestamps.append(i / config.sampling_rate)
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('VibeGuard AI - Vibration Monitoring & Anomaly Detection', fontsize=16)
    
    # Plot 1: Raw vibration signal
    ax1 = axes[0]
    ax1.plot(t, normal_vibration, 'b-', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel('Vibration (units)')
    ax1.set_title('Raw Sensor Data')
    ax1.grid(True, alpha=0.3)
    
    # Highlight anomaly regions
    for start, end, label in anomaly_regions:
        ax1.axvspan(start/config.sampling_rate, end/config.sampling_rate, 
                   alpha=0.3, color='red', label=label)
    ax1.legend(loc='upper right')
    
    # Plot 2: Spectrogram
    ax2 = axes[1]
    f, t_spec, Sxx = signal.spectrogram(normal_vibration, config.sampling_rate, 
                                        nperseg=256, noverlap=128)
    pcm = ax2.pcolormesh(t_spec, f[:50], 10 * np.log10(Sxx[:50]), 
                         shading='gouraud', cmap='viridis')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram')
    
    # Plot 3: Anomaly scores
    ax3 = axes[2]
    ax3.plot(timestamps, anomaly_scores, 'g-', linewidth=2, label='Anomaly Score')
    ax3.axhline(y=config.anomaly_threshold, color='r', linestyle='--', 
                label=f'Threshold ({config.anomaly_threshold})')
    ax3.fill_between(timestamps, 0, anomaly_scores, 
                     where=np.array(anomaly_scores) > config.anomaly_threshold,
                     color='red', alpha=0.3, label='Detected Anomaly')
    ax3.set_ylabel('Anomaly Score')
    ax3.set_title('AI Detection Output')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature evolution
    ax4 = axes[3]
    rms_values = [r['features']['rms'] for r in results]
    dominant_freqs = [r['features']['dominant_freq'] for r in results]
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(timestamps, rms_values, 'b-', label='RMS Value')
    line2 = ax4_twin.plot(timestamps, dominant_freqs, 'r-', label='Dominant Freq')
    
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('RMS Value', color='b')
    ax4_twin.set_ylabel('Dominant Frequency (Hz)', color='r')
    ax4.set_title('Extracted Features')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nVibeGuard AI Analysis Summary")
    print("=" * 40)
    print(f"Total samples processed: {len(normal_vibration)}")
    print(f"Windows analyzed: {len(results)}")
    print(f"Anomalies detected: {sum(1 for r in results if r['is_anomaly'])}")
    print(f"Alerts triggered: {sum(1 for r in results if r['alert'])}")
    print(f"Average anomaly score: {np.mean(anomaly_scores):.2f}")
    print(f"Max anomaly score: {np.max(anomaly_scores):.2f}")
    
    # Show feature importance
    print("\nFeature Statistics (Normal vs Anomaly):")
    normal_features = [r['features'] for r in results if not r['is_anomaly']]
    anomaly_features = [r['features'] for r in results if r['is_anomaly']]
    
    if normal_features and anomaly_features:
        for feature in ['rms', 'dominant_freq', 'spectral_centroid']:
            normal_vals = [f[feature] for f in normal_features]
            anomaly_vals = [f[feature] for f in anomaly_features]
            print(f"\n{feature}:")
            print(f"  Normal: {np.mean(normal_vals):.2f} ± {np.std(normal_vals):.2f}")
            print(f"  Anomaly: {np.mean(anomaly_vals):.2f} ± {np.std(anomaly_vals):.2f}")

if __name__ == "__main__":
    # Check if matplotlib is installed
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
        visualize_vibeguard_demo()
    except ImportError:
        print("Please install matplotlib to run the visualization:")
        print("pip install matplotlib")
        print("\nAlternatively, run the basic simulation with:")
        print("python vibeguard_ai.py")