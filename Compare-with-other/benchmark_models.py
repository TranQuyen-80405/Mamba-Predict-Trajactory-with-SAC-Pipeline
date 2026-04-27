
import torch
import time
import numpy as np
from trajectory_models import RecurrentTrajectoryPredictor, TransformerTrajectoryPredictor, MambaTrajectoryPredictor
from risk_models import MLPRiskPredictor, OccupancyHeatmapPredictor, SafetyValueEstimator

def benchmark_model(model, input_data, num_runs=10, warmup=10, task='trajectory'):
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if task == 'risk' and isinstance(model, OccupancyHeatmapPredictor):
                traj = torch.randn(input_data.shape[0], 10, 2).to(device)
                _ = model(input_data, traj)
            else:
                _ = model(input_data)
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            if task == 'risk' and isinstance(model, OccupancyHeatmapPredictor):
                traj = torch.randn(input_data.shape[0], 10, 2).to(device)
                _ = model(input_data, traj)
            else:
                _ = model(input_data)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
    return np.mean(latencies)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on: {device}")
    
    B, T, C, H, W = 1, 40, 384, 16, 16
    dummy_bev = torch.randn(B, T, C, H, W).to(device)
    
    models = {
        'trajectory': {
            'lstm': RecurrentTrajectoryPredictor(in_channels=C, hidden_size=128, horizon=10).to(device),
            'transformer': TransformerTrajectoryPredictor(in_channels=C, d_model=128, nhead=4, num_layers=2, horizon=10).to(device),
            'mamba': MambaTrajectoryPredictor(in_channels=C, d_model=128, d_state=16, d_conv=4, expand=2, horizon=10).to(device),
        },
        'risk': {
            'mlp': MLPRiskPredictor(in_channels=C, hidden_size=128).to(device),
            'heatmap': OccupancyHeatmapPredictor(in_channels=C).to(device),
            'safety': SafetyValueEstimator(in_channels=C, hidden_size=128).to(device),
        }
    }
    
    results = []
    
    print("\n| Task | Model | Avg Latency (ms/sample) |")
    print("|------|-------|-------------------------|")
    
    for task, task_models in models.items():
        for name, model in task_models.items():
            avg_lat = benchmark_model(model, dummy_bev, task=task)
            print(f"| {task:10} | {name:11} | {avg_lat:23.4f} |")
            results.append({'task': task, 'model': name, 'latency_ms': avg_lat})

if __name__ == '__main__':
    main()
