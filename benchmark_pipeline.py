
import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('PointPillars_module'))

from PointPillars_module.models.full_pipeline_risk_traj import FullPipelineRiskAndTraj
from PointPillars_module.models.mamba_temporal import MambaTemporal
from PointPillars_module.models.temporal_encoders import LSTMTemporal, TransformerEncoderTemporal
from PointPillars_module.types import NeckFeatureOutput

class MockPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Parameter(torch.ones(1)) # Dummy param
    def extract_neck_forward(self, frame):
        # frame is a list of tensors
        B = len(frame)
        feat = torch.randn(B, 384, 248, 216, device=self.model.device)
        return NeckFeatureOutput(feature=feat, batch_size=B, channels=384, height=248, width=216, device=str(feat.device))
    def extract_neck(self, frame):
        return self.extract_neck_forward(frame)

def benchmark_pipeline(model_type, device, num_runs=10, warmup=5):
    pp = MockPP().to(device)
    token_dim = 256
    
    if model_type == 'mamba':
        temporal = MambaTemporal(d_model=token_dim, backend='mamba')
    elif model_type == 'gru':
        temporal = MambaTemporal(d_model=token_dim, backend='gru')
    elif model_type == 'lstm':
        temporal = LSTMTemporal(d_model=token_dim)
    elif model_type == 'transformer':
        temporal = TransformerEncoderTemporal(d_model=token_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = FullPipelineRiskAndTraj(pp, mamba=temporal.to(device), token_dim=token_dim).to(device)
    model.eval()
    
    # Input: List of frames (T=10), each frame is a list of batch items (B=1)
    T, B = 10, 1
    dummy_points = torch.randn(100, 4).to(device)
    pts_seq_bt = [[dummy_points for _ in range(B)] for _ in range(T)]
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(pts_seq_bt)
            
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(pts_seq_bt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            
    return np.mean(latencies)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking Pipeline on: {device}")
    
    model_types = ['mamba', 'transformer', 'gru', 'lstm']
    
    print("\n| Model (2-head Pipeline) | Avg Latency (ms/sample) |")
    print("|-------------------------|-------------------------|")
    
    for mt in model_types:
        try:
            avg_lat = benchmark_pipeline(mt, device)
            print(f"| {mt:23} | {avg_lat:23.4f} |")
        except Exception as e:
            import traceback
            print(f"| {mt:23} | Error: {str(e)} |")
            # traceback.print_exc()

if __name__ == '__main__':
    main()
