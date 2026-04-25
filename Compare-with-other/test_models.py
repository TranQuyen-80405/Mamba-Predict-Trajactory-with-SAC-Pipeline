import unittest
import torch
from trajectory_models import RecurrentTrajectoryPredictor, TransformerTrajectoryPredictor, MambaTrajectoryPredictor
from risk_models import MLPRiskPredictor, OccupancyHeatmapPredictor, SafetyValueEstimator

class TestTrajectoryForecasting(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.T = 10
        self.C = 64
        self.H = 16
        self.W = 16
        self.horizon = 10
        self.hidden_size = 128
        self.d_model = 128
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dummy features from the BEV decoder (PointPillars)
        self.dummy_bev = torch.randn(self.B, self.T, self.C, self.H, self.W).to(self.device)

    def test_recurrent_baseline(self):
        model = RecurrentTrajectoryPredictor(in_channels=self.C, hidden_size=self.hidden_size, horizon=self.horizon, rnn_type='LSTM').to(self.device)
        out = model(self.dummy_bev)
        self.assertEqual(out.shape, (self.B, self.horizon, 2), "LSTM Trajectory shape mismatch.")

    def test_transformer_baseline(self):
        model = TransformerTrajectoryPredictor(in_channels=self.C, d_model=self.d_model, nhead=4, num_layers=2, horizon=self.horizon).to(self.device)
        out = model(self.dummy_bev)
        self.assertEqual(out.shape, (self.B, self.horizon, 2), "Transformer Trajectory shape mismatch.")

    def test_mamba_baseline(self):
        try:
            model = MambaTrajectoryPredictor(in_channels=self.C, d_model=self.d_model, d_state=16, d_conv=4, expand=2, horizon=self.horizon).to(self.device)
            out = model(self.dummy_bev)
            self.assertEqual(out.shape, (self.B, self.horizon, 2), "Mamba Trajectory shape mismatch.")
        except ImportError:
            self.skipTest("mamba_ssm is not installed, skipping Mamba test.")


class TestRiskPrediction(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.horizon = 10
        self.hidden_size = 128
        self.C = 64
        self.H = 16
        self.W = 16
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.dummy_hidden = torch.randn(self.B, self.hidden_size).to(self.device)
        self.dummy_bev_feature = torch.randn(self.B, self.C, self.H, self.W).to(self.device)
        self.dummy_traj = (torch.rand(self.B, self.horizon, 2) * 2 - 1).to(self.device) # Normalized to [-1, 1]

    def test_mlp_risk(self):
        model = MLPRiskPredictor(hidden_size=self.hidden_size).to(self.device)
        out = model(self.dummy_hidden)
        self.assertEqual(out.shape, (self.B, 1), "MLP Risk shape mismatch.")
        self.assertTrue((out >= 0).all() and (out <= 1).all(), "MLP Risk output must be in [0, 1].")

    def test_occupancy_heatmap(self):
        model = OccupancyHeatmapPredictor(in_channels=self.C).to(self.device)
        risk, heatmap = model(self.dummy_bev_feature, self.dummy_traj)
        self.assertEqual(heatmap.shape, (self.B, 1, self.H, self.W), "Heatmap shape mismatch.")
        self.assertEqual(risk.shape, (self.B, 1), "Risk shape mismatch.")

    def test_safety_value(self):
        model = SafetyValueEstimator(hidden_size=self.hidden_size).to(self.device)
        out = model(self.dummy_hidden)
        self.assertEqual(out.shape, (self.B, 1), "Safety Value shape mismatch.")


if __name__ == '__main__':
    unittest.main()
