"""Tests for hardware detection and tier selection."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from pixelpoison.detect.hardware import (
    HardwareInfo,
    _select_tier,
    detect_hardware,
    TIER_DESCRIPTIONS,
)


class TestSelectTier:
    """Test tier selection logic."""

    def test_cuda_high_vram_tier3(self):
        assert _select_tier("cuda", 24.0) == 3

    def test_cuda_medium_vram_tier2(self):
        assert _select_tier("cuda", 12.0) == 2

    def test_cuda_low_vram_tier1(self):
        assert _select_tier("cuda", 4.0) == 1

    def test_cuda_boundary_20gb_tier3(self):
        assert _select_tier("cuda", 20.0) == 3

    def test_cuda_boundary_8gb_tier2(self):
        assert _select_tier("cuda", 8.0) == 2

    def test_mps_high_memory_tier3(self):
        assert _select_tier("mps", 36.0) == 3

    def test_mps_medium_memory_tier2(self):
        assert _select_tier("mps", 18.0) == 2

    def test_mps_low_memory_tier1(self):
        assert _select_tier("mps", 8.0) == 1

    def test_mps_boundary_32gb_tier3(self):
        assert _select_tier("mps", 32.0) == 3

    def test_mps_boundary_16gb_tier2(self):
        assert _select_tier("mps", 16.0) == 2

    def test_cpu_always_tier1(self):
        assert _select_tier("cpu", 64.0) == 1


class TestDetectHardware:
    """Test hardware detection with mocked torch backends."""

    @patch("pixelpoison.detect.hardware.torch")
    def test_cuda_detected(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        props = MagicMock()
        props.total_mem = 24 * (1024**3)  # 24GB
        props.name = "NVIDIA RTX 4090"
        mock_torch.cuda.get_device_properties.return_value = props

        hw = detect_hardware()
        assert hw.device == "cuda"
        assert hw.tier == 3
        assert hw.device_name == "NVIDIA RTX 4090"

    @patch("pixelpoison.detect.hardware._get_system_memory_gb", return_value=18.0)
    @patch("pixelpoison.detect.hardware.torch")
    def test_mps_detected(self, mock_torch, mock_mem):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        hw = detect_hardware()
        assert hw.device == "mps"
        assert hw.tier == 2

    @patch("pixelpoison.detect.hardware._get_system_memory_gb", return_value=8.0)
    @patch("pixelpoison.detect.hardware.torch")
    def test_cpu_fallback(self, mock_torch, mock_mem):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        hw = detect_hardware()
        assert hw.device == "cpu"
        assert hw.tier == 1

    def test_tier_descriptions_complete(self):
        for tier in [1, 2, 3]:
            assert tier in TIER_DESCRIPTIONS

    def test_hardware_info_dataclass(self):
        hw = HardwareInfo(device="cpu", device_name="Test", memory_gb=8.0, tier=1)
        assert hw.device == "cpu"
        assert hw.tier == 1
