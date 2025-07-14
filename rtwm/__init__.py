"""
rtwm – Real-Time audio watermarking package.

The public API re-exports the main classes:

    WatermarkEmbedder  – real-time spread-spectrum TX
    WatermarkDetector  – offline RX / verifier
"""
from .embedder import WatermarkEmbedder
from .detector import WatermarkDetector

__all__: list[str] = ["WatermarkEmbedder", "WatermarkDetector"]
