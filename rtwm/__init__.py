"""
rtwm – Real-Time audio watermarking package.

Public API:

    WatermarkEmbedder  – real-time spread-spectrum TX
    WatermarkDetector  – offline RX: scan / analyze / verify recordings
    FrameHit, Report   – detector result types
"""
from .detector import FrameHit, Report, WatermarkDetector
from .embedder import WatermarkEmbedder

__all__: list[str] = ["WatermarkEmbedder", "WatermarkDetector", "FrameHit", "Report"]
