import math

import numpy as np

from app.services.audio_service import aggregate_audio_scores, chunk_audio


def test_chunk_audio_returns_single_chunk_for_short_audio():
    audio = np.zeros(16000 * 2, dtype=np.float32)

    chunks = chunk_audio(
        audio=audio,
        sample_rate=16000,
        chunk_seconds=5.0,
        overlap_seconds=1.0,
        min_chunk_seconds=2.5,
    )

    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert math.isclose(chunks[0]["end_seconds"], 2.0, rel_tol=1e-6)


def test_aggregate_audio_scores_uses_max_mean_hybrid():
    results = [
        {"probability": 0.2},
        {"probability": 0.4},
        {"probability": 0.9},
    ]

    aggregated = aggregate_audio_scores(results)
    expected_ai = (0.7 * 0.9) + (0.3 * ((0.2 + 0.4 + 0.9) / 3))

    assert math.isclose(aggregated["ai_probability"], expected_ai, rel_tol=1e-6)
    assert math.isclose(aggregated["human_probability"], 1.0 - expected_ai, rel_tol=1e-6)
