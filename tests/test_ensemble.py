from app.services.image_service import combine_ai_probabilities


def test_combine_excludes_failures_and_renormalizes():
    per_model = [
        {"success": True, "detector": "a", "ai_probability": 0.8},
        {"success": False, "detector": "b", "ai_probability": 0.1},
        {"success": True, "detector": "c", "ai_probability": 0.2},
    ]

    ai, _ = combine_ai_probabilities(per_model, {"a": 3.0, "b": 100.0, "c": 1.0})
    # weights become a=0.75, c=0.25
    assert abs(ai - (0.75 * 0.8 + 0.25 * 0.2)) < 1e-6


def test_combine_all_failures_returns_zero():
    per_model = [
        {"success": False, "detector": "a", "ai_probability": 0.9},
        {"success": False, "detector": "b", "ai_probability": 0.9},
    ]
    ai, _ = combine_ai_probabilities(per_model, {"a": 1.0, "b": 1.0})
    assert ai == 0.0

