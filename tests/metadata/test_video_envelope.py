"""Tests for the VideoEnvelope capability envelope (geometry + budget arithmetic)."""

from nodetool.metadata.types import VideoEnvelope, VideoModel


def ltx_envelope() -> VideoEnvelope:
    # Frame lattice 8k+1, dims multiple of 32 — the LTX-style constraints.
    return VideoEnvelope(
        fps=24,
        min_frames=9,
        max_frames=257,
        frame_step=8,
        frame_base=1,
        dim_multiple=32,
        min_dim=256,
        max_dim=1280,
        max_px=856064,
        audio=True,
        steps_default=8,
        est_ref={"w": 768, "h": 512, "frames": 121, "steps": 8, "total_s": 17.8},
    )


class TestSnapFrames:
    def test_on_lattice_unchanged(self):
        assert ltx_envelope().snap_frames(121) == 121

    def test_off_lattice_snaps_to_8k_plus_1(self):
        env = ltx_envelope()
        snapped = env.snap_frames(120)
        assert (snapped - 1) % 8 == 0
        assert abs(snapped - 120) <= 4

    def test_clamps_to_range(self):
        env = ltx_envelope()
        assert env.snap_frames(1) >= 9
        assert env.snap_frames(10_000) <= 257


class TestSnapDimension:
    def test_multiple_unchanged(self):
        assert ltx_envelope().snap_dimension(768) == 768

    def test_snaps_to_multiple(self):
        assert ltx_envelope().snap_dimension(770) % 32 == 0

    def test_clamps(self):
        env = ltx_envelope()
        assert env.snap_dimension(10) >= 256
        assert env.snap_dimension(99_999) <= 1280


class TestValidateGeometry:
    def test_valid_geometry_passes(self):
        assert ltx_envelope().validate_geometry(768, 512, 121) == []

    def test_off_lattice_frames_reported(self):
        errors = ltx_envelope().validate_geometry(768, 512, 120)
        assert any("lattice" in e for e in errors)

    def test_bad_dim_multiple_reported(self):
        errors = ltx_envelope().validate_geometry(770, 512, 121)
        assert any("multiple of 32" in e for e in errors)

    def test_joint_px_frames_budget_is_the_product(self):
        """Each dimension can pass its own cap while the product does not —
        the H3 lesson: two 19-minute OOMs at per-dimension-legal geometry."""
        env = VideoEnvelope(
            min_frames=9,
            max_frames=345,
            frame_step=1,
            frame_base=0,
            dim_multiple=16,
            max_dim=1280,
            px_frames_proven=179_000_000,
        )
        # 1216x704x209 fits; 1216x704x226 exceeds the proven budget.
        assert env.validate_geometry(1216, 704, 209) == []
        errors = env.validate_geometry(1216, 704, 226)
        assert any("joint memory budget" in e for e in errors)

    def test_empty_envelope_accepts_anything(self):
        assert VideoEnvelope().validate_geometry(123, 457, 7) == []


class TestEstimate:
    def test_scales_by_voxels_and_steps(self):
        env = ltx_envelope()
        base = env.estimate_seconds(768, 512, 121, steps=8)
        assert base is not None and abs(base - 17.8) < 1e-6
        double_steps = env.estimate_seconds(768, 512, 121, steps=16)
        assert double_steps is not None and abs(double_steps - 35.6) < 1e-6

    def test_no_reference_returns_none(self):
        assert VideoEnvelope().estimate_seconds(768, 512, 121) is None


def test_video_model_carries_envelope():
    model = VideoModel(
        id="ltx-2.5",
        name="LTX 2.5",
        capabilities=["text_to_video", "audio"],
        video=ltx_envelope(),
    )
    dumped = model.model_dump()
    assert dumped["capabilities"] == ["text_to_video", "audio"]
    assert dumped["video"]["frame_step"] == 8
