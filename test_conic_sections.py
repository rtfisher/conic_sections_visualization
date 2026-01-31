#!/usr/bin/env python3
"""
Tests for conic sections visualization
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI
import matplotlib.pyplot as plt

from conic_sections import ConicSectionVisualizer


@pytest.fixture
def visualizer():
    """Create a fresh visualizer instance for each test"""
    viz = ConicSectionVisualizer()
    yield viz
    plt.close('all')


class TestConicClassification:
    """Test conic section classification based on plane tilt"""

    def test_circle_classification(self, visualizer):
        visualizer.slider_tilt.set_val(0)
        assert visualizer.classify_conic() == "Circle"

    def test_ellipse_classification(self, visualizer):
        visualizer.slider_tilt.set_val(20)
        assert visualizer.classify_conic() == "Ellipse"

    def test_parabola_classification(self, visualizer):
        visualizer.slider_tilt.set_val(45)
        assert visualizer.classify_conic() == "Parabola"

    def test_hyperbola_classification(self, visualizer):
        visualizer.slider_tilt.set_val(60)
        assert visualizer.classify_conic() == "Hyperbola"

    def test_negative_tilt_uses_absolute_value(self, visualizer):
        visualizer.slider_tilt.set_val(-30)
        assert visualizer.classify_conic() == "Ellipse"


class TestIntersectionComputation:
    """Test intersection curve computation"""

    def test_circle_produces_closed_curve(self, visualizer):
        visualizer.slider_tilt.set_val(0)
        visualizer.slider_height.set_val(1.0)
        visualizer.on_slider_change(None)

        curves = visualizer.compute_intersection()
        assert curves is not None
        assert len(curves) == 1  # Single closed curve
        assert len(curves[0]) == 500  # Full parameterization

    def test_ellipse_produces_closed_curve(self, visualizer):
        visualizer.slider_tilt.set_val(30)
        visualizer.slider_height.set_val(1.0)
        visualizer.on_slider_change(None)

        curves = visualizer.compute_intersection()
        assert curves is not None
        assert len(curves) == 1

    def test_parabola_produces_open_curve(self, visualizer):
        visualizer.slider_tilt.set_val(45)
        visualizer.slider_height.set_val(1.0)
        visualizer.on_slider_change(None)

        curves = visualizer.compute_intersection()
        assert curves is not None
        assert len(curves) == 1
        # Parabola is open, so fewer than 500 points
        assert len(curves[0]) < 500

    def test_hyperbola_produces_two_branches(self, visualizer):
        visualizer.slider_tilt.set_val(60)
        visualizer.slider_height.set_val(1.0)
        visualizer.on_slider_change(None)

        curves = visualizer.compute_intersection()
        assert curves is not None
        assert len(curves) == 2  # Two branches

    def test_no_intersection_at_apex(self, visualizer):
        visualizer.slider_tilt.set_val(0)
        visualizer.slider_height.set_val(0)
        visualizer.on_slider_change(None)

        curves = visualizer.compute_intersection()
        assert curves is None  # Degenerate case at apex

    def test_negative_height_intersects_lower_cone(self, visualizer):
        visualizer.slider_tilt.set_val(0)
        visualizer.slider_height.set_val(-1.0)
        visualizer.on_slider_change(None)

        curves = visualizer.compute_intersection()
        assert curves is not None
        assert len(curves) == 1
        # All z values should be negative (lower cone)
        assert all(curves[0][:, 2] < 0)


class TestPlaneEquation:
    """Test plane equation computation"""

    def test_horizontal_plane_normal(self, visualizer):
        visualizer.plane_tilt = 0
        visualizer.plane_rotation = 0
        normal, d = visualizer.get_plane_equation()

        # Horizontal plane has normal pointing up
        np.testing.assert_array_almost_equal(normal, [0, 0, 1])

    def test_tilted_plane_normal(self, visualizer):
        visualizer.plane_tilt = np.pi / 4  # 45 degrees
        visualizer.plane_rotation = 0
        normal, d = visualizer.get_plane_equation()

        # Normal should be tilted
        assert normal[1] != 0  # y-component non-zero
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-10  # Unit vector

    def test_plane_passes_through_height(self, visualizer):
        visualizer.plane_tilt = 0
        visualizer.plane_rotation = 0
        visualizer.plane_height = 2.0
        normal, d = visualizer.get_plane_equation()

        # For horizontal plane, d should equal height
        assert abs(d - 2.0) < 1e-10


class Test2DProjection:
    """Test projection from 3D to 2D plane coordinates"""

    def test_circle_projects_to_circle(self, visualizer):
        visualizer.slider_tilt.set_val(0)
        visualizer.slider_height.set_val(1.0)
        visualizer.on_slider_change(None)

        curves_3d = visualizer.compute_intersection()
        curves_2d = visualizer.project_to_2d(curves_3d)

        assert curves_2d is not None
        assert len(curves_2d) == 1

        # Check that 2D points form a circle (constant radius from center)
        pts = curves_2d[0]
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        assert np.std(radii) < 0.01  # Small variance = circle

    def test_projection_preserves_curve_count(self, visualizer):
        visualizer.slider_tilt.set_val(70)
        visualizer.slider_height.set_val(1.0)
        visualizer.on_slider_change(None)

        curves_3d = visualizer.compute_intersection()
        curves_2d = visualizer.project_to_2d(curves_3d)

        assert len(curves_2d) == len(curves_3d)


class TestVisualizationIntegrity:
    """Test that visualization updates without errors"""

    def test_update_plot_runs_without_error(self, visualizer):
        # Should not raise any exceptions
        visualizer.update_plot()

    def test_slider_changes_update_plot(self, visualizer):
        visualizer.slider_tilt.set_val(30)
        visualizer.on_slider_change(None)
        # Should complete without error

    def test_reset_restores_defaults(self, visualizer):
        visualizer.slider_tilt.set_val(60)
        visualizer.slider_height.set_val(2.0)
        visualizer.reset(None)

        assert visualizer.slider_tilt.val == 0
        assert visualizer.slider_height.val == 1.0

    def test_full_tilt_range(self, visualizer):
        """Test that all valid tilt angles work"""
        for tilt in range(-89, 90, 10):
            visualizer.slider_tilt.set_val(tilt)
            visualizer.on_slider_change(None)
            # Should not raise


class TestConeGeometry:
    """Test cone geometry parameters"""

    def test_cone_angle_is_45_degrees(self, visualizer):
        assert abs(visualizer.cone_angle - np.pi/4) < 1e-10

    def test_cone_height(self, visualizer):
        assert visualizer.cone_height == 3.0

    def test_cone_mesh_dimensions(self, visualizer):
        assert visualizer.cone_X.shape == visualizer.cone_Y.shape
        assert visualizer.cone_Y.shape == visualizer.cone_Z.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
