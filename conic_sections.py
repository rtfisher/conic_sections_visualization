#!/usr/bin/env python3
"""
Interactive Conic Sections Visualization
Shows conic sections as slices through a double cone with adjustable cutting plane
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

class ConicSectionVisualizer:
    def __init__(self):
        # Create figure with two subplots
        self.fig = plt.figure(figsize=(16, 8))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)
        
        # Cone parameters
        self.cone_height = 3.0
        self.cone_angle = np.pi / 4  # 45 degrees
        
        # Plane parameters (ax + by + cz = d form)
        self.plane_tilt = 0.0  # rotation around x-axis
        self.plane_rotation = 0.0  # rotation around z-axis
        self.plane_height = 1.0  # default non-zero to show intersection
        
        # Generate cone
        self.setup_cone()
        
        # Setup UI
        self.setup_sliders()
        
        # Initial plot
        self.update_plot()
        
    def setup_cone(self):
        """Generate the double cone mesh"""
        u = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(-self.cone_height, self.cone_height, 50)
        U, Z = np.meshgrid(u, z)
        
        # Radius varies linearly with height
        R = np.abs(Z) * np.tan(self.cone_angle)
        
        self.cone_X = R * np.cos(U)
        self.cone_Y = R * np.sin(U)
        self.cone_Z = Z
        
    def setup_sliders(self):
        """Create interactive sliders"""
        plt.subplots_adjust(bottom=0.25)
        
        # Slider axes
        ax_tilt = plt.axes([0.15, 0.15, 0.3, 0.03])
        ax_rotation = plt.axes([0.15, 0.10, 0.3, 0.03])
        ax_height = plt.axes([0.15, 0.05, 0.3, 0.03])
        
        # Create sliders
        self.slider_tilt = Slider(ax_tilt, 'Tilt', -89, 89, valinit=0, valstep=1)
        self.slider_rotation = Slider(ax_rotation, 'Rotation', 0, 360, valinit=0, valstep=1)
        self.slider_height = Slider(ax_height, 'Height', -2.5, 2.5, valinit=1.0, valstep=0.1)
        
        # Connect update function
        self.slider_tilt.on_changed(self.on_slider_change)
        self.slider_rotation.on_changed(self.on_slider_change)
        self.slider_height.on_changed(self.on_slider_change)
        
        # Reset button
        ax_reset = plt.axes([0.15, 0.01, 0.1, 0.03])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset)
        
    def on_slider_change(self, val):
        """Handle slider changes"""
        self.plane_tilt = np.radians(self.slider_tilt.val)
        self.plane_rotation = np.radians(self.slider_rotation.val)
        self.plane_height = self.slider_height.val
        self.update_plot()
        
    def reset(self, event):
        """Reset all sliders"""
        self.slider_tilt.reset()
        self.slider_rotation.reset()
        self.slider_height.reset()
        
    def get_plane_equation(self):
        """
        Get plane equation coefficients based on tilt and rotation
        Returns normal vector (a, b, c) and d for plane equation ax + by + cz = d
        """
        # Start with horizontal plane (normal pointing up)
        normal = np.array([0, 0, 1])
        
        # Rotate around x-axis (tilt)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.plane_tilt), -np.sin(self.plane_tilt)],
            [0, np.sin(self.plane_tilt), np.cos(self.plane_tilt)]
        ])
        
        # Rotate around z-axis (rotation)
        Rz = np.array([
            [np.cos(self.plane_rotation), -np.sin(self.plane_rotation), 0],
            [np.sin(self.plane_rotation), np.cos(self.plane_rotation), 0],
            [0, 0, 1]
        ])
        
        # Apply rotations
        normal = Rz @ Rx @ normal
        
        # Plane passes through point (0, 0, plane_height)
        d = np.dot(normal, np.array([0, 0, self.plane_height]))
        
        return normal, d
        
    def compute_intersection(self):
        """
        Compute intersection of plane with double cone analytically.
        Returns list of curves (each curve is an array of 3D points).

        Parameterizes the cone by angle phi and solves for z from plane equation.
        Cone: x^2 + y^2 = (z * tan(angle))^2
        Plane: ax + by + cz = d
        """
        normal, d = self.get_plane_equation()
        a, b, c = normal
        tan_angle = np.tan(self.cone_angle)

        # Fine angular sampling for smooth curves
        phi = np.linspace(0, 2*np.pi, 500)

        curves = []

        # Upper cone (z > 0): z = d / (tan_angle * (a*cos(phi) + b*sin(phi)) + c)
        denom_upper = tan_angle * (a * np.cos(phi) + b * np.sin(phi)) + c
        with np.errstate(divide='ignore', invalid='ignore'):
            z_upper = np.where(np.abs(denom_upper) > 1e-10, d / denom_upper, np.nan)

        # Filter valid points: z > 0 and within cone height
        valid_upper = (z_upper > 0) & (z_upper < self.cone_height)
        if np.any(valid_upper):
            curves.append(self._extract_continuous_segments(phi, z_upper, valid_upper, upper=True))

        # Lower cone (z < 0): z = d / (c - tan_angle * (a*cos(phi) + b*sin(phi)))
        denom_lower = c - tan_angle * (a * np.cos(phi) + b * np.sin(phi))
        with np.errstate(divide='ignore', invalid='ignore'):
            z_lower = np.where(np.abs(denom_lower) > 1e-10, d / denom_lower, np.nan)

        # Filter valid points: z < 0 and within cone height
        valid_lower = (z_lower < 0) & (z_lower > -self.cone_height)
        if np.any(valid_lower):
            curves.append(self._extract_continuous_segments(phi, z_lower, valid_lower, upper=False))

        # Flatten list of segments
        all_curves = []
        for curve_list in curves:
            all_curves.extend(curve_list)

        return all_curves if all_curves else None

    def _extract_continuous_segments(self, phi, z_vals, valid_mask, upper=True):
        """
        Extract continuous curve segments from parameterized intersection.
        Returns list of 3D point arrays, one per continuous segment.
        """
        tan_angle = np.tan(self.cone_angle)
        segments = []

        # Find continuous runs of valid points
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return segments

        # Split into continuous segments (gaps in indices)
        splits = np.where(np.diff(valid_indices) > 1)[0] + 1
        index_segments = np.split(valid_indices, splits)

        for idx_seg in index_segments:
            if len(idx_seg) < 2:
                continue

            phi_seg = phi[idx_seg]
            z_seg = z_vals[idx_seg]
            r_seg = np.abs(z_seg) * tan_angle

            x_seg = r_seg * np.cos(phi_seg)
            y_seg = r_seg * np.sin(phi_seg)

            points = np.column_stack([x_seg, y_seg, z_seg])
            segments.append(points)

        # Check if first and last segments should be joined (wrap-around)
        if len(segments) >= 2:
            if valid_mask[0] and valid_mask[-1]:
                # Join last and first segments
                segments[0] = np.vstack([segments[-1], segments[0]])
                segments.pop()

        return segments
        
    def project_to_2d(self, curves_3d):
        """
        Project 3D intersection curves to 2D plane coordinates.
        Input: list of 3D curve segments
        Output: list of 2D curve segments
        """
        if curves_3d is None or len(curves_3d) == 0:
            return None

        normal, d = self.get_plane_equation()

        # Create basis in the plane
        if abs(normal[0]) > 0.1:
            v1 = np.array([-normal[1]/normal[0], 1, 0])
        elif abs(normal[1]) > 0.1:
            v1 = np.array([1, -normal[0]/normal[1], 0])
        else:
            v1 = np.array([1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)

        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        # Center point
        if normal[2] != 0:
            center = normal * d / np.dot(normal, normal)
        else:
            center = np.array([0, 0, self.plane_height])

        # Project each curve segment
        curves_2d = []
        for curve in curves_3d:
            if len(curve) < 2:
                continue
            rel = curve - center
            x_2d = np.dot(rel, v1)
            y_2d = np.dot(rel, v2)
            curves_2d.append(np.column_stack([x_2d, y_2d]))

        return curves_2d if curves_2d else None
        
    def classify_conic(self):
        """Classify the type of conic section based on plane tilt vs cone angle"""
        tilt_deg = abs(self.slider_tilt.val)
        cone_angle_deg = np.degrees(self.cone_angle)

        if tilt_deg < 1:
            return "Circle"
        elif tilt_deg < cone_angle_deg:
            return "Ellipse"
        elif tilt_deg == cone_angle_deg:
            return "Parabola"
        else:
            return "Hyperbola"
        
    def draw_plane(self):
        """Draw the cutting plane"""
        normal, d = self.get_plane_equation()
        
        # Create plane mesh
        size = 4
        xx, yy = np.meshgrid(np.linspace(-size, size, 10), 
                             np.linspace(-size, size, 10))
        
        if abs(normal[2]) > 0.01:
            zz = (d - normal[0]*xx - normal[1]*yy) / normal[2]
        else:
            # Vertical plane
            if abs(normal[0]) > 0.01:
                xx = (d - normal[1]*yy) / normal[0] * np.ones_like(yy)
                zz = yy.copy()
                yy = np.linspace(-size, size, 10).reshape(-1, 1) * np.ones_like(xx)
            else:
                yy = (d - normal[0]*xx) / normal[1] * np.ones_like(xx)
                zz = xx.copy()
                xx = np.linspace(-size, size, 10).reshape(-1, 1) * np.ones_like(yy)
        
        # Clip to reasonable bounds
        mask = (np.abs(zz) < self.cone_height * 1.5) & \
               (np.abs(xx) < size) & (np.abs(yy) < size)
        zz = np.where(mask, zz, np.nan)
        
        self.ax_3d.plot_surface(xx, yy, zz, alpha=0.3, color='blue')

    def _draw_cone_angle(self):
        """Draw the cone opening angle annotation on the 3D plot"""
        # Draw in the x-z plane (y=0)
        length = 1.5  # length of reference lines

        # Line along z-axis from apex
        self.ax_3d.plot([0, 0], [0, 0], [0, length], 'k--', linewidth=1.5)

        # Line along cone surface
        z_cone = length
        x_cone = z_cone * np.tan(self.cone_angle)
        self.ax_3d.plot([0, x_cone], [0, 0], [0, z_cone], 'k--', linewidth=1.5)

        # Draw arc to show angle
        arc_radius = 0.6
        arc_angles = np.linspace(0, self.cone_angle, 20)
        arc_x = arc_radius * np.sin(arc_angles)
        arc_z = arc_radius * np.cos(arc_angles)
        arc_y = np.zeros_like(arc_x)
        self.ax_3d.plot(arc_x, arc_y, arc_z, 'k-', linewidth=1.5)

        # Label the angle
        label_angle = self.cone_angle / 2
        label_r = arc_radius + 0.3
        label_x = label_r * np.sin(label_angle)
        label_z = label_r * np.cos(label_angle)
        angle_deg = int(np.degrees(self.cone_angle))
        self.ax_3d.text(label_x, 0.1, label_z, f'{angle_deg}°', fontsize=10, fontweight='bold')

    def update_plot(self):
        """Update the visualization"""
        # Clear axes
        self.ax_3d.clear()
        self.ax_2d.clear()
        
        # Draw cone
        self.ax_3d.plot_surface(self.cone_X, self.cone_Y, self.cone_Z, 
                                alpha=0.3, color='gray', edgecolor='none')
        
        # Draw plane
        self.draw_plane()
        
        # Compute and draw intersection
        curves_3d = self.compute_intersection()
        if curves_3d is not None and len(curves_3d) > 0:
            # Draw 3D curves
            for curve in curves_3d:
                self.ax_3d.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                               c='red', linewidth=2)

            # Project to 2D and draw
            curves_2d = self.project_to_2d(curves_3d)
            if curves_2d is not None and len(curves_2d) > 0:
                for curve in curves_2d:
                    self.ax_2d.plot(curve[:, 0], curve[:, 1],
                                   c='red', linewidth=2)
        
        # Draw cone angle annotation
        self._draw_cone_angle()

        # Configure 3D plot
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D View: Cone and Cutting Plane')
        
        limit = 3
        self.ax_3d.set_xlim(-limit, limit)
        self.ax_3d.set_ylim(-limit, limit)
        self.ax_3d.set_zlim(-self.cone_height, self.cone_height)
        
        # Configure 2D plot
        conic_type = self.classify_conic()
        self.ax_2d.set_xlabel('u')
        self.ax_2d.set_ylabel('v')
        self.ax_2d.set_title(f'Cross-Section View: {conic_type}')
        self.ax_2d.set_aspect('equal')
        self.ax_2d.grid(True, alpha=0.3)
        self.ax_2d.set_xlim(-4, 4)
        self.ax_2d.set_ylim(-4, 4)
        
        # Add legend for conic type
        legend_elements = [Patch(facecolor='red', alpha=0.6, label=conic_type)]
        self.ax_2d.legend(handles=legend_elements, loc='upper right')
        
        plt.draw()

def main():
    visualizer = ConicSectionVisualizer()
    plt.show()

if __name__ == '__main__':
    main()
