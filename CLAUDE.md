# Conic Sections Visualization

Interactive matplotlib visualization demonstrating conic sections as cross-sections of a double cone.

## Quick Reference

- **Run**: `python conic_sections.py`
- **Dependencies**: numpy, matplotlib
- **Python**: 3.8+

## Architecture

Single-file application using matplotlib's interactive widgets. The `ConicSectionVisualizer` class handles all visualization logic.

### Key Methods

- `compute_intersection()` - Computes cone-plane intersection curve analytically
- `project_to_2d()` - Projects 3D curve onto the cutting plane's coordinate system
- `classify_conic()` - Determines conic type based on plane tilt vs cone angle

### Conic Classification

The conic type depends on the relationship between plane tilt angle and cone half-angle (45 degrees):
- **Circle**: tilt = 0
- **Ellipse**: 0 < tilt < cone angle
- **Parabola**: tilt = cone angle
- **Hyperbola**: tilt > cone angle

## Code Style

- NumPy for all numerical computation
- Angles in radians internally, degrees in UI
- Plane equation: ax + by + cz = d (normal vector form)
