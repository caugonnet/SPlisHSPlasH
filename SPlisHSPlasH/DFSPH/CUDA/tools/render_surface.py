#!/usr/bin/env pvpython
"""Render a splashsurf-reconstructed water surface with the scene's rigid
bodies (boxes) using ParaView offscreen.

Pipeline:
  splashsurf reconstruct frame.vtk --particle-radius <r> --smoothing-length 2.0 \
      --cube-size 0.75 --surface-threshold 0.6 -o surface.obj
  pvpython --force-offscreen-rendering render_surface.py surface.obj scene.json out.png

Boxes are rebuilt from the scene JSON (skips body 0, assumed to be the
inverted world container). Cylinder bodies are not drawn.
"""
import sys, json
from paraview.simple import *

surf_obj, scene_json, out_png = sys.argv[1], sys.argv[2], sys.argv[3]

surf = WavefrontOBJReader(FileName=surf_obj)
rv = GetActiveViewOrCreate('RenderView')
rv.ViewSize = [1700, 950]
rv.Background = [0.08, 0.09, 0.12]
d = Show(surf, rv)
d.Representation = 'Surface'
d.DiffuseColor = [0.25, 0.55, 0.85]
d.Specular = 0.9
d.SpecularPower = 40.0
d.Interpolation = 'PBR'
d.Metallic = 0.1
d.Roughness = 0.25

scene = json.load(open(scene_json))
xmax = ymax = 1.0
for b in scene['RigidBodies'][1:]:
    if 'Cylinder' in b.get('geometryFile', ''):
        continue
    t, s = b['translation'], b['scale']
    box = Box(XLength=s[0], YLength=s[1], ZLength=s[2], Center=t)
    bd = Show(box, rv)
    bd.DiffuseColor = [0.55, 0.57, 0.6]
    bd.Interpolation = 'PBR'
    bd.Roughness = 0.8
    xmax = max(xmax, abs(t[0]) + s[0]/2); ymax = max(ymax, t[1] + s[1]/2)
world = scene['RigidBodies'][0]
g = Box(XLength=world['scale'][0], YLength=0.02, ZLength=world['scale'][2], Center=[world['translation'][0], -0.01, world['translation'][2]])
gd = Show(g, rv); gd.DiffuseColor = [0.18, 0.19, 0.21]

rv.CameraPosition = [xmax * 1.4, ymax * 2.5, world['scale'][2] * 1.15]
rv.CameraFocalPoint = [0, ymax * 0.15, 0]
rv.CameraViewUp = [0, 1, 0]
rv.CameraViewAngle = 32
SaveScreenshot(out_png, rv)
print("saved", out_png)
