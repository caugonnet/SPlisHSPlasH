#!/usr/bin/env python3
"""Numerical parity check between two SPlisHSPlasH VTK particle export runs.

Compares per-particle position/velocity/density (matched by exported particle
id) between two directories of legacy-binary VTK frames written by
ParticleExporter_VTK, e.g. a CPU DFSPH run vs. a DFSPH_CUDA run.

Usage:
    compare_vtk.py <ref_vtk_dir> <test_vtk_dir> [frame ...]

Recommended run configuration for exact frame alignment (see HANDOFF.md):
    cflMethod=0, fixed timeStepSize, enableZSort=false, enableVTKExport=true,
    particleAttributes="velocity;density".
"""
import os
import re
import sys

import numpy as np


def read_vtk(path):
    data = open(path, 'rb').read()
    fields = {}
    m = re.search(rb'POINTS (\d+) float\n', data)
    n = int(m.group(1))
    fields['position'] = np.frombuffer(data, dtype='>f4', count=3 * n,
                                       offset=m.end()).reshape(n, 3)
    m = re.search(rb'SCALARS id unsigned_int 1\nLOOKUP_TABLE \S+\n', data)
    fields['id'] = np.frombuffer(data, dtype='>u4', count=n, offset=m.end())
    m = re.search(rb'velocity 3 (\d+) float\n', data)
    if m:
        fields['velocity'] = np.frombuffer(data, dtype='>f4', count=3 * n,
                                           offset=m.end()).reshape(n, 3)
    m = re.search(rb'density 1 (\d+) float\n', data)
    if m:
        fields['density'] = np.frombuffer(data, dtype='>f4', count=n,
                                          offset=m.end())
    order = np.argsort(fields['id'])
    return {k: v[order] for k, v in fields.items()}


def frame_number(name):
    return int(re.search(r'_(\d+)\.vtk$', name).group(1))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref_dir, test_dir = sys.argv[1], sys.argv[2]
    common = sorted(set(os.listdir(ref_dir)) & set(os.listdir(test_dir)),
                    key=frame_number)
    if len(sys.argv) > 3:
        wanted = {int(a) for a in sys.argv[3:]}
        common = [f for f in common if frame_number(f) in wanted]
    print(f"{'frame':>6} {'max|dx|':>12} {'rms|dx|':>12} {'max|dv|':>12} {'max|drho|':>12}")
    worst = 0.0
    for name in common:
        r = read_vtk(os.path.join(ref_dir, name))
        t = read_vtk(os.path.join(test_dir, name))
        dx = np.linalg.norm(r['position'].astype('f8') - t['position'].astype('f8'), axis=1)
        dv = (np.linalg.norm(r['velocity'].astype('f8') - t['velocity'].astype('f8'), axis=1)
              if 'velocity' in r and 'velocity' in t else np.zeros(1))
        dr = (np.abs(r['density'].astype('f8') - t['density'].astype('f8'))
              if 'density' in r and 'density' in t else np.zeros(1))
        worst = max(worst, dx.max())
        print(f"{frame_number(name):>6} {dx.max():12.4e} "
              f"{np.sqrt((dx ** 2).mean()):12.4e} {dv.max():12.4e} {dr.max():12.4e}")
    print(f"\nworst max|dx| over {len(common)} frames: {worst:.4e}")


if __name__ == '__main__':
    main()
