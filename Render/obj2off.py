#!/usr/bin/env python3
"""
Minimal OBJ -> OFF converter (no external libs)
usage: python obj2off.py in.obj out.off
"""

import sys
import re
from pathlib import Path

def parse_int(s: str) -> int:
    """Parse possibly negative OBJ indices."""
    return int(s) - 1 if s else None

def read_obj(path):
    verts = []
    faces = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = re.split(r'\s+', line)
            tag = parts[0]
            if tag == 'v':
                # v x y z [w]  (ignore w)
                verts.append(tuple(map(float, parts[1:4])))
            elif tag == 'f':
                # f v/vt/vn v/vt/vn ...
                idxs = []
                for p in parts[1:]:
                    v = p.split('/')[0]
                    idx = parse_int(v)
                    if idx < 0:
                        idx += len(verts)   # negative index
                    idxs.append(idx)
                faces.append(idxs)
    return verts, faces

def write_off(verts, faces, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('OFF\n')
        f.write(f'{len(verts)} {len(faces)} 0\n')
        for v in verts:
            f.write(f'{v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'{len(face)} {" ".join(map(str, face))}\n')

def main():
    if len(sys.argv) < 3:
        print("Usage: python obj2off.py input.obj output.off")
        sys.exit(1)
    obj_path = Path(sys.argv[1])
    off_path = Path(sys.argv[2])
    if not obj_path.exists():
        print(f"File not found: {obj_path}")
        sys.exit(1)
    verts, faces = read_obj(obj_path)
    write_off(verts, faces, off_path)
    print(f"Converted {len(verts)} vertices, {len(faces)} faces -> {off_path}")

if __name__ == '__main__':
    main()