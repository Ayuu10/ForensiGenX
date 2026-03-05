#!/usr/bin/env python3
import json
import math
import argparse
from PIL import Image, ImageDraw, ImageFont
import random


def load_scene(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Try several common keys
    objs = None
    rels = None
    for k in ("objects", "nodes", "entities", "items"):
        if k in data:
            objs = data[k]
            break
    for k in ("relationships", "relations", "edges", "predicates"):
        if k in data:
            rels = data[k]
            break
    # Fallback: if top-level is list assume objects
    if objs is None and isinstance(data, dict) and 'objects' in data:
        objs = data['objects']
    if objs is None:
        # Try simple format: keys are object ids
        if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
            objs = []
            for k, v in data.items():
                if isinstance(v, dict) and ('name' in v or 'label' in v):
                    objs.append({'id': k, 'name': v.get('name', v.get('label', k))})
    if objs is None:
        raise ValueError('Could not find objects list in scene file')

    if rels is None:
        # maybe single list called "relationship" or embedded
        rels = data.get('relationships', []) or data.get('relations', []) or []

    # Normalize objects to have id and name
    norm_objs = []
    for i, o in enumerate(objs):
        if isinstance(o, str):
            norm_objs.append({'id': str(i), 'name': o})
        elif isinstance(o, dict):
            oid = o.get('id') or o.get('name') or str(i)
            name = o.get('name') or o.get('label') or str(oid)
            norm_objs.append({'id': str(oid), 'name': name})
        else:
            norm_objs.append({'id': str(i), 'name': str(o)})

    norm_rels = []
    for r in rels:
        if isinstance(r, dict):
            s = r.get('subject') or r.get('from') or r.get('source') or r.get('s')
            p = r.get('predicate') or r.get('relation') or r.get('label') or r.get('p')
            o = r.get('object') or r.get('to') or r.get('target') or r.get('o')
            if s is None or p is None or o is None:
                # try edge format [s,p,o]
                vals = [v for v in r.values()]
                if len(vals) >= 3:
                    s, p, o = vals[0], vals[1], vals[2]
            if s is not None and p is not None and o is not None:
                norm_rels.append({'s': str(s), 'p': str(p), 'o': str(o)})
        elif isinstance(r, list) and len(r) >= 3:
            norm_rels.append({'s': str(r[0]), 'p': str(r[1]), 'o': str(r[2])})

    return norm_objs, norm_rels


def find_obj_index(objs, identifier):
    # identifier may be id, name, or index string
    for i, o in enumerate(objs):
        if identifier == o.get('id') or identifier == o.get('name') or identifier == str(i):
            return i
    # try matching by substring
    for i, o in enumerate(objs):
        if identifier in o.get('name', '') or identifier in o.get('id', ''):
            return i
    return None


def solve_layout(objs, rels, canvas=(512, 512), box=(100, 100), margin=20):
    # Anchor-based layout: pick the most-connected object as anchor
    n = len(objs)
    w, h = canvas
    bw, bh = box

    # parameters
    OFFSET = 150
    NEAR_DISTANCE = 120
    MIN_DISTANCE = 120
    ITER = 120

    # uniform sizes with optional category scaling
    def size_for_name(name):
        lname = name.lower()
        if any(k in lname for k in ('car', 'truck', 'van', 'bus')):
            return (140, 80)
        if any(k in lname for k in ('store', 'building', 'shop')):
            return (180, 140)
        if any(k in lname for k in ('person', 'officer', 'man', 'woman', 'suspect')):
            return (80, 120)
        if any(k in lname for k in ('lamp', 'light', 'street')):
            return (40, 160)
        if any(k in lname for k in ('gun', 'knife', 'flashlight')):
            return (40, 30)
        return (bw, bh)

    positions = [[0, 0, bw, bh] for _ in range(n)]
    for i, o in enumerate(objs):
        sw, sh = size_for_name(o.get('name', ''))
        positions[i][2] = sw
        positions[i][3] = sh

    # build adjacency counts
    counts = {i: 0 for i in range(n)}
    adj = {i: [] for i in range(n)}
    for r in rels:
        si = find_obj_index(objs, r['s'])
        oi = find_obj_index(objs, r['o'])
        if si is None or oi is None:
            continue
        counts[si] += 1
        counts[oi] += 1
        adj[si].append((oi, r['p'].lower()))
        adj[oi].append((si, r['p'].lower()))

    # choose anchor: max degree, tie-break first occurrence
    anchor_idx = max(counts.keys(), key=lambda k: (counts[k], -k))
    # place anchor at center
    positions[anchor_idx][0] = int(w / 2 - positions[anchor_idx][2] / 2)
    positions[anchor_idx][1] = int(h / 2 - positions[anchor_idx][3] / 2)
    placed = {anchor_idx}

    # BFS from anchor to place connected objects relative to placed nodes
    from collections import deque
    q = deque([anchor_idx])
    while q:
        cur = q.popleft()
        cx, cy, cw, ch = positions[cur]
        for (nbr, pred) in adj[cur]:
            if nbr in placed:
                continue
            # default place near anchor cur
            nx = cx
            ny = cy
            if 'left' in pred and 'right' not in pred:
                nx = cx - OFFSET - positions[nbr][2]
                ny = cy
            elif 'right' in pred and 'left' not in pred:
                nx = cx + cw + OFFSET
                ny = cy
            elif 'above' in pred:
                nx = cx
                ny = cy - OFFSET - positions[nbr][3]
            elif 'below' in pred:
                nx = cx
                ny = cy + ch + OFFSET
            elif pred == 'on' or 'on_top' in pred:
                nx = cx
                ny = cy - positions[nbr][3]
            elif 'under' in pred:
                nx = cx
                ny = cy + ch
            elif 'near' in pred or 'close' in pred:
                # place at NEAR_DISTANCE with small random angle to avoid exact overlap
                angle = random.uniform(0, 2 * math.pi)
                nx = int(cx + math.cos(angle) * NEAR_DISTANCE)
                ny = int(cy + math.sin(angle) * NEAR_DISTANCE)
            else:
                # fallback: place to the right
                nx = cx + cw + OFFSET
                ny = cy

            # center-adjust
            nx = int(nx)
            ny = int(ny)
            positions[nbr][0] = nx
            positions[nbr][1] = ny
            placed.add(nbr)
            q.append(nbr)

    # Any unplaced nodes (disconnected) - spread around anchor
    for i in range(n):
        if i in placed:
            continue
        angle = random.uniform(0, 2 * math.pi)
        rdist = NEAR_DISTANCE + 60
        positions[i][0] = int(w / 2 + math.cos(angle) * rdist)
        positions[i][1] = int(h / 2 + math.sin(angle) * rdist)
        placed.add(i)

    # Resolve collisions / enforce min distance iteratively
    for _ in range(ITER):
        moved_any = False
        for i in range(n):
            for j in range(i + 1, n):
                ai = positions[i]
                aj = positions[j]
                # compute centers
                cix = ai[0] + ai[2] / 2.0
                ciy = ai[1] + ai[3] / 2.0
                cjx = aj[0] + aj[2] / 2.0
                cjy = aj[1] + aj[3] / 2.0
                dx = cjx - cix
                dy = cjy - ciy
                dist = math.hypot(dx, dy)
                target = max(MIN_DISTANCE, (ai[2] + aj[2]) / 2.0)
                if dist < target and dist > 0:
                    # push apart half the overlap each
                    overlap = (target - dist)
                    ux = dx / dist
                    uy = dy / dist
                    shift = overlap / 2.0 + 1.0
                    ai[0] = int(ai[0] - ux * shift)
                    ai[1] = int(ai[1] - uy * shift)
                    aj[0] = int(aj[0] + ux * shift)
                    aj[1] = int(aj[1] + uy * shift)
                    moved_any = True
                elif dist == 0:
                    # random jitter
                    ai[0] += random.randint(-10, 10)
                    ai[1] += random.randint(-10, 10)
                    moved_any = True

        # clamp to canvas
        for p in positions:
            p[0] = max(margin, min(w - p[2] - margin, p[0]))
            p[1] = max(margin, min(h - p[3] - margin, p[1]))

        if not moved_any:
            break

    # finalize layout dict
    layout = []
    for i, o in enumerate(objs):
        x, y, bwx, bhy = positions[i]
        layout.append({'id': o.get('id'), 'name': o.get('name'), 'x': int(x), 'y': int(y), 'w': int(bwx), 'h': int(bhy)})
    return layout

    # finalize layout dict
    layout = []
    for i, o in enumerate(objs):
        x, y, bw, bh = positions[i]
        layout.append({'id': o.get('id'), 'name': o.get('name'), 'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh)})
    return layout


def render_layout(layout, outimg, canvas=(512, 512), bgcolor=(255, 255, 255)):
    img = Image.new('RGB', canvas, bgcolor)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    colors = [(200, 80, 80), (80, 200, 80), (80, 80, 200), (200, 200, 80), (200, 80, 200), (80, 200, 200)]
    for i, obj in enumerate(layout):
        x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        c = colors[i % len(colors)]
        draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), fill=c)
        label = obj.get('name', str(obj.get('id')))
        try:
            if font is not None:
                tw, th = font.getsize(label)
            else:
                raise Exception()
        except Exception:
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = (len(label) * 6, 10)
        tx = x + 4
        ty = y + 4
        draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
    img.save(outimg)


def main():
    parser = argparse.ArgumentParser(description='Spatial layout engine: relations -> 2D layout')
    parser.add_argument('scene', help='scene_graph.json input')
    parser.add_argument('--out', default='layout.json', help='output layout json')
    parser.add_argument('--img', default='layout.png', help='output layout image')
    parser.add_argument('--size', type=int, nargs=2, metavar=('W', 'H'), default=(512, 512))
    parser.add_argument('--box', type=int, nargs=2, metavar=('BW', 'BH'), default=(100, 100))
    parser.add_argument('--verbose', action='store_true', help='print debug info')
    args = parser.parse_args()

    objs, rels = load_scene(args.scene)
    if args.verbose:
        print('Loaded objects:')
        for i, o in enumerate(objs):
            print(f"  [{i}] id={o.get('id')} name={o.get('name')}")
        print('Loaded relations:')
        for r in rels:
            print(' ', r)
        # check that relation endpoints resolve
        for r in rels:
            si = find_obj_index(objs, r['s'])
            oi = find_obj_index(objs, r['o'])
            if si is None or oi is None:
                print('Unresolved relation endpoints:', r, '->', 's_idx', si, 'o_idx', oi)
    layout = solve_layout(objs, rels, canvas=tuple(args.size), box=tuple(args.box))
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'objects': layout}, f, indent=2)
    render_layout(layout, args.img, canvas=tuple(args.size))
    print('Wrote', args.out, 'and', args.img)


if __name__ == '__main__':
    main()
