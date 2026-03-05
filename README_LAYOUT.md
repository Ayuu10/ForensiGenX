Spatial layout engine

Usage:

Run the layout generator on a scene graph JSON:

```bash
python spatial_layout.py test_scene_graph.json --out layout.json --img layout.png
```

Outputs:

- `layout.json`: object positions (x,y,w,h)
- `layout.png`: simple visualisation (boxes with labels)

Notes:

- Supports relations: near, left_of, right_of, above, below, on, under
- Max objects per scene assumed small (<=6) for simple layouts
