# Examples

Worked introductions to the SDK, ordered by how people actually meet
it. Each is a short, commented script with no arguments required; the
deeper *evidence drivers* (which generate the figures cited in
[docs/](../docs/README.md)) stay at the repo root as `run_*.py`.

| Script | What it shows | Needs |
|---|---|---|
| [hello_hologram.py](hello_hologram.py) | the algebra in 5 minutes: bind/unbind/bundle, a map, a record, the capacity law failing *soft* | nothing |
| [near_enough_rules.py](near_enough_rules.py) | a rule engine with no Boolean gates: messy-text dispatch, abstention as policy, a 4096-rule book routed at ~43x less compute via clustered bands | nothing |
| [splats_from_ply.py](splats_from_ply.py) | phone capture → holographic bundles → queries and X-ray renders, ~60 lines (figure: `out/example_splats.png`) | a raw 3DGS `.ply` (Scaniverse export) |
| [viewer/](viewer/index.html) | real-time splat rendering with occlusion (Spark/three.js): `python run_viewer.py <scene>` from the repo root, any `.ply`/`.spz`/`.splat` | a capture file; CDN access in the browser |

Run any of them from the repo root:

```bash
.venv/bin/python examples/hello_hologram.py
```

Where next: every technique used here has a docs page with the math,
the measured capacity budget, and the failure modes —
[docs/README.md](../docs/README.md) is the index, and `hdc-demos`
prints every structure's capacity table.
