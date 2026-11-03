# proctune

Live tuning for procedural models in Blender. You write the model in Python: typed parameters, and stages that generate meshes (a scaffold, structure on it, a surface over everything). proctune turns the parameters into sliders, rebuilds on every change, gives you handles to drag, and lets you move and sculpt the generated meshes by hand in a way that survives the next rebuild. A tuning harness fits the parameters to target measurements with Optuna or any optimizer that suggests values the same way.

## Why it exists

Blender has the pieces (properties with update callbacks, timers, depsgraph handlers) but nothing that binds them to an arbitrary Python model. The nearest existing tools, checked on 2026-11-03, solve other problems:

- Geometry Nodes exposes node-group inputs as sliders and live-evaluates them, but the model has to be a node graph. [Geometry Script](https://github.com/carson-katri/geometry-script) and [Node To Python](https://extensions.blender.org/add-ons/node-to-python/) (both GPL-3.0) generate node graphs from Python, which helps only if the model fits Geometry Nodes.
- Sculpt layer add-ons ([Sculpt Layers](https://superhivemarket.com/products/sculpt-layers), [Edit Layers](https://extensions.blender.org/add-ons/edit-layers/), GPL-3.0) keep sculpts across edits of a hand-made base mesh, not across a mesh that is regenerated from parameters. Blender Studio's own notes describe the [limits of that workflow](https://studio.blender.org/blog/layered-sculpting-for-einar/).

## The idea

Every generated mesh is an `Element` with a `Frame`: an origin, axes, and a scale measured in the element's own size (a bone's length along it, its radius across). Vertices are generated in those normalized coordinates, so "half way along" means the same at any length. Hand edits are stored in the same coordinates:

- a *tweak* (the element moved, turned or scaled as a whole) is an affine map in normalized coordinates;
- a *sculpt* is a displacement per vertex, in normalized coordinates for a rigid element, or in a frame per vertex for a surface that bends (a skin's out, along and around directions).

On a rebuild the model regenerates every element from the parameters and the edits are re-applied. A bump sculpted half way along a muscle stays half way along when the bone gets longer; a skin sculpt stays on the same side of the limb when the joint bends. Stages see the elements before them with their edits applied, so the skin wraps a sculpted muscle.

A model must keep each element's topology fixed (same vertex count and order for all parameter values). A sculpt stored for another topology is reported as stale and not applied.

Some hand moves are better read as parameters than kept as moves. A model that implements `bind(p, element, matrix)` turns them into parameter changes: in the limb, scaling a bone along itself sets its length, scaling it across sets its radius, and turning the lower bone about the joint sets the joint's angle. Everything built on the skeleton then follows, and the move itself springs back to the regenerated bone. Other elements' moves stay tweaks.

## Curves, growth and time

A parameter can be a `Curve`: control points on $x \in [0, 1]$ through which a monotone piecewise cubic (scipy's `PchipInterpolator`) is drawn, so it passes through every point and never overshoots. Curves describe how strongly one structure responds to a driver. In the limb, `growth.stage` (think age) drives the overall size through a growth curve that rises fast and then plateaus; each fat deposit has its own curve from overall adiposity to thickness, one filling in early and mildly, the other late and steeply.

For growth relative to another structure there is `allometric(x, b)`, the standard $y = a\,x^{b}$: in the limb the muscle's girth grows as size$^{1.4}$, faster than the bones.

In Blender each curve is edited in Blender's own curve widget in the panel (it lives on a Float Curve node in a node group nothing else uses, as Blender's curve editor cannot be an add-on property). The widget only edits the points; the curve is evaluated by PCHIP everywhere, so Blender and a headless run agree. The widget draws its own spline through the same points, 0.007 to 0.033 apart from the evaluated one on the example's curves.

Any slider or composite can be keyframed, and the model follows the timeline: a frame-change handler rebuilds with the animated values. On save the animated values are sampled once per frame into the project's `tracks`, and `Session.at_frame(f)` gives the model at any frame without Blender. Sculpts and moves carry through all of this, since they are stored relative to each element's size.

## Writing a model

```python
@dataclass(frozen=True)
class Bone:
    length: Annotated[float, Range(0.10, 0.60, 0.005, "m", "Joint to joint")] = 0.30


@dataclass(frozen=True)
class MyModel:
    name: str = "mine"

    def defaults(self) -> Params: ...
    @property
    def stages(self) -> tuple[Stage[Params], ...]:  # each: (params, earlier resolved elements) -> elements
        return (Stage("scaffold", bones), Stage("structure", muscles), Stage("surface", skin))

    @property
    def composites(self) -> tuple[Composite[Params], ...]:  # one slider moving many parameters
        ...
    def handles(self, p: Params) -> tuple[Handle[Params], ...]:  # draggable points bound to parameters
        ...
```

`examples/limb/limb.py` is a complete model: two bones and a joint, a muscle with a bell-shaped belly across the joint, a fat deposit along each bone, a skin swept over everything, growth and fat curves, the composites `flex` (bend and bunch the muscle) and `size`, handles at the elbow and wrist, and bone moves bound to parameters.

## Using it in Blender

```sh
scripts/gui.sh                               # opens examples/limb/limb.json
scripts/gui.sh path/to/project.json
```

The first run installs what the example models import (scipy, trimesh, optuna) into `.blender-site` for Blender's own Python. The panel is in the 3D view's sidebar, tab "proctune":

- sliders for every parameter, grouped as the parameter dataclasses are, and for every composite;
- a curve widget for every curve parameter;
- layer visibility toggles;
- the hand edits per element, each with a button to clear it;
- Open, New (from a `module:attribute`), Save, Rebuild, Reset parameters.

Move, rotate or scale an element with the usual tools, or sculpt or edit it: the change is captured when the grab or stroke ends, and the stages after it rebuild (the skin follows a muscle stroke by stroke). Drag a handle empty to set the parameters it is bound to. A project saves as `NAME.json` (parameters, composites, tweaks, tracks) and `NAME.npz` (sculpts).

## Tuning against targets

```python
from proctune.core.tune import Target, apply, loss, objective, space

targets = [Target("total length (m)", lambda r, p: p.upper.length + p.lower.length, 0.62), ...]
study = optuna.create_study(direction="minimize")
study.optimize(objective(session, space(session, ("upper.*", "muscle.belly")), loss(targets)), n_trials=150)
apply(session, study.best_params)
```

The search space is the parameters' own slider ranges, selected by dotted path patterns (composites by `composite:NAME`). A `Target` measures one number on an evaluated model; `loss` sums their weighted, scaled squared errors, and any other score function works too. Hand edits stay applied in every candidate. Optuna is optional (`pip install proctune[tune]`): the harness only needs a trial with `suggest_float`, `suggest_int` and `suggest_categorical`. `examples/limb/fit.py` fits length, girth and volume targets in 150 trials (about 1.6 s) to within 2 %.

## Development

```sh
poetry install --with dev,blender
poetry run pytest            # core, the add-on under the bpy module, an Optuna fit, and end to end in the GUI
scripts/checks.sh            # ruff, pyright (strict), pytest; run by the pre-push hook
```

The core (`proctune.core`) has no Blender dependency. The add-on (`proctune.blender`) is tested headless with the `bpy` wheel, and end to end (`tests/test_e2e.py`, about 70 s) in the real Blender GUI on a virtual display: `tests/e2e/driver.py` grabs the muscle with G and the mouse, sculpts it with a real brush stroke, moves a slider while in Sculpt Mode, drags the wrist handle, turns the lower bone with R and stretches the upper one with S Y, edits a fat curve, steps a keyframed growth through the timeline with the arrow keys, and saves, all through simulated input (`--enable-event-simulate`) and the add-on's own timers and handlers. It also fails on any traceback from proctune in Blender's output, since Blender only prints errors from handlers, timers and panel drawing. It needs `blender`, `xvfb-run` (Arch: `xorg-server-xvfb`) and `magick` (`imagemagick`, for screenshots) and is skipped without them; `PROCTUNE_E2E_KEEP=DIR` keeps its screenshots.

## Known limits

- Handles with composites active: a handle writes base parameters through the model's `drag`, which sees the base parameters, so with a composite applied the handle and the drawn model can disagree until the composite is back at its default.
- Sliders and curve widgets are driven through their data in the end-to-end test, not by clicking them: Blender has no API for a button's position on screen. For the panel screenshot the driver shows the panel in the sidebar tab that is open, since simulated Ctrl+wheel switched tabs only once.

## Licence

Not chosen yet. Blender requires add-ons that use its Python API to be GPL-3.0-or-later, which is what `blender_manifest.toml` declares; `proctune.core` does not use that API and could be licensed permissively.
