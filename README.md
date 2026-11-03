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

`examples/limb/limb.py` is a complete model: two bones and a joint, a muscle with a bell-shaped belly across the joint, a skin swept over both, the composites `flex` (bend and bunch the muscle) and `size`, and handles at the elbow and wrist.

## Using it in Blender

```sh
scripts/gui.sh                               # opens examples/limb/limb.json
scripts/gui.sh path/to/project.json
```

The first run installs what the example models import (scipy, trimesh, optuna) into `.blender-site` for Blender's own Python. The panel is in the 3D view's sidebar, tab "proctune":

- sliders for every parameter, grouped as the parameter dataclasses are, and for every composite;
- layer visibility toggles;
- the hand edits per element, each with a button to clear it;
- Open, New (from a `module:attribute`), Save, Rebuild, Reset parameters.

Move, rotate or scale an element with the usual tools, or sculpt or edit it: the change is captured when the grab or stroke ends, and the stages after it rebuild (the skin follows a muscle stroke by stroke). Drag a handle empty to set the parameters it is bound to. A project saves as `NAME.json` (parameters, composites, tweaks) and `NAME.npz` (sculpts).

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
poetry run pytest            # core tests, the add-on under the bpy module, and an Optuna fit
scripts/checks.sh            # ruff, pyright (strict), pytest; run by the pre-push hook
```

The core (`proctune.core`) has no Blender dependency. The add-on (`proctune.blender`) is tested headless with the `bpy` wheel.

## Not verified yet

- Sculpt mode: the add-on writes the sculpted object's vertices while it is in Sculpt Mode when a slider changes. Whether Blender redraws them without leaving the mode has not been checked by eye.
- Handles with composites active: a handle writes base parameters through the model's `drag`, which sees the base parameters, so with a composite applied the handle and the drawn model can disagree until the composite is back at its default.

## Licence

Not chosen yet. Blender requires add-ons that use its Python API to be GPL-3.0-or-later, which is what `blender_manifest.toml` declares; `proctune.core` does not use that API and could be licensed permissively.
