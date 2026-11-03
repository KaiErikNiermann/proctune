# proctune

Live tuning for procedural models in Blender. You write the model in Python: typed parameters, and stages that generate meshes (a scaffold, structure on it, a surface over everything). proctune turns the parameters into sliders, rebuilds on every change, gives you handles to drag, and lets you move and sculpt the generated meshes by hand in a way that survives the next rebuild. A tuning harness fits the parameters to target measurements with Optuna or any optimizer that suggests values the same way.

## Why it exists

Blender has the pieces (properties with update callbacks, timers, depsgraph handlers) but nothing that binds them to an arbitrary Python model. The nearest existing tools, checked on 2026-11-03, solve other problems:

- Geometry Nodes exposes node-group inputs as sliders and live-evaluates them, but the model has to be a node graph. [Geometry Script](https://github.com/carson-katri/geometry-script) and [Node To Python](https://extensions.blender.org/add-ons/node-to-python/) (both GPL-3.0) generate node graphs from Python, which helps only if the model fits Geometry Nodes.
- Sculpt layer add-ons ([Sculpt Layers](https://superhivemarket.com/products/sculpt-layers), [Edit Layers](https://extensions.blender.org/add-ons/edit-layers/), GPL-3.0) keep sculpts across edits of a hand-made base mesh, not across a mesh that is regenerated from parameters. Blender Studio's own notes describe the [limits of that workflow](https://studio.blender.org/blog/layered-sculpting-for-einar/).

## The idea

Every generated mesh is an `Element` with a `Frame`: an origin, axes, and a scale measured in the element's own size (a bone's length along it, its radius across). Vertices are generated in those normalized coordinates, so "half way along" means the same at any length. Hand edits are stored in the same coordinates:

- a *tweak* (the element moved, turned or scaled as a whole) is a translation, rotation and scale in the element's frame, with the translation in element sizes so it grows with the element;
- a *sculpt* is a displacement per vertex, in normalized coordinates for a rigid element, or in a frame per vertex for a surface that bends (a skin's out, along and around directions).

On a rebuild the model regenerates every element from the parameters and the edits are re-applied. A bump sculpted half way along a muscle stays half way along when the bone gets longer; a skin sculpt stays on the same side of the limb when the joint bends. Stages see the elements before them with their edits applied, so the skin wraps a sculpted muscle.

A model must keep each element's topology fixed (same vertex count and order for all parameter values). A sculpt stored for another topology is reported as stale and not applied.

Some hand moves are better read as parameters than kept as moves. A model that implements `bind(p, element, matrix)` turns them into parameter changes: in the limb, scaling a bone along itself sets its length, scaling it across sets its radius, and turning the lower bone about the joint sets the joint's angle. Everything built on the skeleton then follows, and the move itself springs back to the regenerated bone. Other elements' moves stay tweaks.

## Curves, growth and time

A parameter can be a `Curve`: control points on $x \in [0, 1]$ through which a monotone piecewise cubic (scipy's `PchipInterpolator`) is drawn, so it passes through every point and never overshoots. Curves describe how strongly one structure responds to a driver. In the limb, `growth.stage` (think age) drives the overall size through a growth curve that rises fast and then plateaus; each fat deposit has its own curve from overall adiposity to thickness, one filling in early and mildly, the other late and steeply.

For growth relative to another structure there is `allometric(x, b)`, the standard $y = a\,x^{b}$: in the limb the muscle's girth grows as size$^{1.4}$, faster than the bones.

In Blender each curve is edited in Blender's own curve widget in the panel (it lives on a Float Curve node in a node group nothing else uses, as Blender's curve editor cannot be an add-on property). The widget only edits the points; the curve is evaluated by PCHIP everywhere, so Blender and a headless run agree. The widget draws its own spline through the same points, 0.007 to 0.033 apart from the evaluated one on the example's curves.

Any slider or composite can be keyframed, and the model follows the timeline: a frame-change handler rebuilds with the animated values. On save the animated values are sampled once per frame into the project's `tracks`, and `Session.at_frame(f)` gives the model at any frame without Blender. Sculpts and moves carry through all of this, since they are stored relative to each element's size.

## Two ways to make a skin

The limb builds its skin twice, on two layers, so they can be compared with the layer toggles:

- **skin** (the sweep, the example's own heuristic): a tube around a smoothed centreline through the bones; at each ring and angle its radius is the furthest tissue in that direction, smoothed, plus a thickness. It covers every tissue vertex and bridges the gaps between muscle and bone, but it assumes a tube-like body, and the inside of a sharp bend can fold.
- **wrap skin** (what Blender's Shrinkwrap, Nearest Surface Point with an Offset, followed by Corrective Smooth would make): a loose tube around everything, snapped onto the nearest tissue surface plus an offset and relaxed with Taubin smoothing (which does not shrink), in a few rounds. It hugs the tissue (833 ml against the sweep's 921 ml at the defaults), pinches in between muscle and bone the way shrinkwrapped meshes do, and webs across the inside of a sharp bend; relaxing lets about 0.2 % of the tissue's vertices poke through. Its settings are under `wrap` (offset, rounds, relax iterations, how loose the start tube is, and a switch to turn it off).

Both keep a fixed vertex layout, so both can be sculpted. libigl finds the wrap's nearest points (an AABB tree, 25 times faster than trimesh's own query here); the whole rebuild with both skins takes about 40 ms.

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

## Platforms

| | Linux | Windows |
| --- | --- | --- |
| The add-on in Blender 5.0 or newer | supported, tested | supported, tested in CI (unit and add-on tests under the `bpy` wheel) |
| End-to-end tests in the Blender GUI | tested (Xvfb) | not run: they need a virtual X display |
| Development (scripts, test suite) | yes | no |

Nothing in the add-on hardcodes a path. Blender's own interpreter, its user `scripts/modules` folder and the config folder are looked up at run time; projects store their model's folder relative to the project file (absolute only across Windows drives, where no relative path exists).

## Installing

The add-on needs numpy (every Blender has it) and scipy (for curve parameters); a model may need more (the limb example: trimesh). Missing packages are never a silent failure: the panel names them and offers **Install missing packages**, which runs pip with Blender's own Python into Blender's user `scripts/modules` (or the configured `packages` folder), leaving out what Blender already has, so its numpy is never shadowed. The same from a terminal:

```sh
blender -b --python-expr "from proctune import deps; deps.install_into_blender(['scipy', 'trimesh'])"
python -m proctune.deps scipy trimesh        # for a plain Python, into its user site-packages
```

Official Blender builds that ship without pip get it bootstrapped with `ensurepip` on first use.

## Configuration

Optional, per machine: `~/.config/proctune/config.toml` on Linux (`$XDG_CONFIG_HOME`), `%APPDATA%\proctune\config.toml` on Windows, or the file `PROCTUNE_CONFIG` names. Environment variables win over the file.

| Key | Environment | Default | What |
| --- | --- | --- | --- |
| `blender` | `PROCTUNE_BLENDER` | `blender` | The Blender executable (dev scripts, tests) |
| `packages` | `PROCTUNE_PACKAGES` | Blender's user `scripts/modules` | Where missing packages are installed |
| `dev_site` | `PROCTUNE_DEV_SITE` | `.blender-site` in the checkout | Packages for the dev scripts and tests |
| `xvfb_run` | `PROCTUNE_XVFB_RUN` | `xvfb-run` | Virtual display wrapper (end-to-end tests) |
| `magick` | `PROCTUNE_MAGICK` | `magick` | ImageMagick 7; `import` for ImageMagick 6 |

## Using it in Blender

```sh
python scripts/gui.py                        # opens examples/limb/limb.json (Linux, from a checkout)
python scripts/gui.py path/to/project.json
```

The first run installs what the example models import (scipy, trimesh, optuna) into the dev site folder, through Blender's own Python. The panel is in the 3D view's sidebar, tab "proctune":
- sliders for every parameter, grouped as the parameter dataclasses are, and for every composite;
- a curve widget for every curve parameter;
- layer visibility toggles;
- the hand edits per element, each with a button to clear it;
- Open, New (from a `module:attribute`), Save, Rebuild, Reset parameters.

Move, rotate or scale an element with the usual tools, or sculpt or edit it: the change is captured when the grab or stroke ends, and the stages after it rebuild (the skin follows a muscle stroke by stroke). Each movable element's object sits under a hidden empty at the place it was generated, so its Location, Rotation and Scale fields show the move relative to that place (Scale 1.2 is 1.2 times the generated size) and keep showing it through later rebuilds. A bone's move becomes its parameters instead, so its Scale returns to 1 and the slider changes. Every slider that differs from the model's default shows the ratio to it ("length x1.68") and a button to reset it. Moving or scaling the skin, which has no place of its own, is kept as a sculpt of the skin. Drag a handle empty to set the parameters it is bound to; a handle that means something when turned (the limb's elbow: R bends the joint by the angle turned) snaps upright again with the change in its slider, and one that does not is locked against rotation and scale. A project saves as `NAME.json` (parameters, composites, tweaks, tracks) and `NAME.npz` (sculpts).

## When something goes wrong

Every failure proctune can cause says what went wrong and what to do (`proctune.errors`: `ProjectError`, `ModelError`, `MissingDependenciesError`, each with a `remedy`). In Blender it appears in a red box at the top of the panel, and operators report it in the status bar; a failed rebuild keeps the last good mesh. Nothing Blender calls on its own (timers, handlers, slider callbacks, panel drawing) can fail silently: each is guarded, logs the traceback and shows the error in the panel. Covered by tests (`tests/test_errors.py`, `tests/test_stress.py`):

- saving into a read-only folder, over a read-only file, into a missing folder, or onto a full disk: nothing half-written is left, the old project stays as it was (every file is written to a temporary file and moved into place);
- opening a missing file, broken JSON (with line and column), another format, unknown parameters, values outside their ranges (each listed), invalid curves, a corrupt sculpt file;
- a model module that is not found (with the folders searched), imports a package Blender lacks (named, with the install command), fails on import, or fails while building (with the stage);
- a hand edit that cannot be kept (a sculpt that changed the vertex count, say): reported once, that object put back, the rest of the edits kept;
- an interpreter that cannot be started, pip failing, an unreadable config file or unknown config keys.

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

Linux only. Python 3.13, Poetry, and for the end-to-end tests Blender, Xvfb (`xvfb-run`) and ImageMagick.

```sh
poetry install --with dev,blender
poetry run pytest            # everything: core, add-on under the bpy wheel, errors, leaks, Optuna fit, end to end
poetry run pytest benchmarks # timings (pytest-benchmark), not part of the default run
scripts/checks.sh            # ruff, pyright (strict), pytest; run by the pre-push hook
```

The end-to-end suite (`tests/test_e2e.py`, about 70 s) runs the real Blender GUI on a virtual display: `tests/e2e/driver.py` grabs the muscle with G and the mouse, sculpts it with a real brush stroke, moves a slider while in Sculpt Mode, drags the wrist handle, turns the lower bone with R and stretches the upper one with S Y, edits a fat curve, steps a keyframed growth through the timeline with the arrow keys, and saves, all through simulated input (`--enable-event-simulate`) and the add-on's own timers and handlers. It fails on any traceback from proctune in Blender's output. Without its tools it is skipped, unless `PROCTUNE_E2E_REQUIRED=1` (set in CI), where it fails instead; `PROCTUNE_E2E_KEEP=DIR` keeps its screenshots.

CI (GitHub Actions) runs lint and type checks, the tests on Linux and Windows, and the end-to-end suite on Linux with the official Blender build.

### Performance and leaks

Medians on a Ryzen 9 9900X3D with Blender 5.2.3, from `poetry run pytest benchmarks`: a full rebuild in Blender (capture hand edits, evaluate, write every mesh, place handles, sync sliders) 43 ms with both skins, 9.8 ms with the wrap skin off; evaluating the limb in plain Python 40 ms (6.3 ms without the wrap skin), the same with every element sculpted; the sweep skin 2.3 ms, the wrap skin 33 ms; detecting hand edits 0.7 ms; capturing a sculpt 0.09 ms; saving and loading a project with sculpts 10 ms; a curve at 100,000 points 0.5 ms. Rebuild time per element stays flat from 20 to 400 elements (`test_rebuild_time_is_linear_in_the_element_count`). Over 300 rebuilds Python memory grows by less than 2 MB, resident memory by less than 40 MB, and Blender's data (objects, meshes, materials, node groups, collections, images) not at all; reopening a project 15 times and registering the add-on 10 times leave nothing behind.

## Known limits

- Handles with composites active: a handle writes base parameters through the model's `drag`, which sees the base parameters, so with a composite applied the handle and the drawn model can disagree until the composite is back at its default.
- Sliders and curve widgets are driven through their data in the end-to-end test, not by clicking them: Blender has no API for a button's position on screen. For the panel screenshot the driver shows the panel in the sidebar tab that is open, since simulated Ctrl+wheel switched tabs only once.

## Licence

GPL-3.0-or-later (`LICENSE`), as Blender requires of add-ons that use its Python API.
