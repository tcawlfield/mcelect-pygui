"""Panel app for visualizing multi-winner (committee) election results from mcelect.

Run with ``mcelect-hvplot`` or ``panel serve src/mcelect_hvplot/app.py``.
"""

import asyncio
import itertools
from collections.abc import Callable
from dataclasses import dataclass

import awkward as ak
import holoviews as hv
import hvplot.pandas  # noqa: F401
import mcelect
import numpy as np
import pandas as pd
import panel as pn
from bokeh.palettes import Category10_10
from mcelect.config import (
    RP,
    RRV,
    STAR,
    Borda,
    Config,
    Electorate,
    Faction,
    Harmonic,
    InstantRunoff,
    PluralityTopN,
    QGaussian2,
    RangeVoting,
    Sequential,
    TopStandings,
)

pn.extension("tabulator")

DEFAULT_FACTIONS = pd.DataFrame(
    {
        "name": ["A", "B", "C", "D"],
        "popularity": [1.0, 0.95, 0.5, 0.3],
        "center_x": [-1.0, 1.0, -1.0, 1.0],
        "center_y": [1.0, 1.0, -1.0, -1.0],
        "voter_spread": [1.0, 1.0, 1.0, 1.0],
        "candidate_spread": [1.0, 1.0, 1.0, 1.0],
    }
)

# --- Method presets -----------------------------------------------------------

type CommitteeMethod = RRV | PluralityTopN | TopStandings | Sequential | Harmonic

# A preset builds its method given a column name, and says which column the results
# land in: Sequential and TopStandings prefix the inner method's column name.
type Preset = Callable[[str], tuple[CommitteeMethod, str]]


def _plain(make: Callable[[str], CommitteeMethod]) -> Preset:
    return lambda col: (make(col), col)


def _sequential(make: Callable[[str], object]) -> Preset:
    return lambda col: (Sequential.model_validate(make(col)), f"seq_{col}")


def _top(make: Callable[[str], object]) -> Preset:
    return lambda col: (TopStandings.model_validate(make(col)), f"top_{col}")


PRESETS: dict[str, Preset] = {
    "Plurality top-N": _plain(lambda c: PluralityTopN(colname=c)),
    "RRV, k=0.5": _plain(lambda c: RRV(strat="Honest", ranks=10, k=0.5, colname=c)),
    "RRV, k=1": _plain(lambda c: RRV(strat="Honest", ranks=10, k=1.0, colname=c)),
    "PAV": _plain(lambda c: Harmonic(colname=c)),
    "Harmonic, 0-9 scores": _plain(lambda c: Harmonic(ranks=10, colname=c)),
    "Sequential IRV": _sequential(lambda c: InstantRunoff(colname=c)),
    "Sequential STAR": _sequential(lambda c: STAR(strat="Honest", colname=c)),
    "Sequential ranked pairs": _sequential(lambda c: RP(strat="Honest", colname=c)),
    "Top-N Borda": _top(lambda c: Borda(colname=c)),
    "Top-N range, 0-9": _top(lambda c: RangeVoting(strat="Honest", nranks=10, colname=c)),
    "Top-N ranked pairs": _top(lambda c: RP(strat="Honest", colname=c)),
}
NO_METHOD = "(none)"
DEFAULT_SLOTS = ["Plurality top-N", "RRV, k=0.5", "PAV", "Sequential STAR"]

# --- Simulation ---------------------------------------------------------------


def build_config(
    factions: pd.DataFrame,
    methods: list[CommitteeMethod],
    *,
    voters: int,
    candidates: int,
    committee_size: int,
    sigma: float,
) -> Config:
    """A multi-winner config from the faction table and settings.

    Factions with popularity <= 0 are left out.
    """
    # mcelect doesn't check this itself; RRV panics instead.
    if committee_size > candidates:
        raise ValueError(f"committee size ({committee_size}) exceeds candidates ({candidates})")
    return Config(
        voters=voters,
        candidates=candidates,
        mode="multi_winner",
        committee_size=committee_size,
        considerations=[
            Electorate(
                dimensions=2,
                distance_function=QGaussian2(sigma),
                candidate_distribution="Uniform",
                factions=[
                    Faction(
                        popularity=row["popularity"],
                        voter_center=[row["center_x"], row["center_y"]],
                        voter_spread=row["voter_spread"],
                        candidate_spread=row["candidate_spread"],
                    )
                    for row in factions.to_dict("records")
                ],
            )
        ],
        committee_methods=methods,
    )


def required[T](value: T | None, name: str) -> T:
    """`value`, or a ValueError naming the input if it was left empty."""
    if value is None:
        raise ValueError(f"{name} is empty")
    return value


# --- Results to frames ----------------------------------------------------------


def candidates_frame(results: ak.Array, names: np.ndarray) -> pd.DataFrame:
    """One row per candidate across all trials: faction name and position."""
    factions = ak.to_numpy(ak.flatten(results.electorate.faction))
    positions = ak.to_numpy(ak.flatten(results.electorate.positions))
    return pd.DataFrame({"faction": names[factions], "x": positions[:, 0], "y": positions[:, 1]})


def winners(results: ak.Array, column: str, names: np.ndarray) -> pd.DataFrame:
    """One row per seat won by a method across all trials: the winner's faction
    name and position."""
    seats = ak.from_regular(results.methods[column].winners, axis=1)
    factions = ak.to_numpy(ak.flatten(results.electorate.faction[seats]))
    positions = ak.to_numpy(ak.flatten(results.electorate.positions[seats]))
    return pd.DataFrame({"faction": names[factions], "x": positions[:, 0], "y": positions[:, 1]})


def population_share(results: ak.Array) -> np.ndarray:
    """Each faction's share of all voters, pooled across trials."""
    voters = ak.to_numpy(ak.sum(results.electorate.faction_voters, axis=0))
    return voters / voters.sum()


@dataclass
class Grid:
    """Square bins covering the candidates, shared by every method's plot."""

    xedges: np.ndarray
    yedges: np.ndarray

    @classmethod
    def covering(cls, df: pd.DataFrame, bins: int = 40) -> Grid:
        lo = np.array([df.x.min(), df.y.min()])
        hi = np.array([df.x.max(), df.y.max()])
        width = (hi - lo).max() / bins
        return cls(
            np.arange(lo[0], hi[0] + width, width),
            np.arange(lo[1], hi[1] + width, width),
        )

    def counts(self, df: pd.DataFrame) -> np.ndarray:
        """Points per bin, shaped (y, x)."""
        h, _, _ = np.histogram2d(df.x, df.y, bins=[self.xedges, self.yedges])
        return h.T

    def image(self, z: np.ndarray, label: str) -> hv.Image:
        xs = (self.xedges[:-1] + self.xedges[1:]) / 2
        ys = (self.yedges[:-1] + self.yedges[1:]) / 2
        return hv.Image((xs, ys, z), kdims=["x", "y"], vdims=[label])


# Bins with fewer candidates than this are left blank in the election-rate view,
# where a handful of candidates would give a noisy rate.
MIN_CANDIDATES_PER_BIN = 10


@dataclass
class RunResult:
    """What the plots need from one simulation run."""

    factions: pd.DataFrame  # the active factions only
    cmap: dict[str, str]
    candidates: pd.DataFrame
    winners: dict[str, pd.DataFrame]  # by method label
    population: np.ndarray
    base_rate: float  # committee size / candidates


# --- Plots ------------------------------------------------------------------


def faction_markers(factions: pd.DataFrame, cmap: dict[str, str]) -> hv.Overlay:
    """A circle of radius voter_spread around each faction's voter center, labeled
    with the faction's name."""
    elements = []
    for f in factions.to_dict("records"):
        color = cmap[f["name"]]
        x, y = f["center_x"], f["center_y"]
        elements.append(
            hv.Ellipse(x, y, 2 * f["voter_spread"]).opts(color=color, line_width=2, line_dash="dashed")
        )
        elements.append(
            # The white backing keeps the label legible over points of the same color.
            hv.Text(x, y, f["name"]).opts(
                text_color=color,
                text_font_size="14pt",
                text_font_style="bold",
                background_fill_color="white",
                background_fill_alpha=0.8,
            )
        )
    return hv.Overlay(elements)


def winners_plot(label: str, run: RunResult, grid: Grid) -> hv.Overlay:
    """Winners as faction-colored points over a faint density of all candidates."""
    background = grid.counts(run.candidates)
    background[background == 0] = np.nan
    points = hv.Points(run.winners[label], ["x", "y"], ["faction"]).opts(
        color="faction", cmap=run.cmap, size=2, alpha=0.3, show_legend=False
    )
    return (
        grid.image(background, "candidates").opts(cmap="Greys", alpha=0.35)
        * points
        * faction_markers(run.factions, run.cmap)
    ).opts(title=label)


def rate_plot(label: str, run: RunResult, grid: Grid) -> hv.Overlay:
    """The fraction of candidates in each bin that the method elected."""
    cands = grid.counts(run.candidates)
    won = grid.counts(run.winners[label])
    rate = np.full_like(cands, np.nan)
    np.divide(won, cands, out=rate, where=cands >= MIN_CANDIDATES_PER_BIN)
    image = grid.image(rate, "election rate").opts(
        cmap="viridis", clim=(0, 1), colorbar=True, tools=["hover"]
    )
    return (image * faction_markers(run.factions, run.cmap)).opts(title=label)


def method_grid(run: RunResult, view: str) -> hv.Layout:
    grid = Grid.covering(run.candidates)
    plot = rate_plot if view == "Election rate" else winners_plot
    plots = [plot(label, run, grid) for label in run.winners]
    return (
        hv.Layout(plots)
        .cols(2)
        .opts(hv.opts.Overlay(data_aspect=1, frame_width=330), hv.opts.Layout(shared_axes=True))
    )


def share_plot(run: RunResult):
    """Grouped bars: each faction's share of the voters, next to its share
    of the seats won by each method."""
    names = run.factions.name
    shares = [pd.DataFrame({"faction": names, "source": "population", "share": run.population})]
    for label, df in run.winners.items():
        share = df.faction.value_counts(normalize=True).reindex(names, fill_value=0)
        shares.append(pd.DataFrame({"faction": share.index, "source": label, "share": share.values}))
    df = pd.concat(shares)
    # Ordered, so the bars follow the slots; plain strings would be sorted (in reverse).
    df["source"] = pd.Categorical(df.source, categories=["population", *run.winners], ordered=True)
    return (
        df.hvplot.bar(
            x="faction",
            y="share",
            by="source",
            title="Voter vs. seat share",
            height=250,
            responsive=True,
        )
        # A legend instead of a second axis level: method names overlap under each faction.
        .opts(multi_level=False, legend_position="right")
    )


VIEW_CAPTIONS = {
    "Winners": "Each method's winners, colored by faction, over the density of all "
    "candidates (gray). Dashed circles: each faction's voters, radius = voter σ.",
    "Election rate": "The fraction of candidates in each area that the method elected. "
    "Overall rate: {base_rate:.2f} (committee size / candidates). Areas with fewer than "
    f"{MIN_CANDIDATES_PER_BIN} candidates are blank.",
}


# --- App ----------------------------------------------------------------------


def create_app() -> pn.template.FastListTemplate:
    factions = pn.widgets.Tabulator(
        DEFAULT_FACTIONS.copy(),
        show_index=False,
        layout="fit_columns",
        sizing_mode="stretch_width",
        # Sort arrows would truncate the headers in the narrow sidebar.
        configuration={"columnDefaults": {"headerSort": False}},
        titles={
            "popularity": "pop.",
            "center_x": "x",
            "center_y": "y",
            "voter_spread": "voter σ",
            "candidate_spread": "cand. σ",
        },
    )
    add_faction = pn.widgets.Button(name="Add faction", button_type="light")
    trials = pn.widgets.IntInput(name="Trials", value=5000, start=1, step=1000)
    voters = pn.widgets.IntInput(name="Voters", value=600, start=1, step=100)
    candidates = pn.widgets.IntInput(name="Candidates", value=12, start=1)
    committee_size = pn.widgets.IntInput(name="Committee size", value=5, start=1)
    sigma = pn.widgets.FloatInput(name="QGaussian2 sigma", value=1.0, start=0.01, step=0.1)
    slots = [
        pn.widgets.Select(name=f"Method {i + 1}", options=[NO_METHOD, *PRESETS], value=default)
        for i, default in enumerate(DEFAULT_SLOTS)
    ]
    run_button = pn.widgets.Button(name="Run simulation", button_type="primary")
    view = pn.widgets.RadioButtonGroup(options=list(VIEW_CAPTIONS), value="Winners")
    output = pn.Column(sizing_mode="stretch_width")
    last_run: list[RunResult] = []  # the latest results, so changing views needn't rerun

    def on_add_faction(_):
        df = factions.value
        used = set(df.name)
        name = next(c for c in map(chr, range(ord("A"), ord("Z") + 1)) if c not in used)
        row = {
            "name": name,
            "popularity": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "voter_spread": 1.0,
            "candidate_spread": 0.5,
        }
        factions.value = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    def render(_=None):
        if not last_run:
            return
        run = last_run[0]
        output.objects = [
            pn.pane.Markdown(VIEW_CAPTIONS[view.value].format(base_rate=run.base_rate)),
            method_grid(run, view.value),
            share_plot(run),
        ]

    async def simulate(_=None):
        active = factions.value[factions.value.popularity > 0].reset_index(drop=True)
        # Each slot gets its own column name, so picking a method twice is harmless.
        chosen = [(s.value, PRESETS[s.value](f"slot{i}")) for i, s in enumerate(slots) if s.value != NO_METHOD]
        output.loading = True
        try:
            if not chosen:
                raise ValueError("Pick at least one method")
            n_candidates = required(candidates.value, candidates.name)
            size = required(committee_size.value, committee_size.name)
            config = build_config(
                active,
                [method for _, (method, _) in chosen],
                voters=required(voters.value, voters.name),
                candidates=n_candidates,
                committee_size=size,
                sigma=required(sigma.value, sigma.name),
            )
            n_trials = required(trials.value, trials.name)
            table = await asyncio.to_thread(mcelect.simulate, config, n_trials)
        except ValueError as e:  # includes pydantic's ValidationError
            output.objects = [pn.pane.Alert(str(e), alert_type="danger")]
            return
        finally:
            output.loading = False

        results = ak.from_arrow(table)
        names = active.name.to_numpy()
        labels = _unique_labels([label for label, _ in chosen])
        last_run[:] = [
            RunResult(
                factions=active,
                # Colored by row in the full table, so disabling a faction doesn't recolor the rest.
                cmap=dict(zip(factions.value.name, itertools.cycle(Category10_10))),
                candidates=candidates_frame(results, names),
                winners={
                    label: winners(results, column, names)
                    for label, (_, (_, column)) in zip(labels, chosen)
                },
                population=population_share(results),
                base_rate=size / n_candidates,
            )
        ]
        render()

    add_faction.on_click(on_add_faction)
    run_button.on_click(simulate)
    view.param.watch(render, "value")
    pn.state.onload(simulate)

    return pn.template.FastListTemplate(
        title="mcelect committee results",
        sidebar=[
            run_button,
            trials,
            "### Electorate",
            "Factions with popularity 0 are left out.",
            factions,
            add_faction,
            voters,
            candidates,
            sigma,
            "### Committee",
            committee_size,
            *slots,
        ],
        sidebar_width=420,
        main=[view, output],
    )


def _unique_labels(labels: list[str]) -> list[str]:
    """`labels`, with repeats numbered: ["PAV", "PAV"] -> ["PAV", "PAV (2)"]."""
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def main() -> None:
    pn.serve(create_app, title="mcelect committee results", show=True)


if __name__.startswith("bokeh"):
    create_app().servable()
