"""Panel app for visualizing multi-winner (committee) election results from mcelect.

Run with ``mcelect-hvplot`` or ``panel serve src/mcelect_hvplot/app.py``.
"""

import asyncio
import itertools

import awkward as ak
import hvplot.pandas  # noqa: F401
import mcelect
import numpy as np
import pandas as pd
import panel as pn
from bokeh.palettes import Category10_10
from mcelect.config import RRV, Config, Electorate, Faction, PluralityTopN, QGaussian2

pn.extension("tabulator")

DEFAULT_FACTIONS = pd.DataFrame(
    dict(
        name=["A", "B", "C", "D"],
        popularity=[1.0, 0.95, 0.5, 0.3],
        center_x=[-1.0, 1.0, -1.0, 1.0],
        center_y=[1.0, 1.0, -1.0, -1.0],
        voter_spread=[1.0, 1.0, 1.0, 1.0],
        candidate_spread=[1.0, 1.0, 1.0, 1.0],
    )
)


def build_config(
    factions: pd.DataFrame,
    *,
    voters: int,
    candidates: int,
    committee_size: int,
    sigma: float,
    rrv_ranks: int,
    rrv_k: float,
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
                        popularity=row.popularity,
                        voter_center=[row.center_x, row.center_y],
                        voter_spread=row.voter_spread,
                        candidate_spread=row.candidate_spread,
                    )
                    for row in factions.itertuples()
                ],
            )
        ],
        committee_methods=[
            PluralityTopN(),
            RRV(strat="Honest", ranks=rrv_ranks, k=rrv_k),
        ],
    )


def winners(results: ak.Array, method: str, names: np.ndarray) -> pd.DataFrame:
    """One row per seat won by `method` across all trials: the winner's faction
    name and position."""
    seats = ak.from_regular(results.methods[method].winners, axis=1)
    factions = ak.to_numpy(ak.flatten(results.electorate.faction[seats]))
    positions = ak.to_numpy(ak.flatten(results.electorate.positions[seats]))
    return pd.DataFrame(
        dict(faction=names[factions], x=positions[:, 0], y=positions[:, 1])
    )


def population_share(results: ak.Array) -> np.ndarray:
    """Each faction's share of all voters, pooled across trials."""
    voters = ak.to_numpy(ak.sum(results.electorate.faction_voters, axis=0))
    return voters / voters.sum()


def share_plot(names: np.ndarray, population: np.ndarray, by_method: dict[str, pd.DataFrame]):
    """Grouped bars: each faction's share of the voters, next to its share
    of the seats won by each method."""
    shares = [pd.DataFrame(dict(faction=names, source="population", share=population))]
    for method, df in by_method.items():
        share = df.faction.value_counts(normalize=True).reindex(names, fill_value=0)
        shares.append(pd.DataFrame(dict(faction=share.index, source=method, share=share.values)))
    return pd.concat(shares).hvplot.bar(
        x="faction",
        y="share",
        by="source",
        title="Population vs. seat share",
        legend="top_right",
        height=350,
        responsive=True,
    )


def position_plot(method: str, df: pd.DataFrame, factions: pd.DataFrame, cmap: dict[str, str]):
    """Winning candidates' positions, colored by faction, with faction centers marked."""
    # Both elements need the same sizing, or hvplot's default width/height on one of
    # them conflicts with the aspect and holoviews warns.
    sizing = dict(data_aspect=1, frame_width=330)
    winners = df.hvplot.scatter(
        x="x",
        y="y",
        color="faction",
        cmap=cmap,
        size=3,
        alpha=0.3,
        title=f"{method} winner positions",
        **sizing,
    )
    centers = factions.rename(columns=dict(center_x="x", center_y="y")).hvplot.scatter(
        x="x", y="y", hover_cols=["name"], **sizing
    ).opts(marker="x", size=15, line_width=3, color="black")
    return winners * centers


def create_app() -> pn.template.FastListTemplate:
    factions = pn.widgets.Tabulator(
        DEFAULT_FACTIONS.copy(),
        show_index=False,
        layout="fit_columns",
        sizing_mode="stretch_width",
        # Sort arrows would truncate the headers in the narrow sidebar.
        configuration=dict(columnDefaults=dict(headerSort=False)),
        titles=dict(
            popularity="pop.",
            center_x="x",
            center_y="y",
            voter_spread="voter σ",
            candidate_spread="cand. σ",
        ),
    )
    add_faction = pn.widgets.Button(name="Add faction", button_type="light")
    trials = pn.widgets.IntInput(name="Trials", value=5000, start=1, step=1000)
    voters = pn.widgets.IntInput(name="Voters", value=600, start=1, step=100)
    candidates = pn.widgets.IntInput(name="Candidates", value=12, start=1)
    committee_size = pn.widgets.IntInput(name="Committee size", value=5, start=1)
    sigma = pn.widgets.FloatInput(name="QGaussian2 sigma", value=1.0, start=0.01, step=0.1)
    rrv_ranks = pn.widgets.IntInput(name="RRV ranks", value=10, start=2)
    rrv_k = pn.widgets.FloatInput(name="RRV k", value=0.5, start=0.0, step=0.1)
    run_button = pn.widgets.Button(name="Run simulation", button_type="primary")
    output = pn.Column(sizing_mode="stretch_width")

    def on_add_faction(_):
        df = factions.value
        used = set(df.name)
        name = next(c for c in map(chr, range(ord("A"), ord("Z") + 1)) if c not in used)
        row = dict(name=name, popularity=1.0, center_x=0.0, center_y=0.0,
                   voter_spread=1.0, candidate_spread=0.5)
        factions.value = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    async def run(_=None):
        active = factions.value[factions.value.popularity > 0].reset_index(drop=True)
        output.loading = True
        try:
            config = build_config(
                active,
                voters=voters.value,
                candidates=candidates.value,
                committee_size=committee_size.value,
                sigma=sigma.value,
                rrv_ranks=rrv_ranks.value,
                rrv_k=rrv_k.value,
            )
            table = await asyncio.to_thread(mcelect.simulate, config, trials.value)
        except ValueError as e:  # includes pydantic's ValidationError
            output.objects = [pn.pane.Alert(str(e), alert_type="danger")]
            return
        finally:
            output.loading = False

        results = ak.from_arrow(table)
        names = active.name.to_numpy()
        # Colored by row in the full table, so disabling a faction doesn't recolor the rest.
        cmap = dict(zip(factions.value.name, itertools.cycle(Category10_10)))
        by_method = {m: winners(results, m, names) for m in results.methods.fields}
        output.objects = [
            pn.FlexBox(*(position_plot(m, df, active, cmap) for m, df in by_method.items())),
            share_plot(names, population_share(results), by_method),
        ]

    add_faction.on_click(on_add_faction)
    run_button.on_click(run)
    pn.state.onload(run)

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
            rrv_ranks,
            rrv_k,
        ],
        sidebar_width=420,
        main=[output],
    )


def main() -> None:
    pn.serve(create_app, title="mcelect committee results", show=True)


if __name__.startswith("bokeh"):
    create_app().servable()
