"""Read annual observed SEEG emissions for a simulated metro region (ACP)."""

from pathlib import Path
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKBOOK = ROOT / "analysis/validation/real_world_data/emission_data_municipality.xlsx"
DEFAULT_MUNICIPALITIES = ROOT / "input/ACPs_MUN_CODES.csv"


def _normalize(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore").decode().upper().split()
    )


def load_metro_emissions(
    metro_region: str,
    xlsx_path: str | Path = DEFAULT_WORKBOOK,
    municipalities_path: str | Path = DEFAULT_MUNICIPALITIES,
) -> pd.DataFrame:
    """Return net annual emissions as a year-indexed ``emissions_tco2e`` frame.

    Metro names match ACPs_MUN_CODES.csv, ignoring case, accents and whitespace.
    Sum all sectors for the ACP's municipalities, selecting SEEG emission and
    removal rows (excluding NCI and bunker). Removals are already negative in
    the inventory, so net emissions are their signed sum. These totals are
    not yet adjusted to the simulated firms' emissions scope.

    All workbook years are retained. Missing annual values propagate to the
    metro total rather than being treated as zero. Unknown metros or missing
    municipalities raise ValueError to prevent incomplete regional totals.

    pandas reads the entire Dados sheet before filtering municipalities; this
    large workbook can take several minutes and substantial memory per call.
    No files are written and no simulation data are loaded.

    Example:
        emissions = load_metro_emissions("BELO HORIZONTE")
        recent = emissions.loc[2012:2024]
    """
    mapping = pd.read_csv(municipalities_path, sep=";", dtype=str)
    selected = mapping.loc[
        mapping["ACPs"].map(_normalize).eq(_normalize(metro_region)), "cod_mun"
    ].str.strip().drop_duplicates()
    if selected.empty:
        raise ValueError(f"Unknown metro region: {metro_region!r}")
    if not selected.str.fullmatch(r"\d{7}").all():
        raise ValueError("Municipality mapping must contain seven-digit IBGE codes.")

    # SEEG territory IDs prefix the seven-digit IBGE municipality code with 1.
    territory_column = "ID Territ\u00f3rio"
    type_column = "Emiss\u00e3o/Remo\u00e7\u00e3o/Bunker"
    expected_ids = set("1" + selected)
    raw = pd.read_excel(
        xlsx_path, sheet_name="Dados", engine="openpyxl",
        dtype={territory_column: str},
    )
    years = sorted(
        [c for c in raw.columns if str(c).isdigit() and len(str(c)) == 4],
        key=int,
    )
    if not years:
        raise ValueError("No annual emissions columns found in the Dados sheet.")
    ids = raw[territory_column].str.strip()
    rows = raw.loc[
        ids.isin(expected_ids) & raw[type_column].map(_normalize).isin(["EMISSAO", "REMOCAO"])
    ]
    missing = expected_ids - set(rows[territory_column].str.strip())
    if missing:
        raise ValueError(f"No emission or removal rows for municipality codes: {sorted(i[1:] for i in missing)}")

    values = rows[years].apply(pd.to_numeric, errors="raise")
    totals = values.sum(axis=0, skipna=False)
    result = pd.DataFrame(
        {"emissions_tco2e": totals.to_numpy(dtype=float)},
        index=pd.Index([int(y) for y in years], name="year"),
    )
    result.attrs.update(
        metro_region=_normalize(metro_region),
        municipality_codes=sorted(selected.tolist()),
        scope="net emissions (all sectors; emissions plus signed removals)",
        source=str(xlsx_path),
    )
    return result


def plot_emissions(emissions: pd.DataFrame, ax=None):
    """Plot a year-indexed frame returned by load_metro_emissions.

    Returns the Matplotlib Axes for further styling or saving. Does not call
    show() or write files. Slice the frame first to select a year range.

    Example:
        df = load_metro_emissions("BELO HORIZONTE")
        ax = plot_emissions(df.loc[2012:2024])
        ax.figure.savefig("emissions.pdf", bbox_inches="tight")
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if emissions.empty:
        raise ValueError("No emissions data to plot.")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4), layout="constrained")

    series = emissions["emissions_tco2e"].sort_index()
    ax.plot(series.index, series.values, color="#2166ac", linewidth=1.8)
    metro = emissions.attrs.get("metro_region")
    ax.set_title(f"Observed net emissions: {metro}" if metro else "Observed net emissions")
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Annual net emissions (tCO$_2$e)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    emissions = load_metro_emissions("BELO HORIZONTE")
    plot_emissions(emissions.loc[2012:2024])
    plt.show()
    pass
