"""Shared IEEE-size typography and export settings for manuscript figures."""
from pathlib import Path
import matplotlib
from matplotlib.text import Text

COL_W = 252.0 / 72.27
TEXT_W = 516.0 / 72.27
C_LIGHT, C_MID, C_DARK = "#CBE4ED", "#0072B2", "#203D4A"
C_WARN, C_WARN_FILL = "#B23A48", "#F2D9DC"
FS_TEXT, FS_NOTE, FS_LEG, FS_TITLE = 9.0, 9.0, 9.0, 9.5


def use(usetex=False):
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
        "font.size": FS_TEXT,
        "mathtext.fontset": "stix",
        "text.usetex": usetex,
        "axes.labelsize": FS_TEXT,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "normal",
        "axes.titlepad": 6,
        "axes.labelpad": 3,
        "xtick.labelsize": FS_NOTE,
        "ytick.labelsize": FS_NOTE,
        "legend.fontsize": FS_LEG,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .65,
        "axes.edgecolor": "#5B6065",
        "axes.labelcolor": "#20252A",
        "text.color": "#20252A",
        "xtick.color": "#20252A",
        "ytick.color": "#20252A",
        "xtick.major.width": .6,
        "ytick.major.width": .6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "lines.linewidth": 1.4,
        "patch.linewidth": .7,
        "hatch.linewidth": .45,
        "grid.linewidth": .5,
        "grid.alpha": .18,
        "savefig.bbox": None,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.w_pad": .055,
        "figure.constrained_layout.h_pad": .065,
    }
    if usetex:
        rc["text.latex.preamble"] = r"\usepackage{times}\usepackage{amsmath,amssymb}"
    matplotlib.rcParams.update(rc)
    return rc


def save(fig, stem):
    """Export at final print width with embedded fonts and legible labels."""
    stem = Path(stem)
    for text in fig.findobj(match=Text):
        text.set_fontfamily("serif")
        text.set_fontsize(max(FS_TEXT, min(10., text.get_fontsize())))
        text.set_fontweight("normal")
    fig.savefig(stem.with_suffix('.pdf'), dpi=600, bbox_inches=None,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix('.png'), dpi=240, bbox_inches=None)
