import matplotlib
import seaborn as sns
import hypercluster
from hypercluster import (
    utilities, additional_clusterers, additional_metrics, classes, constants, visualize
)
from hypercluster.classes import AutoClusterer, MultiAutoClusterer
__version__ = '0.1.9'
__all__ = [
    "AutoClusterer",
    "MultiAutoClusterer"
]

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set(font="arial", style="white", color_codes=True, font_scale=1.3)
matplotlib.rcParams.update({"savefig.bbox": "tight"})