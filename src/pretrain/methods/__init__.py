"""Pretrain method modules."""

from .edge_pred import EdgePrediction
from .attr_masking import AttrMasking
from .context_pred import ContextPred
from .dgi import DGI
from .infograph import InfoGraph
from .graphcl import GraphCL
from .supervised import Supervised

__all__ = [
    "AttrMasking",
    "ContextPred",
    "EdgePrediction",
    "DGI",
    "GraphCL",
    "InfoGraph",
    "Supervised",
]
