"""Flat objects domain (non-hierarchical dataset).

Load from CSV: data/objects/objects_test.csv (quick) or objects_full.csv (complete).
"""

from .loader import load_flat_object_candidates, load_flat_object_graph

__all__ = ["load_flat_object_candidates", "load_flat_object_graph"]
