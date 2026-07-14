"""Diseases domain: flat pool of diseases with associated symptoms."""

from .loader import load_flat_disease_candidates, load_flat_disease_graph

__all__ = ["load_flat_disease_candidates", "load_flat_disease_graph"]
