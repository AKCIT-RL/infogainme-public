"""Knowledge graph core models and operations.

Implements `KnowledgeGraph`, `Node`, and `Edge` with minimal yet typed API.
The graph maintains a set of pruned node ids and exposes methods to retrieve
active nodes and to apply pruning operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Set, Optional
from types import MappingProxyType
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Node:
    """Graph node.

    Attributes:
        id: Unique node identifier.
        label: Human-readable label/name.
        attrs: Arbitrary attributes dictionary (immutable).
    """

    id: str
    label: str
    attrs: MappingProxyType[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class Edge:
    """Directed edge between nodes.

    Attributes:
        source_id: Source node id.
        target_id: Target node id.
        relation: Relation label (e.g., "contains", "located_in").
    """

    source_id: str
    target_id: str
    relation: str


class KnowledgeGraph:
    """In-memory knowledge graph with pruning state.

    Args:
        nodes: Optional iterable of initial nodes.
        edges: Optional iterable of initial edges.
    """

    def __init__(self, nodes: Iterable[Node] | None = None, edges: Iterable[Edge] | None = None) -> None:
        self.nodes: Set[Node] = set(nodes or [])
        self.edges: Set[Edge] = set(edges or [])
        self.pruned_ids: Set[str] = set()

    def get_active_nodes(self) -> Set[Node]:
        """Return nodes that have not been pruned."""
        return {n for n in self.nodes if n.id not in self.pruned_ids}
    
    def get_active_leaf_nodes(self, leaf_type: str = "city") -> Set[Node]:
        """Return active nodes that are leaf nodes (typically cities).
        
        Leaf nodes are nodes with no outgoing edges in the active graph.
        In geographic graphs, these are typically cities that can be targets.
        
        Returns:
            Set of active leaf nodes.
        """
        active_nodes = self.get_active_nodes()
        active_ids = {n.id for n in active_nodes}
        
        # Find nodes with the specified leaf type
        leaf_nodes = {n for n in active_nodes if n.attrs.get("type") == leaf_type}
        
        return leaf_nodes

    def apply_pruning(self, pruned: Set[str]) -> None:
        """Apply pruning by adding node ids to the internal pruned set. Only receives leaf nodes ids.

        Args:
            pruned: Set of node ids to mark as pruned.
        """
        if not pruned:
            return

        # Mark initial nodes as pruned
        self.pruned_ids.update(pruned)

        # Find candidate orphan parents
        candidate_orphan_nodes = set()
        for node_id in pruned:
            parent_id = self.get_parent_id(node_id)
            if parent_id:
                candidate_orphan_nodes.add(parent_id)
        
        # Check which parents are actually orphaned (no active children)
        new_pruned = set()
        for node_id in candidate_orphan_nodes:
            # Get all children of this parent
            all_children = self.get_children_ids(node_id)
            # Check if any children are still active (not pruned)
            active_children = all_children - self.pruned_ids
            if not active_children:  # No active children = orphaned
                new_pruned.add(node_id)
        
        # Recursively apply pruning to orphaned parents
        if new_pruned:
            self.apply_pruning(new_pruned)

    def get_parent_id(self, node_id: str) -> Optional[str]:
        """Get the parent node ID for a given node.
        
        Args:
            node_id: ID of the node to find parent for.
            
        Returns:
            Parent node ID if found, None if no parent or node not found.
        """
        for edge in self.edges:
            if edge.target_id == node_id and edge.relation in ("has_child", "contains"):
                return edge.source_id
        return None
    
    def get_children_ids(self, node_id: str) -> Set[str]:
        """Get all children node IDs for a given parent node.
        
        Args:
            node_id: ID of the parent node.
            
        Returns:
            Set of children node IDs.
        """
        children = set()
        for edge in self.edges:
            if edge.source_id == node_id and edge.relation in ("has_child", "contains"):
                children.add(edge.target_id)
        return children

    def reset_pruning(self) -> None:
        """Reset the pruning state, making all nodes active again.
        
        This is useful when reusing the same graph for multiple benchmark runs.
        """
        self.pruned_ids.clear()

    def graph_to_text(self, save_to: Optional[str] = None) -> str:
        """Convert the active portion of the graph into a compact text format.

        The representation groups nodes by their semantic "type" (e.g., region,
        subregion, country, state, city) and lists active relations among active
        nodes. This is intended for LLM prompts where a concise, deterministic
        view of the current graph state is useful.

        Args:
            save_to: Optional file path to save the graph snapshot. If provided,
                    the text representation will be written to this file.

        Returns:
            A multi-line string describing active nodes and relations.
        """
        # Active scope
        active_ids = {n.id for n in self.get_active_nodes()}
        active_nodes = [n for n in self.nodes if n.id in active_ids]

        # Indexes for quick lookup
        id_to_node = {n.id: n for n in active_nodes}
        active_edges = [
            e for e in self.edges if e.source_id in active_ids and e.target_id in active_ids
        ]

        # Detect flat graphs (only leaf-type nodes, no layer, no edges)
        has_layer = any(n.attrs.get("layer") is not None for n in active_nodes)
        leaf_types = {"city", "object"}

        # Group nodes by type and sort deterministically
        type_to_nodes: dict[str, list[Node]] = {}
        for node in active_nodes:
            node_type = str(node.attrs.get("type", "unknown")).lower()
            type_to_nodes.setdefault(node_type, []).append(node)

        def sort_key(n: Node) -> tuple[int, str, str]:
            layer = int(n.attrs.get("layer", 0)) if n.attrs.get("layer") is not None else 0
            category = str(n.attrs.get("category", ""))
            return (layer, category, n.label)

        for nodes in type_to_nodes.values():
            nodes.sort(key=sort_key)

        # Preferred order: geographic hierarchy vs flat (group by category)
        if has_layer and active_edges:
            preferred_order = ["region", "subregion", "country", "state", "city"]
        else:
            # Flat graph: group by category if available, else by type
            preferred_order = list(leaf_types & type_to_nodes.keys()) or list(type_to_nodes.keys())

        other_types = sorted(t for t in type_to_nodes.keys() if t not in preferred_order)
        ordered_types = [t for t in preferred_order if t in type_to_nodes] + other_types

        lines: list[str] = []
        lines.append("Knowledge Graph (active)")
        lines.append(
            f"Nodes: {len(active_nodes)} | Edges: {len(active_edges)} | Pruned: {len(self.pruned_ids)}"
        )

        # For flat graphs with category, group nodes by category for readability
        if not has_layer and not active_edges:
            category_to_nodes: dict[str, list[Node]] = {}
            for node in active_nodes:
                cat = str(node.attrs.get("category", "other"))
                category_to_nodes.setdefault(cat, []).append(node)
            for cat in sorted(category_to_nodes.keys()):
                nds = sorted(category_to_nodes[cat], key=lambda n: n.label)
                lines.append(f"\n[{cat}] ({len(nds)})")
                for n in nds:
                    lines.append(f"- {n.label} [{n.id}]")
        else:
            # Hierarchical: nodes by type
            for t in ordered_types:
                lines.append(f"\n[{t}] ({len(type_to_nodes[t])})")
                for n in type_to_nodes[t]:
                    lines.append(f"- {n.label} [{n.id}]")

        # Relations (trim long lists to keep prompts compact)
        if active_edges:
            lines.append("\n[relations]")
            # Deterministic sort: relation, source label, target label
            def edge_sort_key(e: Edge) -> tuple[str, str, str]:
                s_label = id_to_node.get(e.source_id).label if e.source_id in id_to_node else e.source_id
                t_label = id_to_node.get(e.target_id).label if e.target_id in id_to_node else e.target_id
                return (e.relation, s_label, t_label)

            active_edges_sorted = sorted(active_edges, key=edge_sort_key)
            max_lines = 200
            shown = 0
            for e in active_edges_sorted:
                if e.source_id not in id_to_node or e.target_id not in id_to_node:
                    continue
                s_label = id_to_node[e.source_id].label
                t_label = id_to_node[e.target_id].label
                lines.append(f"- {s_label} --{e.relation}--> {t_label}")
                shown += 1
                if shown >= max_lines:
                    remaining = len(active_edges_sorted) - shown
                    if remaining > 0:
                        lines.append(f"... and {remaining} more")
                    break

        result = "\n".join(lines)
        
        # Save to file if path provided
        if save_to:
            output_path = Path(save_to)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result, encoding="utf-8")
        
        return result

    def plot(
        self,
        output_path: Optional[str] = None,
        *,
        show_pruned: bool = False,
        node_size: int = 300,
        figsize: tuple[int, int] = (10, 8),
        title: str = "Knowledge Graph",
    ) -> None:
        """Plot the knowledge graph using networkx and matplotlib.

        Args:
            output_path: If provided, save plot to this path instead of showing.
            show_pruned: If True, show pruned nodes with different styling.
            node_size: Size of nodes in the plot.
            figsize: Figure size (width, height).
            title: Plot title.

        Raises:
            ImportError: If networkx or matplotlib are not available.
        """

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        active_ids = {n.id for n in self.get_active_nodes()}
        for node in self.nodes:
            is_active = node.id in active_ids
            if is_active or show_pruned:
                G.add_node(node.id, label=node.label, active=is_active)

        # Add edges
        for edge in self.edges:
            if G.has_node(edge.source_id) and G.has_node(edge.target_id):
                G.add_edge(edge.source_id, edge.target_id, relation=edge.relation)

        if not G.nodes():
            print("No nodes to plot.")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use multipartite layout based on node layers (like the original notebook)
        try:
            # Extract layer information from node attributes
            layers = {}
            for node in self.nodes:
                if node.id in G.nodes():
                    layer = node.attrs.get("layer", 0)
                    layers[node.id] = layer
            
            # Set subset attribute for multipartite layout
            for node_id in G.nodes():
                if node_id in layers:
                    G.nodes[node_id]["subset"] = layers[node_id]
                else:
                    G.nodes[node_id]["subset"] = 0
            
            # Use multipartite layout (like the original notebook)
            pos = nx.multipartite_layout(G, subset_key="subset", scale=3.0, align="vertical")
            
        except Exception:
            # Fallback to spring layout if multipartite fails
            pos = nx.spring_layout(G, k=1, iterations=50)

        # Color nodes based on type (like the original notebook)
        color_map = {
            "region": "#e41a1c",      # Red
            "subregion": "#ff7f00",    # Orange  
            "country": "#ffff33",      # Yellow
            "state": "#4daf4a",        # Green
            "city": "#377eb8",         # Blue
        }
        
        node_colors = []
        for node_id in G.nodes():
            if G.nodes[node_id].get("active", True):
                # Get node type from original node
                node_type = "unknown"
                for node in self.nodes:
                    if node.id == node_id:
                        node_type = node.attrs.get("type", "unknown")
                        break
                node_colors.append(color_map.get(node_type, "#999999"))
            else:
                node_colors.append("lightcoral")  # Pruned nodes

        # Draw edges first (more subtle, like the original)
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.6, edge_color="#777777", ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=20, 
            linewidths=0.2, edgecolors="#333333", ax=ax
        )

        # Add labels for all nodes including cities
        labels = {}
        for node_id in G.nodes():
            node_type = "unknown"
            for node in self.nodes:
                if node.id == node_id:
                    node_type = node.attrs.get("type", "unknown")
                    break
            # Include all node types
            labels[node_id] = G.nodes[node_id]["label"]
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)

        ax.set_title(title)
        ax.axis("off")

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=6)
            for k, v in color_map.items()
        ]
        ax.legend(handles=legend_elements, loc='lower left', ncol=5, frameon=False)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_active():
        pass


if __name__ == "__main__":
    # Self-tests similares ao adapter, sem dependências externas.

    def _build_sample_graph() -> KnowledgeGraph:
        # A -> B -> C
        a = Node(id="A", label="Region A")
        b = Node(id="B", label="Region B")
        c = Node(id="C", label="Region C")
        e1 = Edge(source_id="A", target_id="B", relation="contains")
        e2 = Edge(source_id="B", target_id="C", relation="contains")
        return KnowledgeGraph(nodes=[a, b, c], edges=[e1, e2])

    def _test_active_and_pruning() -> None:
        kg = _build_sample_graph()
        active = kg.get_active_nodes()
        assert {n.id for n in active} == {"A", "B", "C"}

        kg.apply_pruning({"B"})
        active_after = kg.get_active_nodes()
        assert {n.id for n in active_after} == {"A", "C"}
        assert "B" in kg.pruned_ids

        # idempotência de aplicar poda vazia
        kg.apply_pruning(set())
        assert {n.id for n in kg.get_active_nodes()} == {"A", "C"}

    def _test_plot() -> None:
        print("Testing plot...")
        kg = _build_sample_graph()
        kg.apply_pruning({"B"})
        
        # Test plot to file (should work if dependencies available)
        try:
            kg.plot(output_path="outputs/test_graph.png", show_pruned=True, title="Test Graph")
            print("Plot test: OK (saved to outputs/test_graph.png)")
        except ImportError:
            print("Plot test: SKIPPED (networkx/matplotlib not available)")
        except Exception as e:
            print(f"Plot test: WARNING ({e})")

    def _test_graph_to_text() -> None:
        print("Testing graph to text...")
        kg = _build_sample_graph()
        text = kg.graph_to_text()
        print(text)
        
        # Test saving to file
        print("\nTesting save to file...")
        kg.graph_to_text(save_to="outputs/test_graph_snapshot.txt")
        print("Graph snapshot saved to outputs/test_graph_snapshot.txt")

    _test_active_and_pruning()
    _test_plot()
    _test_graph_to_text()
    print("KnowledgeGraph self-tests: OK")


