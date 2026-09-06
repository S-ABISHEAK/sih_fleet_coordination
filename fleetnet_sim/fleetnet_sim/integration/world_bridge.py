"""Layout JSON -> ``core.world.World`` + reverse indices.

This is the *only* place a warehouse_layout JSON gets turned into
something ``Fleet_SIH``'s real ``DStarLite`` can plan over. See the
plan's "The one required bridge" section for the rationale: the
navigation graph stays authoritative for topology/IDs (never
duplicated), the grid exists purely so D* Lite has cells to search.

Rasterization is at 1 cell = 1 meter, matching the coordinate contract
``Fleet_SIH/GAZEBO_INTEGRATION.md`` documents (so this bridge is
reusable for a later Gazebo port too). A cell is an obstacle iff its
center falls inside a rack, column, or exclusion polygon — aisles and
zones are never obstacles, and a wide aisle naturally rasterizes into
multiple free parallel columns of cells (this is what lets two robots
occupy adjacent cells and be planned/avoided independently rather than
funneled single-file).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from core.world import Cell, World


def _polygon_from_points(points: list[list[float]]) -> Polygon:
    return Polygon([(p[0], p[1]) for p in points])


@dataclass
class WorldBridge:
    """Holds the rasterized ``World`` plus every lookup index the
    simulation engine and telemetry layer need to relate grid cells back
    to the authoritative navigation graph's node/edge/zone IDs."""

    world: World
    origin_x: float
    origin_y: float
    cell_size: float

    node_positions: dict[str, tuple[float, float]]
    node_to_cell: dict[str, Cell]
    zone_to_cells: dict[str, set[Cell]]
    dock_role_to_zone: dict[str, str]  # "receiving"/"shipping" -> zone_id used as its Task endpoint

    _node_tree: STRtree = field(repr=False)
    _node_ids_by_tree_index: list[str] = field(repr=False)
    _edge_tree: STRtree = field(repr=False)
    _edge_ids_by_tree_index: list[tuple[str, str]] = field(repr=False)  # (u, v) node id pairs
    cell_to_zone: dict[Cell, str] = field(default_factory=dict)  # reverse of zone_to_cells, for occupancy telemetry

    def world_to_cell(self, x: float, y: float) -> Cell:
        return (int((x - self.origin_x) / self.cell_size), int((y - self.origin_y) / self.cell_size))

    def cell_to_world(self, cx: int, cy: int) -> tuple[float, float]:
        return (
            self.origin_x + (cx + 0.5) * self.cell_size,
            self.origin_y + (cy + 0.5) * self.cell_size,
        )

    def nearest_node_id(self, x: float, y: float) -> str | None:
        if not self._node_ids_by_tree_index:
            return None
        idx = self._node_tree.nearest(Point(x, y))
        return self._node_ids_by_tree_index[int(idx)]

    def nearest_edge(self, x: float, y: float) -> tuple[str, str] | None:
        if not self._edge_ids_by_tree_index:
            return None
        idx = self._edge_tree.nearest(Point(x, y))
        return self._edge_ids_by_tree_index[int(idx)]

    def nearest_free_cell(self, x: float, y: float, max_radius_cells: int = 40) -> Cell:
        """BFS outward from the nearest cell to (x, y) until a free cell
        is found — used to seed dock/zone-entry cells that might land
        exactly on a wall-adjacent obstacle cell.

        Deliberately walks *raw* 8-connected adjacency (not
        ``World.neighbors``, which only ever yields already-free cells):
        starting from an obstacle cell whose immediate neighbors are all
        themselves obstacles (e.g. the interior of a multi-cell-thick
        rack block) needs to traverse obstacle-to-obstacle to reach the
        block's edge — `World.neighbors` can't do that since it filters
        to free cells relative to a query cell, not relative to `start`."""
        start = self.world_to_cell(x, y)
        if self.world.is_free(start):
            return start
        w, h = self.world.width, self.world.height
        seen = {start}
        frontier = [start]
        for _ in range(max_radius_cells):
            next_frontier = []
            for cx, cy in frontier:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        n = (cx + dx, cy + dy)
                        if n in seen or not (0 <= n[0] < w and 0 <= n[1] < h):
                            continue
                        seen.add(n)
                        if self.world.is_free(n):
                            return n
                        next_frontier.append(n)
            frontier = next_frontier
            if not frontier:
                break
        return start  # give up gracefully; caller may still fail to route, which is a validity signal


def build_world_bridge(layout: dict) -> WorldBridge:
    footprint_pts = layout["geometry"]["footprint"]
    xs = [p[0] for p in footprint_pts]
    ys = [p[1] for p in footprint_pts]
    origin_x, origin_y = min(xs), min(ys)
    width_m, height_m = max(xs) - origin_x, max(ys) - origin_y

    cell_size = 1.0
    grid_w = max(1, math.ceil(width_m / cell_size))
    grid_h = max(1, math.ceil(height_m / cell_size))
    world = World(grid_w, grid_h, strict_diagonal_corners=True)

    obstacle_objects = layout["geometry"]["racks"] + layout["geometry"]["columns"] + layout["geometry"]["exclusions"]
    for obj in obstacle_objects:
        poly = _polygon_from_points(obj["points"])
        minx, miny, maxx, maxy = poly.bounds
        cx0 = max(0, int((minx - origin_x) / cell_size))
        cy0 = max(0, int((miny - origin_y) / cell_size))
        cx1 = min(grid_w - 1, int((maxx - origin_x) / cell_size))
        cy1 = min(grid_h - 1, int((maxy - origin_y) / cell_size))
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                wx = origin_x + (cx + 0.5) * cell_size
                wy = origin_y + (cy + 0.5) * cell_size
                if poly.contains(Point(wx, wy)):
                    world.add_obstacle((cx, cy))

    graph = layout["navigation_graph"]
    node_positions: dict[str, tuple[float, float]] = {}
    for node in graph["nodes"]:
        nid = node["id"]
        node_positions[nid] = (float(node["x"]), float(node["y"]))

    node_ids_by_tree_index = list(node_positions.keys())
    node_points = [Point(*node_positions[nid]) for nid in node_ids_by_tree_index]
    node_tree = STRtree(node_points) if node_points else STRtree([])

    edge_ids_by_tree_index: list[tuple[str, str]] = []
    edge_lines = []
    from shapely.geometry import LineString

    for link in graph.get("links", graph.get("edges", [])):
        u, v = link["source"], link["target"]
        if u not in node_positions or v not in node_positions:
            continue
        edge_ids_by_tree_index.append((u, v))
        edge_lines.append(LineString([node_positions[u], node_positions[v]]))
    edge_tree = STRtree(edge_lines) if edge_lines else STRtree([])

    def _bridge_stub() -> WorldBridge:
        return WorldBridge(
            world=world,
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size=cell_size,
            node_positions=node_positions,
            node_to_cell={},
            zone_to_cells={},
            dock_role_to_zone={},
            _node_tree=node_tree,
            _node_ids_by_tree_index=node_ids_by_tree_index,
            _edge_tree=edge_tree,
            _edge_ids_by_tree_index=edge_ids_by_tree_index,
        )

    bridge = _bridge_stub()

    node_to_cell: dict[str, Cell] = {}
    for nid, (x, y) in node_positions.items():
        node_to_cell[nid] = bridge.nearest_free_cell(x, y)
    bridge.node_to_cell = node_to_cell

    zone_to_cells: dict[str, set[Cell]] = {}
    for zone in layout["geometry"]["zones"]:
        zt = zone["metadata"]["zone_type"]
        poly = _polygon_from_points(zone["points"])
        minx, miny, maxx, maxy = poly.bounds
        cx0 = max(0, int((minx - origin_x) / cell_size))
        cy0 = max(0, int((miny - origin_y) / cell_size))
        cx1 = min(grid_w - 1, int((maxx - origin_x) / cell_size))
        cy1 = min(grid_h - 1, int((maxy - origin_y) / cell_size))
        cells = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                wx = origin_x + (cx + 0.5) * cell_size
                wy = origin_y + (cy + 0.5) * cell_size
                if poly.contains(Point(wx, wy)) and world.is_free((cx, cy)):
                    cells.add((cx, cy))
        zone_to_cells[zt] = cells
    bridge.zone_to_cells = zone_to_cells

    cell_to_zone: dict[Cell, str] = {}
    for zt, cells in zone_to_cells.items():
        for cell in cells:
            cell_to_zone[cell] = zt
    bridge.cell_to_zone = cell_to_zone

    dock_role_to_zone = {}
    for dock in layout["geometry"]["docks"]:
        role = dock["metadata"].get("role")
        if role:
            dock_role_to_zone[role] = role  # zone_type == role for receiving/shipping by construction
    bridge.dock_role_to_zone = dock_role_to_zone

    return bridge
