"""L-flow archetype (Section B): receiving and shipping on adjacent
(perpendicular) walls, corner-biased flow.

KNOWN LIMITATION (v1): a true corner geometry needs the zone/dock band
packer to support east/west walls, which ``generation.zones``/
``generation.docks`` don't yet (see ``generation.sampling``'s dock_wall
comment and ``docks.py``'s module docstring) — building that properly
means making the packers axis-aware, deferred past this revision. As an
honest approximation that's still structurally distinct from `grid`
rather than a relabeled copy: receiving is forced to the near corner and
shipping to the far corner of the *same* dock wall (vs. `grid`'s
area-proportional zone ordering), which is the part of L-flow's
signature ("corner-zone logic") that doesn't require a second wall.
"""
from __future__ import annotations

import numpy as np

from fleetnet_layout.config.schema import LayoutConfig
from fleetnet_layout.generation.archetypes.base import (
    finalize_one_way,
    reserve_zone_band,
    sample_and_place_docks,
)
from fleetnet_layout.generation.constraints import apply_subtractive_layer, generate_column_grid
from fleetnet_layout.generation.storage import build_storage_field
from fleetnet_layout.generation.types import LayoutBuildResult
from fleetnet_layout.generation.zones import pack_zone_band
from fleetnet_layout.geometry.primitives import Rect

_CORNER_ORDER = ["receiving", "returns", "staging", "picking", "packing", "charging", "office", "shipping"]


def build(rng: np.random.Generator, cfg: LayoutConfig) -> LayoutBuildResult:
    footprint = Rect(0.0, 0.0, cfg.warehouse.length_m, cfg.warehouse.width_m)
    wall = cfg.structural_constraints.dock_wall

    enabled_zone_area = sum(getattr(cfg.zones, name).area_m2 for name in type(cfg.zones).model_fields if getattr(cfg.zones, name).enabled)
    band_depth = max(min(enabled_zone_area / footprint.w, footprint.h * 0.35), 8.0)

    zone_band, storage_rect = reserve_zone_band(footprint, wall, band_depth)
    zone_result = pack_zone_band(zone_band, cfg.zones, id_prefix="z0", order=_CORNER_ORDER)

    field = build_storage_field(rng, storage_rect, cfg.storage, cfg.aisles, id_prefix="f0")

    columns = generate_column_grid(
        rng, storage_rect, cfg.structural_constraints.column_grid_spacing_m, cfg.diversity_controls.irregularity_level
    )
    racks = apply_subtractive_layer(field.racks, columns)

    docks: list = []
    for z in zone_result.zones:
        zt = z.metadata["zone_type"]
        if zt not in ("receiving", "shipping"):
            continue
        minx, miny, maxx, maxy = z.polygon.bounds
        zr = Rect(minx, miny, maxx - minx, maxy - miny)
        docks += sample_and_place_docks(rng, zr, wall, cfg, role=zt, id_prefix=zt)

    result = LayoutBuildResult(
        footprint=footprint,
        racks=racks,
        main_aisles=field.main_aisles,
        secondary_aisles=field.secondary_aisles,
        cross_aisles=field.cross_aisles,
        zones=zone_result.zones,
        docks=docks,
        columns=columns,
    )
    if field.main_aisles:
        result.fire_lane_ids = [field.main_aisles[0].id]
    finalize_one_way(rng, result, cfg)
    return result
