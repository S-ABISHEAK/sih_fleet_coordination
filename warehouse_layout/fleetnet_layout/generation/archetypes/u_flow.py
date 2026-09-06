"""U-flow archetype (Section B): receiving and shipping on the *same*
wall; storage sits between them and flow loops around the storage
block's far end. The structural signature that distinguishes this from
`grid` is the perimeter loop aisle: a main-aisle-width strip wrapping the
far wall and both side walls of the storage field, connecting back to
the dock wall on either side so the aisle network forms a loop rather
than a simple comb.
"""
from __future__ import annotations

import numpy as np

from fleetnet_layout.config.enums import ArchetypeRole
from fleetnet_layout.config.schema import LayoutConfig
from fleetnet_layout.generation.archetypes.base import (
    finalize_one_way,
    opposite_wall,
    reserve_zone_band,
    sample_and_place_docks,
)
from fleetnet_layout.generation.constraints import apply_subtractive_layer, generate_column_grid
from fleetnet_layout.generation.storage import build_storage_field
from fleetnet_layout.generation.types import LayoutBuildResult
from fleetnet_layout.generation.zones import pack_zone_band
from fleetnet_layout.geometry.primitives import GeoObject, Rect


def build(rng: np.random.Generator, cfg: LayoutConfig) -> LayoutBuildResult:
    footprint = Rect(0.0, 0.0, cfg.warehouse.length_m, cfg.warehouse.width_m)
    wall = cfg.structural_constraints.dock_wall
    far_wall = opposite_wall(wall)

    enabled_zone_area = sum(getattr(cfg.zones, name).area_m2 for name in type(cfg.zones).model_fields if getattr(cfg.zones, name).enabled)
    band_depth = max(min(enabled_zone_area / footprint.w, footprint.h * 0.3), 8.0)

    zone_band, storage_rect = reserve_zone_band(footprint, wall, band_depth)
    zone_result = pack_zone_band(zone_band, cfg.zones, id_prefix="z0")

    from fleetnet_layout.generation.sampling import aisle_width_m

    loop_w = aisle_width_m(rng, cfg.aisles.main_width_class)
    loop_w = min(loop_w, storage_rect.w * 0.2, storage_rect.h * 0.3)

    inner_rect = Rect(
        storage_rect.x + loop_w,
        storage_rect.y,
        max(storage_rect.w - 2 * loop_w, 1.0),
        max(storage_rect.h - loop_w, 1.0),
    )

    field = build_storage_field(rng, inner_rect, cfg.storage, cfg.aisles, id_prefix="f0")

    loop_aisles = [
        GeoObject(
            id="aisle_loop_far",
            kind="aisle",
            polygon=Rect(storage_rect.x, storage_rect.y + storage_rect.h - loop_w, storage_rect.w, loop_w).to_polygon(),
            metadata={"archetype_role": ArchetypeRole.MAIN.value, "width_class": cfg.aisles.main_width_class.value, "width_m": loop_w, "one_way": False, "loop": True},
        ),
        GeoObject(
            id="aisle_loop_left",
            kind="aisle",
            polygon=Rect(storage_rect.x, storage_rect.y, loop_w, storage_rect.h).to_polygon(),
            metadata={"archetype_role": ArchetypeRole.MAIN.value, "width_class": cfg.aisles.main_width_class.value, "width_m": loop_w, "one_way": False, "loop": True},
        ),
        GeoObject(
            id="aisle_loop_right",
            kind="aisle",
            polygon=Rect(storage_rect.x + storage_rect.w - loop_w, storage_rect.y, loop_w, storage_rect.h).to_polygon(),
            metadata={"archetype_role": ArchetypeRole.MAIN.value, "width_class": cfg.aisles.main_width_class.value, "width_m": loop_w, "one_way": False, "loop": True},
        ),
    ]

    columns = generate_column_grid(
        rng, inner_rect, cfg.structural_constraints.column_grid_spacing_m, cfg.diversity_controls.irregularity_level
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
        main_aisles=field.main_aisles + loop_aisles,
        secondary_aisles=field.secondary_aisles,
        cross_aisles=field.cross_aisles,
        zones=zone_result.zones,
        docks=docks,
        columns=columns,
    )
    result.fire_lane_ids = ["aisle_loop_far"]
    finalize_one_way(rng, result, cfg)
    return result
