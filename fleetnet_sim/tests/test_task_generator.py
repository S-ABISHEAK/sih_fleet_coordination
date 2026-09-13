"""Regression guard for TaskGenerator's picking->rack-aisle routing
(core/task_generator.py::_cells_for). Measures the mechanism directly
at the source -- calling task generation many times against a real
WorldBridge -- rather than through sparse robot_state_ts telemetry
sampling, which only incidentally catches a robot mid-pickup and gives
a tiny, misleading denominator (5 samples in one real run, versus the
1000+ direct draws here).
"""
from __future__ import annotations

import numpy as np
import pytest
from fleetnet_layout.config.enums import Archetype, ScaleClass
from fleetnet_layout.generation.generator import generate_layout
from fleetnet_layout.generation.sampling import SamplingOverrides
from fleetnet_layout.serialization.json_io import to_json_dict

from fleetnet_sim.config.schema import TaskGenConfig
from fleetnet_sim.core.task_generator import TaskGenerator
from fleetnet_sim.integration.world_bridge import build_world_bridge

N_DRAWS = 1000


@pytest.fixture(scope="module")
def medium_grid_bridge():
    """A MEDIUM grid layout -- big enough to have a real secondary-aisle
    population (unlike the tiny SMALL fixture other tests use), small
    enough to build/rasterize fast in a test."""
    layout = generate_layout(7, SamplingOverrides(archetype=Archetype.GRID, scale_class=ScaleClass.MEDIUM))
    data = to_json_dict(layout.config, layout.result, layout.graph, layout.metrics, layout.validation)
    return build_world_bridge(data)


def test_rack_aisle_cells_populated_and_disjoint_from_picking_zone(medium_grid_bridge):
    bridge = medium_grid_bridge
    assert bridge.rack_aisle_cells, "expected a non-empty rack_aisle_cells pool on a MEDIUM grid layout"
    picking_zone_cells = bridge.zone_to_cells.get("picking", set())
    assert picking_zone_cells, "fixture layout should have a picking zone"
    # The two pools should be a genuinely different (and here, disjoint)
    # part of the map -- rack aisles sit in the storage field, the
    # picking zone sits in the dock-wall band.
    assert not (bridge.rack_aisle_cells & picking_zone_cells)


def test_picking_tasks_target_rack_aisle_cells_not_the_zone_band(medium_grid_bridge):
    """The actual regression guard: every 'picking' cell TaskGenerator
    hands out must come from rack_aisle_cells (not the flat zone
    polygon) whenever that pool exists -- exercised N_DRAWS times
    directly, not inferred from a handful of lucky telemetry samples."""
    bridge = medium_grid_bridge
    gen = TaskGenerator(bridge=bridge, config=TaskGenConfig(arrival_rate_per_s=1.0), rng=np.random.default_rng(0))

    picking_draws = 0
    in_rack_aisle = 0
    for _ in range(N_DRAWS):
        cell = gen._pick_cell("picking")
        picking_draws += 1
        if cell in bridge.rack_aisle_cells:
            in_rack_aisle += 1

    assert picking_draws == N_DRAWS
    assert in_rack_aisle == N_DRAWS, (
        f"expected 100% of {N_DRAWS} 'picking' cell draws to land in rack_aisle_cells, got {in_rack_aisle}/{N_DRAWS}"
    )


def test_make_task_end_to_end_respects_the_same_routing(medium_grid_bridge):
    """Same guard, but through the real _make_task()/FLOW_EDGES path
    (not calling _pick_cell directly), over enough draws to collect a
    real sample of picking-sourced and picking-destined tasks."""
    bridge = medium_grid_bridge
    gen = TaskGenerator(bridge=bridge, config=TaskGenConfig(arrival_rate_per_s=1.0), rng=np.random.default_rng(1))

    picking_pickups = 0
    picking_dropoffs = 0
    picking_pickups_in_aisle = 0
    picking_dropoffs_in_aisle = 0
    for _ in range(N_DRAWS):
        task, record = gen._make_task(sim_time=0.0)
        if record.source_zone_id == "picking":
            picking_pickups += 1
            if task.pickup in bridge.rack_aisle_cells:
                picking_pickups_in_aisle += 1
        if record.destination_zone_id == "picking":
            picking_dropoffs += 1
            if task.dropoff in bridge.rack_aisle_cells:
                picking_dropoffs_in_aisle += 1

    # FLOW_EDGES guarantees both directions occur with real weight
    # (picking->packing/returns as source, staging->picking as dest) --
    # a real denominator, not a lucky handful.
    assert picking_pickups > 50, f"too few picking-sourced tasks sampled to be meaningful: {picking_pickups}"
    assert picking_dropoffs > 50, f"too few picking-destined tasks sampled to be meaningful: {picking_dropoffs}"
    assert picking_pickups_in_aisle == picking_pickups
    assert picking_dropoffs_in_aisle == picking_dropoffs
