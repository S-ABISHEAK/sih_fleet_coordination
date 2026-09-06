def test_bridge_builds_grid_and_zones(small_bridge):
    assert small_bridge.world.width > 0 and small_bridge.world.height > 0
    assert len(small_bridge.world.obstacles) > 0
    assert any(small_bridge.zone_to_cells.values())


def test_nearest_lookups_return_something(small_bridge):
    cx, cy = small_bridge.world.width // 2, small_bridge.world.height // 2
    x, y = small_bridge.cell_to_world(cx, cy)
    assert small_bridge.nearest_node_id(x, y) is not None
    assert small_bridge.nearest_edge(x, y) is not None


def test_nearest_free_cell_is_actually_free(small_bridge):
    # Pick a cell known to be an obstacle and confirm we get pushed off it.
    obstacle_cell = next(iter(small_bridge.world.obstacles))
    x, y = small_bridge.cell_to_world(*obstacle_cell)
    free_cell = small_bridge.nearest_free_cell(x, y)
    assert small_bridge.world.is_free(free_cell)
