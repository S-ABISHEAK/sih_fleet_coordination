from fleetnet_sim.config.schema import ExperimentConfig, FleetConfig, SimulationConfig, TaskGenConfig, WarehouseConfig
from fleetnet_sim.core.engine import Engine
from fleetnet_sim.core.robot_agent import TaskState


def _config(seed=1, robots=4, duration=30.0, arrival=0.5):
    return ExperimentConfig(
        experiment_id="exp_test",
        run_id=f"run_{seed}",
        warehouse=WarehouseConfig(layout_path="in-memory"),
        simulation=SimulationConfig(dt=0.1, duration_s=duration, seed=seed),
        fleet=FleetConfig(robot_count=robots),
        tasks=TaskGenConfig(arrival_rate_per_s=arrival),
    )


def test_engine_runs_without_crashing(small_bridge):
    eng = Engine(config=_config(), bridge=small_bridge)
    eng.run()
    assert eng.clock.tick == 300


def test_no_collisions_over_a_run(small_bridge):
    eng = Engine(config=_config(duration=60.0), bridge=small_bridge)
    eng.run()
    assert eng.metrics.collision_count == 0


def test_tasks_get_completed(small_bridge):
    completed = []
    eng = Engine(
        config=_config(duration=60.0, arrival=1.0),
        bridge=small_bridge,
        on_task_event=lambda e: completed.append(e) if e.get("event") == "task_completed" else None,
    )
    eng.run()
    assert len(completed) > 0


def test_same_seed_reproducible(small_bridge):
    positions_a = []
    positions_b = []
    eng_a = Engine(config=_config(seed=7, duration=20.0), bridge=small_bridge, on_robot_sample=lambda r: positions_a.append((r["robot_id"], r["x"], r["y"])))
    eng_a.run()
    eng_b = Engine(config=_config(seed=7, duration=20.0), bridge=small_bridge, on_robot_sample=lambda r: positions_b.append((r["robot_id"], r["x"], r["y"])))
    eng_b.run()
    assert positions_a == positions_b


def test_edge_node_zone_samples_emitted_and_shared_across_robots(small_bridge):
    """Occupancy aggregation should fire alongside robot_state_ts sampling
    and, with several robots, occasionally report >1 robot on the same
    edge/node/zone at once (proof the grouping is real, not per-robot)."""
    edge_samples, node_samples, zone_samples = [], [], []
    eng = Engine(
        config=_config(duration=60.0, arrival=1.0, robots=8),
        bridge=small_bridge,
        on_edge_sample=edge_samples.append,
        on_node_sample=node_samples.append,
        on_zone_sample=zone_samples.append,
    )
    eng.run()
    assert edge_samples or node_samples or zone_samples
    for ev in edge_samples:
        assert ev["occupancy_count"] >= 1
        assert ev["min_speed"] <= ev["avg_speed"] <= ev["max_speed"]
    assert any(ev["occupancy_count"] > 1 for ev in edge_samples + node_samples + zone_samples), (
        "expected at least one edge/node/zone sample with more than one robot present"
    )


def test_congestion_detour_fires_under_dense_load(small_bridge):
    """Regression guard for the ported fleet_sim._attempt_congestion_detour:
    with enough robots forced into a small layout's shared aisles, at
    least one genuine (non-gridlock-causing) detour should fire over a
    long enough run. Detour events are logged as algorithm_events with
    trigger='congestion_detour' (see engine.py::_attempt_congestion_detour).

    robots=30 (not 20): once the `grid`/`zone_based` archetypes were
    fixed to split zones onto opposite walls (see warehouse_layout's
    archetypes/grid.py) the `small` scale class layout gained a wider
    storage field between the two zone bands, so 20 robots no longer
    reliably congests it -- confirmed empirically (0/5/21 detours at
    20/30/40 robots on this exact fixture) rather than just bumped
    blindly."""
    detours = []
    eng = Engine(
        config=_config(seed=5, robots=30, duration=90.0, arrival=2.0),
        bridge=small_bridge,
        on_algorithm_event=lambda e: detours.append(e) if e.get("trigger") == "congestion_detour" else None,
    )
    eng.run()
    assert detours, "expected at least one congestion detour under dense load"
    for ev in detours:
        assert ev["output_state"]["detour_path_len"] > 0
        # DETOUR_MAX_COST_RATIO guard: a committed detour must be a real
        # alternative route, never wildly more expensive than the original.
        from fleetnet_sim.config import timing

        assert ev["output_state"]["detour_cost"] <= ev["output_state"]["original_cost"] * timing.DETOUR_MAX_COST_RATIO


def test_congestion_detour_never_routes_through_static_obstacles(small_bridge):
    """The scratch World a detour is planned on always includes every real
    static obstacle (see _attempt_congestion_detour's `scratch.obstacles =
    set(world.obstacles) | phantom_obstacles`) — a detour must never cut
    through a rack/column/exclusion cell."""
    eng = Engine(config=_config(seed=5, robots=20, duration=90.0, arrival=2.0), bridge=small_bridge)
    for _ in range(900):
        eng.step()
        for agent in eng.agents.values():
            for cell in agent.path:
                assert small_bridge.world.is_free(cell), f"detour/plan routed through blocked cell {cell}"


def test_cbba_announce_livelock_is_caught_not_fatal(small_bridge, monkeypatch):
    """Regression guard for a real, reproduced bug: Fleet_SIH's
    InProcessBus dispatches CBBA bid consensus synchronously/recursively,
    and a rare two-agent bid tie can ping-pong forever (confirmed via
    direct reproduction: robot_4/robot_5 alternating indefinitely on a
    real generated layout — raising the recursion limit to 1_000_000
    just hangs, proving it's a genuine livelock in Fleet_SIH's own
    CBBA/comms interaction, not merely deep recursion). Since we never
    modify Fleet_SIH, engine.py::_safe_announce is our circuit breaker:
    catch the RecursionError, fail only that task, keep simulating."""
    eng = Engine(config=_config(duration=5.0, arrival=5.0), bridge=small_bridge)
    # Every agent's announce_task explodes, regardless of which one CBBA's
    # own rng-driven announcer selection picks -- deterministic without
    # having to fake out the engine's shared rng (also used for Poisson
    # task-arrival thinning, which a partial rng stub would break).
    for agent in eng.agents.values():
        monkeypatch.setattr(agent.cbba, "announce_task", lambda task: (_ for _ in ()).throw(RecursionError()))

    events = []
    eng.on_algorithm_event = events.append

    # Should not raise, even though every announce attempt explodes.
    for _ in range(20):
        eng.step()

    livelock_events = [e for e in events if e.get("trigger") == "announce_livelock"]
    assert livelock_events, "expected at least one caught announce_livelock event"
    assert livelock_events[0]["success"] is False
    failed_records = [r for r in eng.task_records.values() if r.final_status == "failed_cbba_livelock"]
    assert failed_records


def test_no_motion_onto_blocked_cell(small_bridge):
    eng = Engine(config=_config(duration=40.0, arrival=1.0), bridge=small_bridge)
    for _ in range(400):
        eng.step()
        for agent in eng.agents.values():
            cell = agent.current_cell()
            assert small_bridge.world.is_free(cell), f"{agent.robot_id} standing on blocked cell {cell}"
