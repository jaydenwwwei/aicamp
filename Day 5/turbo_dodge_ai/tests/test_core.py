from __future__ import annotations

import numpy as np
import pytest

from turbo_dodge_ai.core import Obstacle, ObstacleGroup, Simulation


def test_seeded_runs_are_deterministic() -> None:
    first = Simulation(phase="easy", seed=123)
    second = Simulation(phase="easy", seed=123)
    first.reset(seed=123)
    second.reset(seed=123)
    for _ in range(180):
        first.step({"human": 0.15})
        second.step({"human": 0.15})
    assert np.allclose(first.observation("human"), second.observation("human"))
    assert first.elapsed_time == second.elapsed_time
    assert [(group.group_id, len(group.obstacles)) for group in first.obstacle_groups] == [
        (group.group_id, len(group.obstacles)) for group in second.obstacle_groups
    ]


def test_observation_is_19_finite_bounded_features() -> None:
    simulation = Simulation(seed=7)
    simulation.reset(seed=7)
    observation = simulation.observation("human")
    assert observation.shape == (19,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_steering_changes_heading_and_lateral_position() -> None:
    simulation = Simulation(phase="easy", seed=15)
    simulation.reset(seed=15)

    for _ in range(30):
        simulation.step({"human": 1.0})

    driver = simulation.drivers["human"]
    assert driver.applied_steering > 0.0
    assert driver.heading > 0.0
    assert driver.x > 0.0


def test_road_edge_crashes_driver() -> None:
    simulation = Simulation(seed=2)
    simulation.reset(seed=2)
    simulation.drivers["human"].x = simulation.config.road_half_width
    result = simulation.step({"human": 0.0})
    assert not simulation.drivers["human"].alive
    assert simulation.drivers["human"].crash_reason == "road_edge"
    assert result.finished


def test_a_group_is_rewarded_once_after_passing() -> None:
    simulation = Simulation(seed=4)
    simulation.reset(seed=4)
    simulation.obstacle_groups = [
        ObstacleGroup(1, [Obstacle(0.0, -20.0, 1.5, 2.0, "barrier")])
    ]
    simulation.step({"human": 0.0})
    passed = [event for event in simulation.last_result.events if event["type"] == "group_passed"]
    assert len(passed) == 1
    simulation.step({"human": 0.0})
    assert simulation.drivers["human"].passed_groups == 1


def test_car_contact_is_a_draw() -> None:
    simulation = Simulation(phase="multiplayer", seed=9)
    simulation.reset(seed=9, versus=True)
    simulation.drivers["human"].x = 0.0
    simulation.drivers["ai"].x = 0.0
    result = simulation.step({"human": 0.0, "ai": 0.0})
    assert result.match_result == "draw"
    assert not simulation.drivers["human"].alive
    assert not simulation.drivers["ai"].alive


def test_gym_wrapper_has_expected_spaces() -> None:
    pytest.importorskip("gymnasium")
    from turbo_dodge_ai.environment import TurboDodgeEnv

    environment = TurboDodgeEnv(phase="easy", seed=11)
    observation, info = environment.reset(seed=11)
    assert environment.observation_space.contains(observation)
    next_observation, reward, terminated, truncated, info = environment.step(np.asarray([0.0], dtype=np.float32))
    assert environment.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    environment.close()


def test_unseeded_environment_resets_vary_the_episode_seed() -> None:
    pytest.importorskip("gymnasium")
    from turbo_dodge_ai.environment import TurboDodgeEnv

    environment = TurboDodgeEnv(phase="easy")
    environment.reset(seed=21)
    seeded_episode = environment.simulation.seed_value
    environment.reset()
    random_episode = environment.simulation.seed_value
    assert seeded_episode != random_episode
    environment.close()


def test_multiplayer_time_limit_is_a_gymnasium_truncation() -> None:
    pytest.importorskip("gymnasium")
    from turbo_dodge_ai.environment import TurboDodgeEnv

    environment = TurboDodgeEnv(phase="multiplayer", max_episode_seconds=1.0 / 30.0, seed=31)
    environment.reset(seed=31)
    _, _, terminated, truncated, info = environment.step(np.asarray([0.0], dtype=np.float32))
    assert not terminated
    assert truncated
    assert info["match_result"] == "draw"
    environment.close()
