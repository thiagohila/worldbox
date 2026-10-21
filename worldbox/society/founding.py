"""Planting a tribe from outside the simulation's own rules.

The simulation never calls this: tribes there form from people meeting. It
is for tools that place a civilisation on purpose -- the live map's "found"
button, or a program driving the engine. Callers hold the runner's lock.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..agents.agent import Agent, create_agent
from ..agents.needs import Needs
from .groups import Group
from .settlements import Settlement


def settle_people(
    engine: Any,
    group: Group,
    x: int,
    y: int,
    count: int,
    family: str,
    houses: Optional[Sequence[str]] = None,
) -> List[Agent]:
    """Add ``count`` young adults to ``group`` around ``(x, y)``.

    ``houses`` cycles over the newcomers' LLM houses; without it each takes
    ``group.ai_house`` or, failing that, the house its id falls to.
    """
    from ..ai.client import house_for_index

    world = engine.world
    rng = engine.rng
    day = engine.clock.day
    dpy = engine.config.simulation.days_per_year
    spots = [
        (tx, ty) for tx, ty in world.tiles_within(x, y, 2) if world.is_passable(tx, ty)
    ] or [(x, y)]
    spawned: List[Agent] = []
    for index in range(count):
        tx, ty = spots[index % len(spots)]
        agent = create_agent(
            agent_id=engine._next_agent_id,
            x=tx,
            y=ty,
            age_days=int(rng.uniform(16, 32) * dpy),
            rng=rng,
            config=engine.config.agents,
            days_per_year=dpy,
            day=day,
            group_id=group.id,
            inherited_family=family,
            needs=Needs(hunger=12.0, energy=95.0, health=100.0),
        )
        if houses:
            agent.ai_house = houses[index % len(houses)]
        else:
            agent.ai_house = group.ai_house or house_for_index(agent.id)
        engine._next_agent_id += 1
        engine.agents.append(agent)
        spawned.append(agent)
    group.member_ids.extend(agent.id for agent in spawned)
    return spawned


def found_tribe(
    engine: Any,
    x: int,
    y: int,
    count: int,
    name: str = "",
    family: str = "",
    house: str = "",
    houses: Optional[Sequence[str]] = None,
    rename: Optional[Callable[[Group], str]] = None,
) -> Tuple[Group, Optional[Settlement], List[Agent]]:
    """A new tribe of ``count`` people at a walkable ``(x, y)``, with a camp.

    The first person is its chieftain. The land around is cleared and stocked
    so the newcomers do not starve in their first week. ``rename`` gets the
    new group (with its generated name) and returns the name to use. Records
    no event: the caller knows how to tell the story.
    """
    world = engine.world
    day = engine.clock.day
    group = engine.groups.create(day, engine.rng)
    if rename is not None:
        group.name = rename(group)
    elif name:
        group.name = name
    group.ai_house = house
    group.centre = (x, y)

    spawned = settle_people(
        engine, group, x, y, count, family or group.name.split()[0], houses
    )
    group.chieftain_id = spawned[0].id
    spawned[0].role = "warrior"

    ceiling = engine.config.settlements.max_fertility + 1.0
    for tx, ty in world.tiles_within(x, y, 3):
        world.resources.cultivate(tx, ty, 0.5, ceiling)
        cap = world.resources.effective_capacity(tx, ty)
        if cap > 0:
            world.resources.food[ty][tx] = cap

    settlement = engine.settlements.found_at(group, x, y, day, engine.rng, food_store=50.0)
    return group, settlement, spawned
