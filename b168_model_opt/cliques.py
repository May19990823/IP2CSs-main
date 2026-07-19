from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from numbers import Integral

from b168_model_opt.types import CliqueCover


_MAX_CLIQUE_CANDIDATES = 50_000


def _clique_edges(clique: tuple[int, ...]) -> frozenset[tuple[int, int]]:
    return frozenset(combinations(clique, 2))


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _canonical_edges(
    node_count: int, edges: Iterable[tuple[int, int]]
) -> frozenset[tuple[int, int]]:
    try:
        iterator = iter(edges)
    except TypeError as exc:
        raise ValueError("edges must be an iterable of two-item integer edges") from exc

    canonical: set[tuple[int, int]] = set()
    for edge in iterator:
        try:
            pair = tuple(edge)
        except TypeError as exc:
            raise ValueError("edges must contain two-item integer edges") from exc
        if len(pair) != 2:
            raise ValueError("edges must contain two-item integer edges")
        left, right = pair
        if (
            not _is_integer(left)
            or not _is_integer(right)
            or not 0 <= left < node_count
            or not 0 <= right < node_count
            or left == right
        ):
            raise ValueError(
                "edges must contain distinct integer endpoints in node_count range"
            )
        canonical.add((min(left, right), max(left, right)))
    return frozenset(canonical)


def _maximal_cliques(
    adjacency: tuple[frozenset[int], ...],
) -> tuple[tuple[int, ...], ...]:
    found: list[tuple[int, ...]] = []
    exhausted = False

    def visit(
        clique: tuple[int, ...],
        possible: set[int],
        excluded: set[int],
    ) -> None:
        nonlocal exhausted
        if exhausted:
            return
        if not possible and not excluded:
            found.append(tuple(sorted(clique)))
            exhausted = len(found) >= _MAX_CLIQUE_CANDIDATES
            return

        pivot = min(
            possible | excluded,
            key=lambda node: (-len(possible & adjacency[node]), node),
        )
        for node in sorted(possible - adjacency[pivot]):
            if exhausted:
                return
            neighbors = adjacency[node]
            visit(
                clique + (node,), possible & neighbors, excluded & neighbors
            )
            possible.remove(node)
            excluded.add(node)

    visit((), set(range(len(adjacency))), set())
    return tuple(sorted(clique for clique in found if len(clique) >= 3))


def greedy_edge_clique_cover(
    node_count: int,
    edges: Iterable[tuple[int, int]],
) -> CliqueCover:
    if not _is_integer(node_count) or node_count < 0:
        raise ValueError("node_count must be a non-boolean nonnegative integer")

    canonical_edges = _canonical_edges(node_count, edges)
    adjacency = [set() for _ in range(node_count)]
    for left, right in canonical_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    candidates = _maximal_cliques(tuple(frozenset(nodes) for nodes in adjacency))
    candidate_edges = {clique: _clique_edges(clique) for clique in candidates}
    uncovered = set(canonical_edges)
    selected: list[tuple[int, ...]] = []

    for clique in sorted(
        candidates,
        key=lambda candidate: (-len(candidate_edges[candidate]), candidate),
    ):
        if len(candidate_edges[clique] & uncovered) >= 2:
            selected.append(clique)
            uncovered.difference_update(candidate_edges[clique])

    rebuilt = set(uncovered)
    for clique in selected:
        rebuilt.update(candidate_edges[clique])
    if rebuilt != canonical_edges:
        raise RuntimeError("clique cover does not reproduce the conflict graph")

    return CliqueCover(
        cliques=tuple(selected),
        residual_edges=tuple(sorted(uncovered)),
        covered_edges=frozenset(rebuilt),
    )
