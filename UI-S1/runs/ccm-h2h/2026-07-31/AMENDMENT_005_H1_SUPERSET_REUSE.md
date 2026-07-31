# Amendment 005: H1 Deterministic Superset Reuse

Date: 2026-07-31

Status: frozen before H1 candidate generation.

The official proposal function sorts the same unique candidate-region list by coverage and truncates it only after sorting. Therefore requesting 18 subimages yields exact ordered prefixes for requests of 9, 3, and 1 subimages. Greedy prediction for a region is independent of later regions.

H1 runs the official generator once per row for full image plus 18 subimages. It derives:

- main N=2 from candidates `[0:2]`;
- main N=4 from candidates `[0:4]`;
- main N=10 from candidates `[0:10]`;
- MDE seeded N=10 sets from candidate zero plus nine seeded selections among candidates `[1:19]`.

The merger asserts stage order, unique regions, candidate count 19, and deterministic prefix hashes. No target field enters prefix or seeded selection. This removes a duplicate full+9 run and does not change any main or MDE candidate set defined in Amendments 002-003.