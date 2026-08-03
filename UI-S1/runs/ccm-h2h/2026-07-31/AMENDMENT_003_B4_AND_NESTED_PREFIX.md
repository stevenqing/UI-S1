# Amendment 003: B4 Scope and Deterministic Candidate Prefixes

Date: 2026-07-31

Status: frozen before H1 candidate generation.

## Deterministic N prefixes

Official MVP ranks unique attention regions once by coverage and processes them in order. Greedy generation is deterministic. Therefore the N=2 and N=4 candidate sets are exact ordered prefixes of an N=10 run on the same row: full image plus the first 1 or 3 subimages. H1 generates the deterministic N=10 set once and derives N=2/N=4 by prefix, with candidate-hash assertions. This is algorithmically identical to separate `max_inferences=1/3/9` runs and removes duplicated forwards. The prior five-candidate W3 anchor remains excluded because it was generated in a separate run and cannot establish prefix identity with the new N=10 artifact.

## B4 algorithm-level ReGUIDE comparator

No public checkpoint reproduces ReGUIDE RL/view-consistency training in the pinned environment. B4 is explicitly an algorithm-level same-candidate-set comparator, not a full ReGUIDE reproduction.

1. First stage: Scott-bandwidth KDE candidate peak over all N normalized coordinates.
2. RoI membership: retain subimage candidates whose official crop region contains the first-stage peak in original-image coordinates. The full-image candidate is not a second-stage RoI candidate.
3. Second stage: Scott-bandwidth KDE candidate peak over retained candidates.
4. If no retained candidate exists, return the first-stage peak.

B4 uses no additional forwards and receives exactly the same candidate tensor as B3 and M1. This tests the two-stage RoI-to-KDE structure under equal compute; the paper must label it `ReGUIDE algorithm-level` and must not claim reproduction of learned ReGUIDE weights.