# Matched SID-semantics control

This control keeps the exact collision-free OPQ4 catalog but randomly permutes
the mapping from item IDs to SID paths (seed 2026). Consequently the path
multiset, code marginals, vocabulary, number of digits, catalog size, and trie
shape are identical to the semantic OPQ4 condition. Only item/path semantic
alignment is destroyed.

The run trains both standalone AR and the one-pass pairwise drafter from random
initialization with the same Science23 L20 protocol and capacity as the OPQ4
reference. It is queued on GPU 3 after the current proposal-aware verifier arm.
