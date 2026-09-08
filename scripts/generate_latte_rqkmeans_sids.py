#!/usr/bin/env python3
"""Export Latte-compatible RQ-KMeans or OPQ semantic IDs for CleanGR data.

This deliberately mirrors the public Latte tokenizer rather than DiffGRM's
older ``rq_kmeans`` branch:

* Sentence-T5-base embeddings are whitened with PCA-192.
* Faiss learns either three 256-way residual codebooks (RQ-KMeans) or three
  equally sized product subspaces after an OPQ rotation (OPQ/PQ) on items
  visible in the training split.
* Latte/PSID's embedding-space semantic mapping (ESM) repairs duplicate tuples
  without appending an identity digit.

The output is a JSON ``item -> [c0,c1,c2]`` mapping accepted by
``sid_override_path``.  Item and embedding row order follow ``item_vocab.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
import sys

import faiss
import numpy as np
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.models.DIFF_GRM.collision import (
    collision_stats,
    repair_code_digit,
    repair_product_codes,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits-dir", default="splits_l20")
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--pca-dim", type=int, default=192)
    parser.add_argument(
        "--quantizer", choices=("rqkmeans", "opq", "rq2opq2"), default="rqkmeans"
    )
    parser.add_argument("--n-codebooks", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--faiss-threads", type=int, default=32)
    parser.add_argument("--esm-neighbors", type=int, default=5)
    parser.add_argument(
        "--repair-strategy",
        choices=("esm", "hungarian", "none"),
        default="esm",
        help="Collision repair applied to the raw quantizer tuples.",
    )
    parser.add_argument(
        "--raw-codes-output",
        type=Path,
        default=None,
        help="Optional .npy path for auditing the pre-repair tuples.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_items(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "item_id" not in reader.fieldnames:
            raise ValueError(f"item vocabulary lacks item_id: {path}")
        items = [row["item_id"].strip() for row in reader]
    if not items or len(set(items)) != len(items):
        raise ValueError("item vocabulary is empty or contains duplicates")
    return items


def training_mask(path: Path, item_to_row: dict[str, int]) -> np.ndarray:
    used = np.zeros(len(item_to_row), dtype=bool)
    unknown = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sequence = row.get("history_item_ids", row.get("inter_history", []))
            sequence = [str(item) for item in sequence] + [str(row["target_id"])]
            for item in sequence:
                index = item_to_row.get(item)
                if index is None:
                    unknown.add(item)
                else:
                    used[index] = True
    if unknown:
        raise ValueError(f"training split contains unknown items: {sorted(unknown)[:5]}")
    return used


def unpack_codes(index, vectors: np.ndarray, n_codebooks: int, n_bits: int) -> np.ndarray:
    packed = index.rq.compute_codes(vectors)
    codes = np.empty((len(vectors), n_codebooks), dtype=np.int64)
    for row, packed_row in enumerate(packed):
        reader = faiss.BitstringReader(faiss.swig_ptr(packed_row), packed.shape[1])
        for digit in range(n_codebooks):
            codes[row, digit] = reader.read(n_bits)
    return codes


def unpack_opq_codes(index, n_items: int, n_codebooks: int, n_bits: int):
    """Read IVF-PQ codes in original add-id order and return PQ centroids."""
    if isinstance(index, faiss.IndexPreTransform):
        ivf_index = faiss.downcast_index(index.index)
    else:
        ivf_index = faiss.downcast_index(index)
    ivf = faiss.extract_index_ivf(ivf_index)
    if ivf.nlist != 1:
        raise ValueError(f"Latte OPQ expects IVF1, got IVF{ivf.nlist}")
    invlists = ivf.invlists
    list_size = invlists.list_size(0)
    packed = faiss.rev_swig_ptr(
        invlists.get_codes(0), list_size * invlists.code_size
    ).reshape(list_size, invlists.code_size)
    ids = faiss.rev_swig_ptr(invlists.get_ids(0), list_size).copy()
    codes = np.full((n_items, n_codebooks), -1, dtype=np.int64)
    for packed_row, item_row in zip(packed, ids):
        reader = faiss.BitstringReader(
            faiss.swig_ptr(packed_row), packed.shape[1]
        )
        codes[int(item_row)] = [reader.read(n_bits) for _ in range(n_codebooks)]
    if np.any(codes < 0):
        raise RuntimeError("OPQ/PQ did not return a code for every catalog item")
    pq = ivf_index.pq
    centroids = faiss.vector_to_array(pq.centroids).reshape(
        pq.M, pq.ksub, pq.dsub
    )
    return codes, centroids


def opq_metric_geometry(index, vectors: np.ndarray, n_codebooks: int):
    """Return the exact rotated PQ subvectors used for assignment costs."""
    transformed = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    if isinstance(index, faiss.IndexPreTransform):
        for transform_index in range(index.chain.size()):
            transform = faiss.downcast_VectorTransform(
                index.chain.at(transform_index)
            )
            transformed = transform.apply_py(transformed)
        ivf_index = faiss.downcast_index(index.index)
    else:
        ivf_index = faiss.downcast_index(index)
    if bool(getattr(ivf_index, "by_residual", False)):
        coarse = faiss.downcast_index(ivf_index.quantizer).reconstruct(0)
        transformed = transformed - np.asarray(coarse, dtype=np.float32)[None]
    pq = ivf_index.pq
    if pq.M != n_codebooks:
        raise ValueError(f"PQ digits {pq.M} != requested {n_codebooks}")
    return transformed.reshape(len(vectors), pq.M, pq.dsub)


def rq_stage_inputs(
    vectors: np.ndarray, codes: np.ndarray, centroids: np.ndarray
) -> np.ndarray:
    """Reconstruct the residual presented to every RQ stage."""
    residual = np.ascontiguousarray(vectors.astype(np.float32, copy=True))
    stages = []
    for digit in range(codes.shape[1]):
        stages.append(residual.copy())
        residual -= centroids[digit, codes[:, digit]]
    return np.stack(stages, axis=1)


def train_rq2opq2(
    vectors: np.ndarray,
    train_mask: np.ndarray,
    codebook_size: int,
    faiss_threads: int,
):
    """Two residual KMeans stages followed by two OPQ residual subspaces."""
    if vectors.shape[1] % 2:
        raise ValueError("RQ2+OPQ2 requires an even PCA dimension")
    faiss.omp_set_num_threads(faiss_threads)
    residual = np.ascontiguousarray(vectors.astype(np.float32, copy=True))
    rq_codes = np.zeros((len(vectors), 2), dtype=np.int64)
    rq_centroids = []
    for stage in range(2):
        kmeans = faiss.Kmeans(
            d=vectors.shape[1],
            k=codebook_size,
            niter=25,
            verbose=False,
            seed=2026 + stage,
            max_points_per_centroid=10000,
        )
        kmeans.train(np.ascontiguousarray(residual[train_mask]))
        centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(
            codebook_size, vectors.shape[1]
        )
        search = faiss.IndexFlatL2(vectors.shape[1])
        search.add(centroids)
        _, assigned = search.search(residual, 1)
        rq_codes[:, stage] = assigned[:, 0]
        residual -= centroids[assigned[:, 0]]
        rq_centroids.append(centroids)

    n_bits = int(np.log2(codebook_size))
    index = faiss.index_factory(
        vectors.shape[1], f"OPQ2,IVF1,PQ2x{n_bits}", faiss.METRIC_INNER_PRODUCT
    )
    index.train(np.ascontiguousarray(residual[train_mask]))
    index.add(np.ascontiguousarray(residual))
    opq_codes, opq_centroids = unpack_opq_codes(index, len(vectors), 2, n_bits)
    opq_subvectors = opq_metric_geometry(index, residual, 2)
    codes = np.concatenate([rq_codes, opq_codes], axis=1)
    return codes, opq_centroids, opq_subvectors, rq_centroids


def esm_repair(
    items: list[str],
    codes: np.ndarray,
    centroids: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, dict]:
    """Mirror Latte's deterministic top-neighbor Cartesian ESM repair."""
    owners: dict[tuple[int, ...], list[int]] = {}
    for row, code in enumerate(codes):
        owners.setdefault(tuple(int(value) for value in code), []).append(row)
    conflicts = {code: rows for code, rows in owners.items() if len(rows) > 1}
    repaired = codes.copy()
    used = set(owners)
    resolved = 0
    unresolved = []

    for original_code, rows in conflicts.items():
        original_reconstruction = sum(
            (centroids[digit, token] for digit, token in enumerate(original_code)),
            start=np.zeros(centroids.shape[-1], dtype=np.float32),
        )
        neighbor_tokens = []
        for digit, token in enumerate(original_code):
            diff = centroids[digit] - centroids[digit, token][None, :]
            distances = np.sum(diff * diff, axis=1)
            neighbor_tokens.append(np.argsort(distances)[:neighbors].tolist())

        for row in rows[1:]:
            best_code = None
            best_distance = float("inf")
            for candidate in product(*neighbor_tokens):
                candidate = tuple(int(value) for value in candidate)
                if candidate in used:
                    continue
                candidate_reconstruction = sum(
                    (centroids[digit, token] for digit, token in enumerate(candidate)),
                    start=np.zeros(centroids.shape[-1], dtype=np.float32),
                )
                distance = float(
                    np.sum((original_reconstruction - candidate_reconstruction) ** 2)
                )
                if distance < best_distance:
                    best_code = candidate
                    best_distance = distance
            if best_code is None:
                unresolved.append(items[row])
                continue
            repaired[row] = best_code
            used.add(best_code)
            resolved += 1

    diagnostics = {
        "raw_unique_tuples": int(len(owners)),
        "raw_collision_groups": int(len(conflicts)),
        "raw_colliding_extra_items": int(sum(len(rows) - 1 for rows in conflicts.values())),
        "esm_resolved_items": int(resolved),
        "esm_unresolved_items": unresolved,
        "final_unique_tuples": int(len(np.unique(repaired, axis=0))),
    }
    return repaired, diagnostics


def esm_repair_opq(
    items: list[str],
    codes: np.ndarray,
    centroids: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, dict]:
    """Mirror PSID/Latte's centroid-similarity ESM repair for OPQ/PQ."""
    owners: dict[tuple[int, ...], list[int]] = {}
    for row, code in enumerate(codes):
        owners.setdefault(tuple(int(value) for value in code), []).append(row)
    conflicts = {code: rows for code, rows in owners.items() if len(rows) > 1}
    repaired = codes.copy()
    used = set(owners)
    resolved = 0
    unresolved = []

    for original_code, rows in conflicts.items():
        neighbor_tokens = []
        for digit, token in enumerate(original_code):
            similarities = centroids[digit] @ centroids[digit, token]
            neighbor_tokens.append(np.argsort(-similarities)[:neighbors].tolist())
        for row in rows[1:]:
            best_code = None
            best_distance = float("inf")
            for candidate in product(*neighbor_tokens):
                candidate = tuple(int(value) for value in candidate)
                if candidate in used:
                    continue
                distance = -sum(
                    float(
                        centroids[digit, original_code[digit]]
                        @ centroids[digit, candidate[digit]]
                    )
                    for digit in range(len(original_code))
                )
                if distance < best_distance:
                    best_code = candidate
                    best_distance = distance
            if best_code is None:
                unresolved.append(items[row])
                continue
            repaired[row] = best_code
            used.add(best_code)
            resolved += 1

    diagnostics = {
        "raw_unique_tuples": int(len(owners)),
        "raw_collision_groups": int(len(conflicts)),
        "raw_colliding_extra_items": int(sum(len(rows) - 1 for rows in conflicts.values())),
        "esm_resolved_items": int(resolved),
        "esm_unresolved_items": unresolved,
        "final_unique_tuples": int(len(np.unique(repaired, axis=0))),
    }
    return repaired, diagnostics


def main() -> None:
    args = arguments()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"output already exists (pass --force): {args.output}")
    if args.codebook_size <= 0 or args.codebook_size & (args.codebook_size - 1):
        raise ValueError("codebook size must be a positive power of two")

    items = read_items(args.data_dir / "item_vocab.csv")
    embeddings = np.fromfile(args.embedding_path, dtype=np.float32)
    expected = len(items) * args.embedding_dim
    if embeddings.size != expected:
        raise ValueError(
            f"embedding size mismatch: {embeddings.size} values, expected {expected}"
        )
    embeddings = embeddings.reshape(len(items), args.embedding_dim)
    mask = training_mask(
        args.data_dir / args.splits_dir / "train.jsonl",
        {item: row for row, item in enumerate(items)},
    )
    print(f"items={len(items)} training_items={int(mask.sum())}", flush=True)

    # sklearn's default full PCA path and whitening match Latte's released code.
    print(f"fitting PCA-{args.pca_dim} with whitening", flush=True)
    reduced = PCA(n_components=args.pca_dim, whiten=True).fit_transform(embeddings)
    reduced = np.ascontiguousarray(reduced.astype(np.float32, copy=False))

    faiss.omp_set_num_threads(args.faiss_threads)
    n_bits = int(np.log2(args.codebook_size))
    repair_geometry = None
    repair_centroids = None
    if args.quantizer == "rqkmeans":
        index = faiss.IndexResidualQuantizer(
            args.pca_dim,
            args.n_codebooks,
            n_bits,
            faiss.METRIC_INNER_PRODUCT,
        )
        print(
            f"training Faiss residual quantizer {args.n_codebooks}x{args.codebook_size}",
            flush=True,
        )
        index.train(np.ascontiguousarray(reduced[mask]))
        index.add(reduced)
        centroids = faiss.vector_to_array(index.rq.codebooks).reshape(
            args.n_codebooks, args.codebook_size, args.pca_dim
        )
        codes = unpack_codes(index, reduced, args.n_codebooks, n_bits)
        repair_geometry = rq_stage_inputs(reduced, codes, centroids)
        repair_centroids = centroids
        faiss_factory = "IndexResidualQuantizer"
    elif args.quantizer == "opq":
        faiss_factory = (
            f"OPQ{args.n_codebooks},IVF1,"
            f"PQ{args.n_codebooks}x{n_bits}"
        )
        print(f"training Faiss {faiss_factory}", flush=True)
        index = faiss.index_factory(
            args.pca_dim, faiss_factory, faiss.METRIC_INNER_PRODUCT
        )
        index.train(np.ascontiguousarray(reduced[mask]))
        index.add(reduced)
        codes, centroids = unpack_opq_codes(
            index, len(items), args.n_codebooks, n_bits
        )
        repair_geometry = opq_metric_geometry(
            index, reduced, args.n_codebooks
        )
        repair_centroids = centroids
    else:
        if args.n_codebooks != 4:
            raise ValueError("rq2opq2 requires --n-codebooks 4")
        print("training aligned RQ2+OPQ2 hybrid quantizer", flush=True)
        (
            codes,
            repair_centroids,
            repair_geometry,
            _,
        ) = train_rq2opq2(
            reduced,
            mask,
            args.codebook_size,
            args.faiss_threads,
        )
        faiss_factory = "RQ2+OPQ2"

    raw_codes = np.ascontiguousarray(codes.astype(np.int64, copy=False))
    raw_hash = hashlib.sha256(raw_codes.astype("<i8", copy=False).tobytes()).hexdigest()
    raw_statistics = collision_stats(raw_codes)
    if args.raw_codes_output is not None:
        args.raw_codes_output.parent.mkdir(parents=True, exist_ok=True)
        if args.raw_codes_output.exists() and not args.force:
            existing = np.load(args.raw_codes_output)
            if not np.array_equal(existing, raw_codes):
                raise FileExistsError(
                    f"raw code artifact differs (pass --force): {args.raw_codes_output}"
                )
        else:
            np.save(args.raw_codes_output, raw_codes)

    if args.repair_strategy == "esm":
        if args.quantizer == "rqkmeans":
            repaired, repair_report = esm_repair(
                items, raw_codes, repair_centroids, args.esm_neighbors
            )
        elif args.quantizer == "opq":
            repaired, repair_report = esm_repair_opq(
                items, raw_codes, repair_centroids, args.esm_neighbors
            )
        else:
            raise ValueError(
                "ESM is not geometry-consistent across mixed RQ/OPQ codebooks; "
                "use --repair-strategy hungarian for rq2opq2"
            )
        if repair_report["esm_unresolved_items"]:
            raise RuntimeError(
                "Latte ESM failed to repair every collision: "
                f'{repair_report["esm_unresolved_items"][:5]}'
            )
    elif args.repair_strategy == "hungarian":
        if args.quantizer != "rq2opq2":
            repaired, repair_report = repair_product_codes(
                raw_codes,
                pq_subvectors=repair_geometry,
                centroids=repair_centroids,
                repair_digit="auto",
            )
        else:
            solutions = []
            for local_digit in range(2):
                global_digit = 2 + local_digit
                candidate, report = repair_code_digit(
                    raw_codes,
                    digit_vectors=repair_geometry[:, local_digit],
                    digit_centroids=repair_centroids[local_digit],
                    repair_digit=global_digit,
                )
                solutions.append((candidate, report, local_digit))
            repaired, selected, local_digit = min(
                solutions,
                key=lambda value: (
                    value[1]["selected"]["distortion_increase"],
                    value[1]["selected"]["changed_items"],
                ),
            )
            repair_report = {
                "strategy": "hungarian_hybrid_auto_opq_digit",
                "selected_local_opq_digit": int(local_digit),
                "selected": selected,
                "candidate_summaries": [report for _, report, _ in solutions],
            }
    else:
        repaired = raw_codes.copy()
        repair_report = {
            "strategy": "none",
            "before": raw_statistics,
            "after": raw_statistics,
        }

    final_statistics = collision_stats(repaired)
    if final_statistics["duplicate_excess"] != 0:
        raise RuntimeError(
            f"{args.repair_strategy} left "
            f'{final_statistics["duplicate_excess"]} duplicate tuples'
        )
    if final_statistics["num_unique_codes"] != len(items):
        raise RuntimeError("exported semantic IDs are not collision free")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        item: [int(value) for value in repaired[row]]
        for row, item in enumerate(items)
    }
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    diagnostics = {
            "data_dir": str(args.data_dir.resolve()),
            "embedding_path": str(args.embedding_path.resolve()),
            "embedding_dim": args.embedding_dim,
            "pca_dim": args.pca_dim,
            "pca_whiten": True,
            "quantizer": args.quantizer,
            "n_codebooks": args.n_codebooks,
            "codebook_size": args.codebook_size,
            "faiss_factory": faiss_factory,
            "faiss_metric": "METRIC_INNER_PRODUCT",
            "training_items": int(mask.sum()),
            "catalog_items": len(items),
            "esm_neighbors": args.esm_neighbors,
            "repair_strategy": args.repair_strategy,
            "raw_codes_sha256": raw_hash,
            "raw": raw_statistics,
            "final": final_statistics,
            "repair": repair_report,
        }
    diagnostics_path = args.output.with_suffix(args.output.suffix + ".diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
