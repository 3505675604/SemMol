"""Deterministic scaffold-group splitting with safe serialization."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SPLIT_SCHEMA = "semmol.scaffold_split.v1"
SPLIT_NAMES = ("train", "valid", "test")


def _stable_hash(seed: int, value: Any) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _scaffold_hash(scaffold: str) -> str:
    return hashlib.sha256(scaffold.encode("utf-8")).hexdigest()


def _acyclic_connectivity_skeleton(mol: Chem.Mol) -> str:
    skeleton = Chem.RWMol(mol)
    for atom in skeleton.GetAtoms():
        atom.SetAtomicNum(6)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for bond in skeleton.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(
        skeleton.GetMol(), canonical=True, isomericSmiles=False, allHsExplicit=True
    )


def generate_scaffold(
    smiles: Any, include_chirality: bool = False
) -> str | None:
    """Generate a non-empty structural group key, or ``None`` if invalid.

    Cyclic molecules use Bemis-Murcko scaffolds.  Acyclic molecules use a
    canonical element-free, bond-order-free connectivity skeleton so unrelated
    acyclic structures are not collapsed into the traditional empty scaffold.
    """

    if smiles is None:
        return None
    if not isinstance(smiles, str):
        try:
            if bool(np.asarray(smiles != smiles).item()):
                return None
        except (TypeError, ValueError):
            return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    murcko = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )
    if murcko:
        scaffold_mol = Chem.MolFromSmiles(murcko)
        if scaffold_mol is None:
            return None
        return "murcko:" + Chem.MolToSmiles(
            scaffold_mol,
            canonical=True,
            isomericSmiles=include_chirality,
        )
    skeleton = _acyclic_connectivity_skeleton(mol)
    return f"acyclic:{skeleton}" if skeleton else None


def _validate_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != 3:
        raise ValueError("exactly three fractions are required")
    result = tuple(float(value) for value in fractions)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"fractions must be finite and non-negative: {result}")
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split fractions must sum to 1.0, got {sum(result)}")
    return result  # type: ignore[return-value]


def _integer_split_targets(
    n_items: int, fractions: Sequence[float]
) -> tuple[int, int, int]:
    raw = [fraction * n_items for fraction in fractions]
    targets = [int(math.floor(value)) for value in raw]
    remainder = n_items - sum(targets)
    remainder_order = sorted(
        range(3), key=lambda index: (-(raw[index] - targets[index]), index)
    )
    for index in remainder_order[:remainder]:
        targets[index] += 1
    return targets[0], targets[1], targets[2]


def _count_objective(
    counts: Sequence[int], targets: Sequence[int]
) -> tuple[int, int, float]:
    deviations = [abs(count - target) for count, target in zip(counts, targets)]
    normalized_squared = sum(
        ((count - target) ** 2) / max(target, 1)
        for count, target in zip(counts, targets)
    )
    return sum(deviations), max(deviations), normalized_squared


def _optimistic_state_score(
    state: tuple[int, int],
    *,
    processed_count: int,
    remaining_count: int,
    targets: Sequence[int],
    seed: int,
) -> tuple[int, int, int, int]:
    valid_count, test_count = state
    current = [processed_count - valid_count - test_count, valid_count, test_count]
    lower_bound = 0
    for count, target in zip(current, targets):
        if target < count:
            lower_bound += count - target
        elif target > count + remaining_count:
            lower_bound += target - (count + remaining_count)
    valid_test_distance = (
        abs(valid_count - targets[1]) + abs(test_count - targets[2])
    )
    return (
        lower_bound,
        valid_test_distance,
        _stable_hash(seed, state),
        valid_count + test_count,
    )


def _pack_scaffold_groups(
    scaffold_groups: Mapping[str, Sequence[int]],
    fractions: Sequence[float],
    seed: int,
    dp_state_limit: int,
) -> tuple[dict[str, list[int]], dict[str, str], dict[str, Any]]:
    if dp_state_limit < 3:
        raise ValueError("dp_state_limit must be at least 3")
    n_valid = sum(len(indices) for indices in scaffold_groups.values())
    targets = _integer_split_targets(n_valid, fractions)
    ordered_groups = sorted(
        scaffold_groups.items(),
        key=lambda item: (
            -len(item[1]),
            _stable_hash(seed, item[0]),
            item[0],
        ),
    )
    dp_group_count = next(
        (
            index
            for index, (_scaffold, indices) in enumerate(ordered_groups)
            if len(indices) == 1
        ),
        len(ordered_groups),
    )
    singleton_count = len(ordered_groups) - dp_group_count
    # State is (valid_count, test_count); train is the implicit remainder.
    # Each state holds a shared persistent chain of non-train choices, avoiding
    # a full assignment tuple per state and releasing chains discarded by pruning.
    states: dict[tuple[int, int], Any] = {(0, 0): None}
    processed_count = 0
    peak_states = 1
    peak_candidate_states = 1
    pruning_events = 0
    discarded_state_count = 0
    pruned = False
    for group_index, (_scaffold, indices) in enumerate(
        ordered_groups[:dp_group_count]
    ):
        group_size = len(indices)
        next_states = dict(states)
        for (valid_count, test_count), previous_node in states.items():
            valid_state = (valid_count + group_size, test_count)
            if valid_state not in next_states:
                next_states[valid_state] = (previous_node, group_index, 1)
            test_state = (valid_count, test_count + group_size)
            if test_state not in next_states:
                next_states[test_state] = (previous_node, group_index, 2)
        processed_count += group_size
        remaining_count = n_valid - processed_count
        peak_candidate_states = max(peak_candidate_states, len(next_states))
        if len(next_states) > dp_state_limit:
            pruned = True
            pruning_events += 1
            discarded_state_count += len(next_states) - dp_state_limit
            retained = heapq.nsmallest(
                dp_state_limit,
                next_states,
                key=lambda state: _optimistic_state_score(
                    state,
                    processed_count=processed_count,
                    remaining_count=remaining_count,
                    targets=targets,
                    seed=seed + group_index,
                ),
            )
            states = {state: next_states[state] for state in retained}
        else:
            states = next_states
        peak_states = max(peak_states, len(states))

    def exact_singleton_completion(
        state: tuple[int, int],
    ) -> tuple[int, int] | None:
        valid_needed = targets[1] - state[0]
        test_needed = targets[2] - state[1]
        if (
            valid_needed >= 0
            and test_needed >= 0
            and valid_needed + test_needed <= singleton_count
        ):
            return valid_needed, test_needed
        return None

    exact_states = [
        (state, completion)
        for state in states
        if (completion := exact_singleton_completion(state)) is not None
    ]
    if exact_states:
        best_state, singleton_completion = min(
            exact_states,
            key=lambda item: (
                _stable_hash(seed, (item[0], item[1])),
                item,
            ),
        )
    else:
        def projected_completions(
            state: tuple[int, int],
        ) -> set[tuple[int, int]]:
            if singleton_count == 0:
                return {(0, 0)}
            valid_needed = targets[1] - state[0]
            test_needed = targets[2] - state[1]
            desired_non_train = valid_needed + test_needed
            k_values = {
                0,
                singleton_count,
                min(singleton_count, max(0, desired_non_train)),
                min(singleton_count, max(0, valid_needed)),
                min(singleton_count, max(0, test_needed)),
            }
            completions: set[tuple[int, int]] = set()
            valid_weight = max(targets[1], 1)
            test_weight = max(targets[2], 1)
            for non_train in k_values:
                weighted_valid = (
                    valid_weight * non_train
                    + test_weight * valid_needed
                    - valid_weight * test_needed
                ) / (valid_weight + test_weight)
                candidate_valid = (
                    0,
                    non_train,
                    valid_needed,
                    non_train - test_needed,
                    (valid_needed + non_train - test_needed) / 2,
                    weighted_valid,
                )
                for value in candidate_valid:
                    for valid_singletons in {
                        math.floor(value),
                        math.ceil(value),
                    }:
                        valid_singletons = min(
                            non_train, max(0, valid_singletons)
                        )
                        completions.add(
                            (
                                valid_singletons,
                                non_train - valid_singletons,
                            )
                        )
            return completions

        best_option = None
        for state in states:
            for completion in projected_completions(state):
                valid_count = state[0] + completion[0]
                test_count = state[1] + completion[1]
                counts = (
                    n_valid - valid_count - test_count,
                    valid_count,
                    test_count,
                )
                option = (
                    _count_objective(counts, targets),
                    _stable_hash(seed, (state, completion)),
                    state,
                    completion,
                )
                if best_option is None or option < best_option:
                    best_option = option
        if best_option is None:
            raise RuntimeError("scaffold split completion produced no candidates")
        _objective, _tie_breaker, best_state, singleton_completion = best_option

    assignments = [0] * len(ordered_groups)
    node = states[best_state]
    while node is not None:
        previous_node, group_index, split_index = node
        assignments[group_index] = split_index
        node = previous_node
    singleton_valid, singleton_test = singleton_completion
    for offset in range(singleton_valid):
        assignments[dp_group_count + offset] = 1
    for offset in range(singleton_valid, singleton_valid + singleton_test):
        assignments[dp_group_count + offset] = 2

    split_indices = {name: [] for name in SPLIT_NAMES}
    scaffold_to_split: dict[str, str] = {}
    for (scaffold, indices), split_index in zip(ordered_groups, assignments):
        split_name = SPLIT_NAMES[split_index]
        split_indices[split_name].extend(indices)
        scaffold_to_split[scaffold] = split_name
    for indices in split_indices.values():
        indices.sort()
    achieved = tuple(len(split_indices[name]) for name in SPLIT_NAMES)
    objective = _count_objective(achieved, targets)
    exact_target_achieved = achieved == targets
    global_minimum_proven = bool(
        exact_target_achieved or (not pruned and singleton_count == 0)
    )
    if global_minimum_proven and not pruned:
        optimality = "global"
    elif exact_target_achieved:
        optimality = "exact_feasible"
    else:
        optimality = "bounded_heuristic"
    optimization = {
        "algorithm": (
            "exhaustive_two_dimensional_group_dp_with_singleton_completion"
            if not pruned
            else "state_bounded_two_dimensional_group_dp_with_singleton_completion"
        ),
        "singleton_completion": (
            "none"
            if singleton_count == 0
            else (
                "exact_target_fill"
                if exact_target_achieved
                else "bounded_projection"
            )
        ),
        "optimality": optimality,
        "state_limit": int(dp_state_limit),
        "peak_states": int(peak_states),
        "peak_candidate_states": int(peak_candidate_states),
        "pruning_events": int(pruning_events),
        "discarded_state_count": int(discarded_state_count),
        "singleton_groups_collapsed": int(singleton_count),
        "states_pruned": bool(pruned),
        "exact_feasibility_proven": bool(not pruned or exact_target_achieved),
        "global_minimum_proven": global_minimum_proven,
        "target_counts": dict(zip(SPLIT_NAMES, targets)),
        "achieved_counts": dict(zip(SPLIT_NAMES, achieved)),
        "exact_target_achieved": bool(exact_target_achieved),
        "absolute_deviation": int(objective[0]),
        "maximum_split_deviation": int(objective[1]),
        "normalized_squared_deviation": float(objective[2]),
    }
    return split_indices, scaffold_to_split, optimization


def scaffold_split(
    smiles_list: Sequence[Any],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
    return_scaffold: bool = False,
    *,
    row_indices: Sequence[int] | None = None,
    invalid_policy: str = "raise",
    return_report: bool = False,
    dp_state_limit: int = 250_000,
):
    """Split raw row indices while keeping every scaffold in one split.

    The original three-list return remains the default for compatibility.
    ``invalid_policy='raise'`` prevents malformed rows from disappearing
    silently.  Use ``invalid_policy='report', return_report=True`` to exclude
    invalid molecules and receive their original row indices explicitly.
    """

    fractions = _validate_fractions((frac_train, frac_valid, frac_test))
    if invalid_policy not in {"raise", "report"}:
        raise ValueError("invalid_policy must be 'raise' or 'report'")
    if invalid_policy == "report" and not return_report:
        raise ValueError(
            "invalid_policy='report' requires return_report=True so invalid rows are explicit"
        )
    if row_indices is None:
        original_indices = list(range(len(smiles_list)))
    else:
        if any(
            not isinstance(index, (int, np.integer)) or isinstance(index, bool)
            for index in row_indices
        ):
            raise ValueError("row_indices must contain strict integers")
        original_indices = [int(index) for index in row_indices]
    if any(index < 0 for index in original_indices):
        raise ValueError("row_indices must be non-negative")
    if len(original_indices) != len(smiles_list):
        raise ValueError("row_indices must have the same length as smiles_list")
    if len(set(original_indices)) != len(original_indices):
        raise ValueError("row_indices must be unique")

    scaffold_groups: dict[str, list[int]] = defaultdict(list)
    invalid: list[dict[str, Any]] = []
    for source_index, smiles in zip(original_indices, smiles_list):
        if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
            invalid.append({"source_index": source_index, "reason": "missing_smiles"})
            continue
        scaffold = generate_scaffold(smiles)
        if scaffold is None:
            reason = "missing_smiles" if not str(smiles).strip() else "invalid_smiles"
            invalid.append({"source_index": source_index, "reason": reason})
            continue
        scaffold_groups[scaffold].append(source_index)

    invalid.sort(key=lambda item: item["source_index"])
    if invalid and invalid_policy == "raise":
        first = invalid[0]
        raise ValueError(
            "invalid SMILES rows encountered; "
            f"source_index={first['source_index']}, reason={first['reason']}, "
            f"invalid_count={len(invalid)}"
        )
    for indices in scaffold_groups.values():
        indices.sort()

    split_indices, scaffold_to_split, optimization = _pack_scaffold_groups(
        scaffold_groups, fractions, seed, dp_state_limit
    )
    base_result: tuple[Any, ...] = (
        split_indices["train"],
        split_indices["valid"],
        split_indices["test"],
    )
    if return_scaffold:
        base_result += (dict(scaffold_groups),)
    if return_report:
        report = {
            "schema": SPLIT_SCHEMA,
            "seed": int(seed),
            "fractions": dict(zip(SPLIT_NAMES, fractions)),
            "input_count": len(smiles_list),
            "valid_count": sum(map(len, scaffold_groups.values())),
            "invalid": invalid,
            "scaffold_groups": dict(scaffold_groups),
            "scaffold_to_split": scaffold_to_split,
            "split_counts": {
                name: len(split_indices[name]) for name in SPLIT_NAMES
            },
            "optimization": optimization,
        }
        base_result += (report,)
    return base_result


def _metadata_payload(
    *,
    train_idx: Sequence[int],
    valid_idx: Sequence[int],
    test_idx: Sequence[int],
    dataset_name: str,
    fractions: Sequence[float],
    seed: int,
    scaffold_groups: Mapping[str, Sequence[int]] | None,
    invalid: Sequence[Mapping[str, Any]],
    optimization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    validated_fractions = _validate_fractions(fractions)
    split_sequences = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }
    if any(
        not isinstance(index, (int, np.integer)) or isinstance(index, bool)
        for values in split_sequences.values()
        for index in values
    ):
        raise ValueError("split indices must contain strict integers")
    split_sets = {
        name: {int(index) for index in values}
        for name, values in split_sequences.items()
    }
    if (
        len(split_sets["train"]) != len(train_idx)
        or len(split_sets["valid"]) != len(valid_idx)
        or len(split_sets["test"]) != len(test_idx)
    ):
        raise ValueError("split index lists must not contain duplicates")
    if any(index < 0 for values in split_sets.values() for index in values):
        raise ValueError("split indices must be non-negative")
    if (
        split_sets["train"] & split_sets["valid"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["valid"] & split_sets["test"]
    ):
        raise ValueError("split indices must be disjoint")

    group_hashes: dict[str, str] = {}
    group_members: dict[str, list[int]] | None = (
        {} if scaffold_groups is not None else None
    )
    scaffold_statistics = {
        "group_count": 0,
        "max_group_size": 0,
        "groups_by_split": {name: 0 for name in SPLIT_NAMES},
    }
    if scaffold_groups is not None:
        grouped_indices: set[int] = set()
        for scaffold, indices in sorted(scaffold_groups.items()):
            if not isinstance(scaffold, str) or not scaffold:
                raise ValueError("scaffold group keys must be non-empty strings")
            scaffold_hash = _scaffold_hash(scaffold)
            if any(
                not isinstance(index, (int, np.integer)) or isinstance(index, bool)
                for index in indices
            ):
                raise ValueError(
                    f"scaffold group {scaffold_hash} indices must be strict integers"
                )
            integer_indices = [int(index) for index in indices]
            if not integer_indices:
                raise ValueError(f"scaffold group {scaffold_hash} is empty")
            if any(index < 0 for index in integer_indices):
                raise ValueError(
                    f"scaffold group {scaffold_hash} indices must be non-negative"
                )
            if len(set(integer_indices)) != len(integer_indices):
                raise ValueError(
                    f"scaffold group {scaffold_hash} contains duplicate indices"
                )
            overlap = grouped_indices.intersection(integer_indices)
            if overlap:
                raise ValueError(
                    f"row index {min(overlap)} belongs to multiple scaffold groups"
                )
            grouped_indices.update(integer_indices)
            member_splits = []
            for index in integer_indices:
                locations = [
                    split_name
                    for split_name, values in split_sets.items()
                    if index in values
                ]
                if len(locations) != 1:
                    raise ValueError(
                        f"scaffold group {scaffold_hash} contains unassigned "
                        f"row index {index}"
                    )
                member_splits.append(locations[0])
            if len(set(member_splits)) != 1:
                raise ValueError(
                    f"scaffold group {scaffold_hash} is missing or crosses splits"
                )
            containing_split = member_splits[0]
            group_hashes[scaffold_hash] = containing_split
            if group_members is None:
                raise RuntimeError("scaffold membership audit was not initialized")
            group_members[scaffold_hash] = sorted(integer_indices)
            scaffold_statistics["groups_by_split"][containing_split] += 1
            scaffold_statistics["max_group_size"] = max(
                scaffold_statistics["max_group_size"], len(indices)
            )
        assigned_indices = set().union(*split_sets.values())
        if grouped_indices != assigned_indices:
            missing = sorted(assigned_indices - grouped_indices)
            raise ValueError(
                "scaffold_groups do not cover every split row; "
                f"first_uncovered_index={missing[0] if missing else None}"
            )
        scaffold_statistics["group_count"] = len(scaffold_groups)

    normalized_invalid = [dict(item) for item in invalid]
    invalid_indices = []
    for item in normalized_invalid:
        source_index = item.get("source_index")
        reason = item.get("reason")
        if (
            not isinstance(source_index, (int, np.integer))
            or isinstance(source_index, bool)
            or int(source_index) < 0
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError("invalid-row report entries must contain index and reason")
        item["source_index"] = int(source_index)
        invalid_indices.append(int(source_index))
    if len(set(invalid_indices)) != len(invalid_indices):
        raise ValueError("invalid-row report contains duplicate source indices")
    if set().union(*split_sets.values()).intersection(invalid_indices):
        raise ValueError("invalid rows overlap assigned split indices")

    return {
        "schema": SPLIT_SCHEMA,
        "dataset_name": str(dataset_name),
        "seed": int(seed),
        "fractions": dict(zip(SPLIT_NAMES, validated_fractions)),
        "statistics": {
            "split_counts": {
                "train": len(train_idx),
                "valid": len(valid_idx),
                "test": len(test_idx),
            },
            "valid_count": len(train_idx) + len(valid_idx) + len(test_idx),
            "invalid_count": len(invalid),
            "scaffolds": scaffold_statistics,
        },
        "scaffold_hashes": group_hashes,
        "scaffold_members": group_members,
        "invalid": normalized_invalid,
        "optimization": dict(optimization or {}),
    }


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def save_scaffold_split(
    path: str | os.PathLike[str],
    train_idx: Sequence[int],
    valid_idx: Sequence[int],
    test_idx: Sequence[int],
    *,
    dataset_name: str = "unknown",
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    scaffold_groups: Mapping[str, Sequence[int]] | None = None,
    invalid: Sequence[Mapping[str, Any]] = (),
    optimization: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a split as JSON or non-pickle NPZ."""

    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".json", ".npz"}:
        raise ValueError(
            "safe scaffold split output must end in .json or .npz; pickle is unsupported"
        )
    metadata = _metadata_payload(
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        dataset_name=dataset_name,
        fractions=fractions,
        seed=seed,
        scaffold_groups=scaffold_groups,
        invalid=invalid,
        optimization=optimization,
    )
    temporary = _temporary_path(destination)
    try:
        if suffix == ".json":
            payload = {
                **metadata,
                "indices": {
                    "train": list(map(int, train_idx)),
                    "valid": list(map(int, valid_idx)),
                    "test": list(map(int, test_idx)),
                },
            }
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
        else:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    train=np.asarray(train_idx, dtype=np.int64),
                    valid=np.asarray(valid_idx, dtype=np.int64),
                    test=np.asarray(test_idx, dtype=np.int64),
                    metadata_json=np.asarray(
                        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                    ),
                )
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            print(
                f"warning: could not remove temporary file {temporary}: "
                f"{cleanup_error}",
                file=sys.stderr,
            )
        raise


def _validate_loaded_split(
    metadata: Mapping[str, Any],
    indices: Mapping[str, Sequence[Any]],
) -> tuple[list[int], list[int], list[int]]:
    if not isinstance(metadata, Mapping):
        raise ValueError("scaffold split metadata must be a mapping")
    if set(indices) != set(SPLIT_NAMES):
        raise ValueError("scaffold split indices must contain exactly three splits")
    if metadata.get("schema") != SPLIT_SCHEMA:
        raise ValueError(
            f"unsupported scaffold split schema: {metadata.get('schema')!r}"
        )
    if not isinstance(metadata.get("dataset_name"), str):
        raise ValueError("scaffold split is missing dataset_name")
    if (
        not isinstance(metadata.get("seed"), int)
        or isinstance(metadata.get("seed"), bool)
    ):
        raise ValueError("scaffold split seed must be an integer")
    fractions = metadata.get("fractions")
    if not isinstance(fractions, dict) or set(fractions) != set(SPLIT_NAMES):
        raise ValueError("scaffold split has invalid fractions")
    _validate_fractions(tuple(fractions[name] for name in SPLIT_NAMES))
    loaded: dict[str, list[int]] = {}
    for split_name in SPLIT_NAMES:
        values = indices.get(split_name)
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(f"scaffold split is missing {split_name} indices")
        if any(
            not isinstance(value, (int, np.integer)) or isinstance(value, bool)
            for value in values
        ):
            raise ValueError(f"{split_name} indices must contain integers")
        converted = [int(value) for value in values]
        if any(value < 0 for value in converted):
            raise ValueError(f"{split_name} indices must be non-negative")
        if len(set(converted)) != len(converted):
            raise ValueError(f"{split_name} indices contain duplicates")
        loaded[split_name] = converted
    split_sets = {name: set(values) for name, values in loaded.items()}
    if (
        split_sets["train"] & split_sets["valid"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["valid"] & split_sets["test"]
    ):
        raise ValueError("loaded scaffold split indices are not disjoint")
    statistics = metadata.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("scaffold split is missing statistics")
    expected_counts = statistics.get("split_counts")
    actual_counts = {name: len(loaded[name]) for name in SPLIT_NAMES}
    if (
        not isinstance(expected_counts, dict)
        or set(expected_counts) != set(SPLIT_NAMES)
        or any(
            not isinstance(expected_counts[name], int)
            or isinstance(expected_counts[name], bool)
            or expected_counts[name] < 0
            for name in SPLIT_NAMES
        )
    ):
        raise ValueError("split count metadata is malformed")
    if expected_counts != actual_counts:
        raise ValueError(
            f"split count integrity failure: expected={expected_counts}, "
            f"actual={actual_counts}"
        )
    if (
        not isinstance(statistics.get("valid_count"), int)
        or isinstance(statistics.get("valid_count"), bool)
        or statistics.get("valid_count") != sum(actual_counts.values())
    ):
        raise ValueError("valid_count integrity failure")
    invalid = metadata.get("invalid")
    if not isinstance(invalid, list):
        raise ValueError("scaffold split invalid-row report is malformed")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("source_index"), int)
        or isinstance(item.get("source_index"), bool)
        or not isinstance(item.get("reason"), str)
        or not item.get("reason")
        for item in invalid
    ):
        raise ValueError("scaffold split invalid-row entries are malformed")
    invalid_indices = [int(item["source_index"]) for item in invalid]
    if any(index < 0 for index in invalid_indices):
        raise ValueError("invalid-row report contains negative source indices")
    if len(set(invalid_indices)) != len(invalid_indices):
        raise ValueError("invalid-row report contains duplicate source indices")
    assigned = set().union(*split_sets.values())
    if assigned.intersection(invalid_indices):
        raise ValueError("invalid rows overlap assigned split indices")
    if (
        not isinstance(statistics.get("invalid_count"), int)
        or isinstance(statistics.get("invalid_count"), bool)
        or statistics.get("invalid_count") != len(invalid)
    ):
        raise ValueError("invalid_count integrity failure")
    scaffold_hashes = metadata.get("scaffold_hashes")
    if not isinstance(scaffold_hashes, dict):
        raise ValueError("scaffold_hashes must be a mapping")
    hexadecimal = set("0123456789abcdef")
    for scaffold_hash, split_name in scaffold_hashes.items():
        if (
            not isinstance(scaffold_hash, str)
            or len(scaffold_hash) != 64
            or set(scaffold_hash) - hexadecimal
            or split_name not in SPLIT_NAMES
        ):
            raise ValueError("scaffold hash integrity failure")
    scaffold_members = metadata.get("scaffold_members")
    audited_group_sizes: list[int] | None = None
    if scaffold_members is None:
        if scaffold_hashes:
            raise ValueError(
                "scaffold membership audit is missing for hashed scaffold groups"
            )
    else:
        if (
            not isinstance(scaffold_members, dict)
            or set(scaffold_members) != set(scaffold_hashes)
        ):
            raise ValueError("scaffold membership audit keys do not match hashes")
        grouped_indices: set[int] = set()
        audited_group_sizes = []
        audited_rows_by_split = {name: 0 for name in SPLIT_NAMES}
        for scaffold_hash, members in scaffold_members.items():
            if not isinstance(members, (list, tuple, np.ndarray)) or not members:
                raise ValueError("scaffold membership groups must be non-empty")
            if any(
                not isinstance(index, (int, np.integer))
                or isinstance(index, bool)
                or int(index) < 0
                for index in members
            ):
                raise ValueError(
                    "scaffold membership groups must contain non-negative integers"
                )
            integer_members = [int(index) for index in members]
            if len(set(integer_members)) != len(integer_members):
                raise ValueError("scaffold membership group contains duplicates")
            overlap = grouped_indices.intersection(integer_members)
            if overlap:
                raise ValueError(
                    "row index belongs to multiple scaffold groups: "
                    f"source_index={min(overlap)}"
                )
            split_name = scaffold_hashes[scaffold_hash]
            if any(index not in split_sets[split_name] for index in integer_members):
                raise ValueError(
                    f"scaffold group {scaffold_hash} crosses or escapes its split"
                )
            grouped_indices.update(integer_members)
            audited_group_sizes.append(len(integer_members))
            audited_rows_by_split[split_name] += len(integer_members)
        if grouped_indices != assigned:
            raise ValueError(
                "scaffold membership groups do not exactly cover split indices"
            )
        if audited_rows_by_split != actual_counts:
            raise ValueError("scaffold membership split counts are inconsistent")
    scaffold_statistics = statistics.get("scaffolds")
    if not isinstance(scaffold_statistics, dict):
        raise ValueError("scaffold statistics are missing")
    for field in ("group_count", "max_group_size"):
        if (
            not isinstance(scaffold_statistics.get(field), int)
            or isinstance(scaffold_statistics.get(field), bool)
            or scaffold_statistics[field] < 0
        ):
            raise ValueError(f"scaffold {field} integrity failure")
    if scaffold_statistics.get("group_count") != len(scaffold_hashes):
        raise ValueError("scaffold group_count integrity failure")
    actual_groups_by_split = {
        name: sum(split_name == name for split_name in scaffold_hashes.values())
        for name in SPLIT_NAMES
    }
    reported_groups_by_split = scaffold_statistics.get("groups_by_split")
    if (
        not isinstance(reported_groups_by_split, dict)
        or set(reported_groups_by_split) != set(SPLIT_NAMES)
        or any(
            not isinstance(reported_groups_by_split[name], int)
            or isinstance(reported_groups_by_split[name], bool)
            or reported_groups_by_split[name] < 0
            for name in SPLIT_NAMES
        )
        or reported_groups_by_split != actual_groups_by_split
    ):
        raise ValueError("scaffold groups_by_split integrity failure")
    expected_max_group_size = (
        max(audited_group_sizes, default=0)
        if audited_group_sizes is not None
        else (0 if not scaffold_hashes else None)
    )
    if (
        expected_max_group_size is not None
        and scaffold_statistics["max_group_size"] != expected_max_group_size
    ):
        raise ValueError("scaffold max_group_size integrity failure")
    optimization = metadata.get("optimization", {})
    if not isinstance(optimization, dict):
        raise ValueError("scaffold split is missing optimization audit data")
    for field in ("algorithm", "optimality"):
        if field in optimization and (
            not isinstance(optimization[field], str) or not optimization[field]
        ):
            raise ValueError(f"optimization {field} integrity failure")
    for field in (
        "states_pruned",
        "exact_feasibility_proven",
        "global_minimum_proven",
        "exact_target_achieved",
    ):
        if field in optimization and not isinstance(optimization[field], bool):
            raise ValueError(f"optimization {field} integrity failure")
    for field in (
        "state_limit",
        "peak_states",
        "peak_candidate_states",
        "pruning_events",
        "discarded_state_count",
        "singleton_groups_collapsed",
    ):
        if field in optimization and (
            not isinstance(optimization[field], int)
            or isinstance(optimization[field], bool)
            or optimization[field] < 0
        ):
            raise ValueError(f"optimization {field} integrity failure")

    def validated_count_mapping(value: Any, field: str) -> dict[str, int]:
        if (
            not isinstance(value, dict)
            or set(value) != set(SPLIT_NAMES)
            or any(
                not isinstance(value[name], int)
                or isinstance(value[name], bool)
                or value[name] < 0
                for name in SPLIT_NAMES
            )
        ):
            raise ValueError(f"optimization {field} integrity failure")
        return {name: int(value[name]) for name in SPLIT_NAMES}

    achieved_counts = optimization.get("achieved_counts")
    if achieved_counts is not None:
        achieved_counts = validated_count_mapping(
            achieved_counts, "achieved_counts"
        )
        if achieved_counts != actual_counts:
            raise ValueError("optimization achieved_counts integrity failure")
    target_counts = optimization.get("target_counts")
    if target_counts is not None:
        target_counts = validated_count_mapping(target_counts, "target_counts")
        if sum(target_counts.values()) != sum(actual_counts.values()):
            raise ValueError("optimization target_counts integrity failure")
        if achieved_counts is None:
            raise ValueError(
                "optimization target_counts require achieved_counts audit data"
            )
        target_values = tuple(target_counts[name] for name in SPLIT_NAMES)
        achieved_values = tuple(actual_counts[name] for name in SPLIT_NAMES)
        objective = _count_objective(achieved_values, target_values)
        exact_target_achieved = achieved_values == target_values
        if (
            "exact_target_achieved" in optimization
            and optimization["exact_target_achieved"] != exact_target_achieved
        ):
            raise ValueError(
                "optimization exact_target_achieved integrity failure"
            )
        if (
            "absolute_deviation" in optimization
            and optimization["absolute_deviation"] != objective[0]
        ):
            raise ValueError("optimization absolute deviation integrity failure")
        if (
            "maximum_split_deviation" in optimization
            and optimization["maximum_split_deviation"] != objective[1]
        ):
            raise ValueError("optimization maximum deviation integrity failure")
        if "normalized_squared_deviation" in optimization:
            value = optimization["normalized_squared_deviation"]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value), objective[2], rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise ValueError(
                    "optimization normalized deviation integrity failure"
                )
    return loaded["train"], loaded["valid"], loaded["test"]


def load_scaffold_split(
    path: str | os.PathLike[str],
) -> tuple[list[int], list[int], list[int]]:
    """Load safe JSON/NPZ while preserving the historical tuple API."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("scaffold split JSON must contain an object")
        indices = payload.get("indices")
        if not isinstance(indices, dict):
            raise ValueError("scaffold split JSON is missing indices")
        return _validate_loaded_split(payload, indices)
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as arrays:
            expected_arrays = {*SPLIT_NAMES, "metadata_json"}
            if set(arrays.files) != expected_arrays:
                raise ValueError(
                    "scaffold split NPZ contains missing or unexpected arrays"
                )
            metadata = json.loads(str(arrays["metadata_json"].item()))
            indices = {}
            for name in SPLIT_NAMES:
                values = arrays[name]
                if values.ndim != 1 or values.dtype.kind not in {"i", "u"}:
                    raise ValueError(f"{name} NPZ indices must be a 1-D integer array")
                indices[name] = values.astype(np.int64, copy=False).tolist()
            return _validate_loaded_split(metadata, indices)
    raise ValueError(
        "safe scaffold split input must end in .json or .npz; pickle loading is disabled"
    )
