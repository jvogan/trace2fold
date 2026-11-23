"""Fixed residue correspondence. Distances are in angstroms."""

import numpy as np

WEIGHTS = {"aligned_coverage": 0.50, "ca_lddt": 0.30, "contact_f1": 0.20}
SPACEFILL_PROFILE = "spacefill-v4"
SPACEFILL_WEIGHTS = {"aligned_coverage": 0.40, "ca_lddt": 0.25, "contact_f1": 0.35}
SPACEFILL_CUTOFFS = (0.5, 1, 2, 4)


class RigidAlignmentError(ValueError):
    """The mobile coordinates do not define a rigid alignment."""


def rigid_fit(mobile, target, fit_mask):
    a, b = mobile[fit_mask], target[fit_mask]
    if len(a) < 3 or np.linalg.matrix_rank(a - a.mean(0)) < 2:
        raise RigidAlignmentError("alignment requires three noncollinear positions")
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    u[:, -1] *= np.linalg.det(u @ vt)
    rotation = u @ vt
    return rotation, b.mean(0) - a.mean(0) @ rotation


def measure(prediction, case, *, profile=None):
    """Score editable residues; use the fixed framework for loop alignment."""
    target = case.xyz
    pred = np.asarray(prediction, dtype=float)
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError("prediction does not cover all target positions")
    editable = case.mutable
    if profile is None:
        weights, cutoffs = WEIGHTS, (1, 2, 4, 8)
        loop = "mutable" in case.info
    elif profile == SPACEFILL_PROFILE:
        if len(case.lengths) != 1 or case.info.get("disulfides"):
            raise ValueError(
                "spacefill-v4 requires one chain without specified disulfides"
            )
        weights, cutoffs = SPACEFILL_WEIGHTS, SPACEFILL_CUTOFFS
        loop = not editable.all()
        anchors = target[~editable]
        if loop and (
            len(anchors) < 3 or np.linalg.matrix_rank(anchors - anchors.mean(0)) < 2
        ):
            raise ValueError(
                "masked scoring requires three noncollinear fixed positions"
            )
    else:
        raise ValueError(f"unknown scoring profile: {profile}")
    fit = ~editable if loop else np.ones(len(pred), dtype=bool)
    rotation, offset = rigid_fit(pred, target, fit)
    aligned = pred @ rotation + offset
    dist = np.linalg.norm(aligned[editable] - target[editable], axis=1)
    # One fixed alignment at all cutoffs. Name explicitly to distinguish from optimized GDT.
    gdt = np.mean(dist[:, None] <= cutoffs)
    inds = np.arange(len(pred))
    i, j = np.triu_indices(len(pred), 1)
    chains = np.repeat(np.arange(len(case.lengths)), case.lengths)
    keep = (editable[i] | editable[j]) & (
        (chains[i] != chains[j]) | (inds[j] - inds[i] > 1)
    )
    i, j = i[keep], j[keep]
    dt = np.linalg.norm(target[i] - target[j], axis=1)
    dp = np.linalg.norm(pred[i] - pred[j], axis=1)
    close = dt < 15
    local = (
        float(np.mean(np.abs(dp[close] - dt[close])[:, None] < [0.5, 1, 2, 4]))
        if close.any()
        else None
    )
    distant = (chains[i] != chains[j]) | (j - i >= 6)
    true_contacts = (dt < 8) & distant
    pred_contacts = (dp < 8) & distant
    # A target without any nonlocal contact carries no information for contact recovery.
    f1 = (
        (
            2
            * np.count_nonzero(true_contacts & pred_contacts)
            / (true_contacts.sum() + pred_contacts.sum())
        )
        if true_contacts.any()
        else None
    )
    terms = {
        "aligned_coverage": float(gdt),
        "ca_lddt": local,
        "contact_f1": None if f1 is None else float(f1),
    }
    used = [k for k, v in terms.items() if v is not None]
    total = sum(weights[k] * terms[k] for k in used) / sum(weights[k] for k in used)
    results = {
        "score": 100 * total,
        **terms,
        "rmsd": float(np.sqrt(np.mean(dist**2))),
        "positions_scored": int(editable.sum()),
        "native_contacts": int(true_contacts.sum()),
    }
    if len(case.lengths) > 1:
        inter = chains[i] != chains[j]
        native = inter & (dt < 8)
        observed = inter & (dp < 8)
        results["interchain_contact_f1"] = (
            2 * int((native & observed).sum()) / int(native.sum() + observed.sum())
            if native.any()
            else None
        )
    return results, aligned
