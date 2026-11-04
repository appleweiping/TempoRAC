"""LF1: one class-balanced pulse BCE criterion; no margin or valley terms."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


class LF1ContractError(ValueError):
    """Invalid supervision or numerical failure; never silently drop an identity."""


@dataclass(frozen=True)
class IdentityLossLF1:
    total: Tensor
    branch: tuple[Tensor, Tensor, Tensor]
    fused: Tensor
    traversal_count: int
    criterion: str = 'class_balanced_pulse_BCE'


def balanced_pulse_bce(probability: Tensor, pulse: Tensor, weight: Tensor) -> Tensor:
    if probability.shape != pulse.shape or probability.shape != weight.shape or not probability.numel():
        raise LF1ContractError('probability/pulse/weight shape mismatch or empty')
    x, p = probability, pulse.detach().to(probability)
    w = weight.detach().to(probability)
    if (not bool(torch.isfinite(x).all()) or bool(((x < 0) | (x > 1)).any())
            or not bool(torch.isfinite(w).all()) or bool((w < 0).any())
            or not bool(torch.isfinite(p).all()) or bool(((p != 0) & (p != 1)).any())):
        raise LF1ContractError('finite probabilities, binary pulse and nonnegative weights required')
    wp, wn = (w * p).sum(), (w * (1 - p)).sum()
    if not bool(torch.isfinite(wp) & torch.isfinite(wn) & (wp > 0) & (wn > 0)):
        raise LF1ContractError('W+ and W- must be finite positive; no epsilon/fallback')
    clipped = x.clamp(1e-7, 1 - 1e-7)
    result = -0.5 * ((w * p * clipped.log()).sum() / wp
                     + (w * (1 - p) * torch.log1p(-clipped)).sum() / wn)
    if not bool(torch.isfinite(result)):
        raise LF1ContractError('nonfinite LF1')
    return result


def traversal_identity_loss(branch_probability: Tensor, fused_probability: Tensor,
                            pulse: Tensor, edge_mask: Tensor, chi: Tensor,
                            responsibility: Tensor, traversal_bounds,
                            *, capacity_control: bool = False) -> IdentityLossLF1:
    """Equal certified traversals across all runs of one complete identity.

    Heads/fused probabilities stay differentiable. NOLA mean responsibility,
    cue-derived weights, targets and masks are detached. Caller supplies only
    complete traversals; any lost response support rejects the call.
    """
    if branch_probability.ndim != 2 or branch_probability.shape[1] != 3:
        raise LF1ContractError('branch probabilities require [E,3]')
    e = branch_probability.shape[0]
    if (fused_probability.shape != (e,) or any(a.shape != (e,) for a in (pulse, edge_mask, chi))
            or responsibility.shape != (e, 3)):
        raise LF1ContractError('full native edge-domain shapes required')
    raw = torch.as_tensor(traversal_bounds)
    if raw.dtype not in (torch.int32, torch.int64) or raw.ndim != 2 or raw.shape[1] != 2 or not len(raw):
        raise LF1ContractError('nonempty integer [K,2] traversal bounds required')
    p, m, c, pi = pulse.detach(), edge_mask.detach(), chi.detach(), responsibility.detach()
    if (not bool(torch.isfinite(m).all()) or bool(((m != 0) & (m != 1)).any())
            or not bool(torch.isfinite(c).all()) or bool((c < 0).any())
            or not bool(torch.isfinite(p).all()) or bool(((p != 0) & (p != 1)).any())
            or not bool(torch.isfinite(pi).all()) or bool((pi[m.bool()] <= 0).any())
            or not bool(torch.allclose(pi[m.bool()].sum(1), torch.ones(int(m.sum()), device=pi.device, dtype=pi.dtype),
                                       rtol=0, atol=1e-6))):
        raise LF1ContractError('invalid detached pulse/mask/chi/soft responsibility')
    branch_rows = [[], [], []]
    fused_rows = []
    previous = 0
    for left, r in raw.detach().cpu().tolist():
        if not 0 <= left < r <= e or left < previous:
            raise LF1ContractError('unordered/overlapping/out-of-domain traversals')
        previous = r
        if not bool((m[left:r] == 1).all()):
            raise LF1ContractError('whole traversal ineligible after response support loss')
        if (p[left] != 1 or p[left:r].sum() != 1
                or abs(float(c[left:r].double().sum()) - 1) > 1e-6):
            raise LF1ContractError('one start pulse and normalized chi required')
        w = m[left:r].to(c) * c[left:r]
        fused_rows.append(balanced_pulse_bce(fused_probability[left:r], p[left:r], w))
        for k in range(3):
            wk = w if capacity_control else w * pi[left:r, k]
            branch_rows[k].append(balanced_pulse_bce(branch_probability[left:r, k], p[left:r], wk))
    branch = tuple(torch.stack(rows).mean() for rows in branch_rows)
    fused = torch.stack(fused_rows).mean()
    return IdentityLossLF1(torch.stack(branch).mean() + fused, branch, fused, len(fused_rows))


def equal_identity_reduction(losses: Sequence[IdentityLossLF1]) -> Tensor:
    if not losses:
        raise LF1ContractError('empty identity family cannot be silently discarded')
    return torch.stack([x.total for x in losses]).mean()


def equal_source_reduction(source_losses: Mapping[str, Sequence[IdentityLossLF1]],
                           *, population_components: Sequence[str]) -> Tensor:
    """Explicit equal source vote; empty components require an external eligibility decision."""
    if set(source_losses) != set(population_components) or not population_components:
        raise LF1ContractError('source coverage must include every population component')
    return torch.stack([equal_identity_reduction(source_losses[k]) for k in sorted(source_losses)]).mean()


def balanced_family_step_loss(x0_loss: Tensor, natural_loss: Tensor) -> Tensor:
    if (x0_loss.numel() != 1 or natural_loss.numel() != 1
            or not bool(torch.isfinite(x0_loss) & torch.isfinite(natural_loss))):
        raise LF1ContractError('both finite scalar families required')
    return 0.5 * x0_loss + 0.5 * natural_loss


@dataclass(frozen=True)
class SourceIdentityCycle:
    """Deterministic component then eligible-identity cycle; aliases share one source vote."""
    eligible_by_component: Mapping[str, Sequence[str]]
    population_components: Sequence[str]

    def __post_init__(self):
        if set(self.eligible_by_component) != set(self.population_components):
            raise LF1ContractError('all components, including ineligible ones, must remain in census')
        all_ids = [i for values in self.eligible_by_component.values() for i in values]
        if len(set(all_ids)) != len(all_ids):
            raise LF1ContractError('identity belongs to multiple source votes')
        if not any(self.eligible_by_component.values()):
            raise LF1ContractError('natural family has no eligible identity')

    @property
    def unavailable_components(self):
        return tuple(sorted(k for k, v in self.eligible_by_component.items() if not v))

    def at(self, step: int):
        if type(step) is not int or step < 0:
            raise LF1ContractError('step must be nonnegative integer')
        components = sorted(k for k, v in self.eligible_by_component.items() if v)
        component = components[step % len(components)]
        identities = sorted(self.eligible_by_component[component])
        return component, identities[(step // len(components)) % len(identities)]
