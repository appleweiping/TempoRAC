"""CPU LF1 hand calculation, gradients and exact unit weighting; no natural labels."""
import math

import pytest
import torch

from pams.temporac.objective import balanced_response_functional
from pams.temporac.objective_lf1 import (
    LF1ContractError,
    SourceIdentityCycle,
    balanced_family_step_loss,
    balanced_pulse_bce,
    equal_source_reduction,
    traversal_identity_loss,
)


def test_hand_bce_gradient_detached_targets():
    x = torch.tensor([.8, .2, .4], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([1., 0., 0.], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([.2, .3, .5], dtype=torch.float64, requires_grad=True)
    loss = balanced_pulse_bce(x, p, w)
    expected = -.5 * (math.log(.8) + (.3 * math.log(.8) + .5 * math.log(.6)) / .8)
    assert float(loss) == pytest.approx(expected, abs=1e-14)
    loss.backward()
    assert float(x.grad[0]) == pytest.approx(-.5 / .8)
    assert float(x.grad[1]) == pytest.approx(.5 * .3 / .8 / .8)
    assert float(x.grad[2]) == pytest.approx(.5 * .5 / .8 / .6)
    assert p.grad is None and w.grad is None


def fixture():
    branches = torch.tensor([[.8, .7, .9], [.2, .3, .1], [.4, .2, .3], [.1, .1, .1],
                             [.9, .7, .8], [.2, .4, .3], [.3, .2, .1]], dtype=torch.float64, requires_grad=True)
    fused = branches.mean(1)
    fused.retain_grad()
    cue = torch.zeros((7, 3), dtype=torch.float64, requires_grad=True)
    pi = cue.softmax(1)
    pulse = torch.tensor([1., 0, 0, 0, 1, 0, 0], dtype=torch.float64, requires_grad=True)
    chi = torch.tensor([.2, .3, .5, 0, .2, .3, .5], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
    return branches, fused, cue, pi, pulse, chi, mask


def test_heads_fused_grad_but_no_cue_and_unsupported_zero():
    b, f, cue, pi, p, c, m = fixture()
    result = traversal_identity_loss(b, f, p, m, c, pi, [[0, 3], [4, 7]])
    result.total.backward()
    assert (b.grad[m].abs().sum(0) > 0).all() and (f.grad[m].abs() > 0).all()
    assert b.grad[3].abs().sum() == 0 and f.grad[3] == 0
    assert cue.grad is None and p.grad is None and c.grad is None
    # Same identity split into two run payloads yields the exact traversal mean.
    pieces = []
    for left, r in ((0, 3), (4, 7)):
        pieces.append(traversal_identity_loss(b[left:r], f[left:r], p[left:r], m[left:r], c[left:r], pi[left:r], [[0, 3]]).total)
    assert torch.allclose(result.total, torch.stack(pieces).mean(), rtol=0, atol=1e-15)


def test_invalid_weight_mass_binary_support_and_underflow():
    x = torch.full((3,), .5, dtype=torch.float64)
    for p, w in (([1, 0, 0], [0., .5, .5]), ([1, 0, 0], [1., 0, 0]), ([.2, 0, 0], [1., 1, 1])):
        with pytest.raises(LF1ContractError):
            balanced_pulse_bce(x, torch.tensor(p), torch.tensor(w))
    b, f, _, pi, p, c, m = fixture()
    bad = m.clone()
    bad[0] = False
    with pytest.raises(LF1ContractError, match='support loss'):
        traversal_identity_loss(b, f, p, bad, c, pi, [[0, 3], [4, 7]])
    pi = pi.detach().clone()
    pi[0, 0] = 0
    with pytest.raises(LF1ContractError):
        traversal_identity_loss(b, f, p, m, c, pi, [[0, 3], [4, 7]])


def test_legacy_minus_lf1_exact_margin_contribution():
    x = torch.tensor([.5, .5, .8], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([1., 0, 0], dtype=torch.float64)
    w = torch.tensor([.2, .3, .5], dtype=torch.float64)
    old = balanced_response_functional(x, p, w)
    new = balanced_pulse_bce(x, p, w)
    assert torch.allclose(old.total - new, .5 * old.positive_margin + .5 * old.negative_valley,
                          rtol=0, atol=1e-15)


def test_source_component_cycle_and_balanced_family():
    cycle = SourceIdentityCycle({'source_a': ['alias_b', 'alias_a'], 'source_b': ['id_c'], 'empty': []},
                                ['source_a', 'source_b', 'empty'])
    assert [cycle.at(i) for i in range(4)] == [('source_a', 'alias_a'), ('source_b', 'id_c'),
                                             ('source_a', 'alias_b'), ('source_b', 'id_c')]
    assert cycle.unavailable_components == ('empty',)
    assert balanced_family_step_loss(torch.tensor(2.), torch.tensor(4.)).item() == 3
    with pytest.raises(LF1ContractError, match='empty identity'):
        equal_source_reduction({'a': []}, population_components=['a'])
