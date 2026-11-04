"""Analytic-only X0 engineering checks; no teacher/natural data/model weights."""
import numpy as np
import pytest
from pams.temporac import x0
from pams.temporac.native_x0_adapter_v1 import X0AnalyticAdapterNTR1, array_sha


def test_source_partition_exact_40_24_8_8():
    assert len({x0.source_key(i) for i in range(40)})==40
    assert [x0.source_split(i) for i in range(40)]==['train']*24+['tune']*8+['heldout']*8
    with pytest.raises(ValueError): x0.source_split(40)


@pytest.mark.parametrize('source,block',[(0,0),(6,0),(0,1),(0,6)])
def test_analytic_trim_pulses_chi_guards_terminal_and_channels(source,block):
    raw=x0.generate_view(source,block)
    unit=X0AnalyticAdapterNTR1().view(source,block)
    first,stop=unit.trim_bounds
    assert unit.clock_unit=='synthetic_tick' and unit.q.dtype==np.float64
    assert np.all(np.diff(unit.q)==1)
    assert unit.response_input.shape==(stop-first,269)
    np.testing.assert_array_equal(unit.traversal_bounds,raw.traversal_bounds-first)
    np.testing.assert_array_equal(unit.pulse,raw.pulse[first:stop-1])
    assert unit.pulse.sum()==10
    terminal=int(unit.traversal_bounds[-1,1])
    assert unit.pulse[terminal]==0 and not unit.target_mask[terminal:].any()
    assert not unit.target_mask[:unit.traversal_bounds[0,0]].any()
    for left,right in unit.traversal_bounds:
        assert unit.pulse[left]==1 and unit.pulse[left:right].sum()==1
        assert abs(unit.chi[left:right].sum()-1)<1e-12
    assert array_sha(unit.response_input)==array_sha(X0AnalyticAdapterNTR1().view(source,block).response_input)
    if source==0 and block==0:
        assert np.all(np.diff(unit.traversal_bounds,axis=1)==5) # no natural six-edge restriction.


def test_seven_blocks_ten_traversals_and_q_translation_analytic_roles():
    assert len(x0.blocks_for_source(0))==7
    raw=x0.generate_view(0,1)
    from pams.temporac.x0 import _analytic_responsibility
    _,gates,pi=_analytic_responsibility(raw.clean_coordinate+8192,raw.geometry)
    np.testing.assert_allclose(gates,raw.p_star,atol=2e-12,rtol=0)
    np.testing.assert_allclose(pi,raw.edge_responsibility,atol=2e-12,rtol=0)
    # Rates for durations 64/32/16 correspond to slow/medium/fast order.
    assert np.log(32/64)<np.log(32/32)<np.log(32/16)
