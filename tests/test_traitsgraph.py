from sleap_roots.traitsgraph import get_traits_graph


def test_get_traits_graph():
    dts = get_traits_graph()
    assert len(dts) == 43
    assert dts[0] == "primary_base_pt"
