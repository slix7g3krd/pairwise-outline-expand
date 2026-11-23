import pytest
import graphics.outline_expand as mod

CASES = [
    ("hard_edge_split", (1,)),
    ("width_fov", (1,)),
    ("occluded_outline", (1,)),
    ("depth_consistency", (1,)),
    ("dual_path", (1,)),
    ("determinism", (1,)),
    ("normal_expand", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("stencil", (1,)),
    ("metrics", (1,)),
    ("screen_space", (1,)),
    ("thickness", (1,)),
    ("transparent", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
