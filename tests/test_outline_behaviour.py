"""行为回归测试：固定六项描边修复的语义。

不能只测“函数不抛异常”，还要测：
硬边确实拆开、软边不裂；像素宽度在 FOV/距离变化下恒定；
描边参与深度测试；两条路径互斥；多 pass 深度基准统一；
后处理描边先重建深度再做屏幕空间检测。
"""

import math

import pytest

import graphics.outline_expand as mod

CUBE_FACE_NORMALS = {
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
}


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --- 1. 硬边按顶点法线拆分 ------------------------------------------------

def test_cube_hard_edges_are_split():
    result = mod.hard_edge_split()
    # 立方体 12 条棱全是硬边；8 个角点各拆 3 份 -> 24 个输出顶点
    assert result["hard_edges"] == 12
    assert result["source_vertex_count"] == 8
    assert result["vertex_count"] == 24
    assert result["split_count"] == 16
    # 每个角点都有三个互相垂直的面法线，没有被平均成体对角线
    for i, normal in enumerate(result["normals"]):
        rounded = tuple(round(c) for c in normal)
        assert rounded in CUBE_FACE_NORMALS, (i, normal)


def test_cube_normals_are_not_averaged_diagonals():
    normals = mod.hard_edge_split()["normals"]
    for normal in normals:
        length = math.sqrt(sum(c * c for c in normal))
        assert _approx(length, 1.0, 1e-9)
        # 平均后的体对角线法线每个分量约为 ±0.577，必须不出现
        assert not all(_approx(abs(c), 1 / math.sqrt(3), 1e-6) for c in normal)


def test_smooth_surface_keeps_shared_vertices():
    planet = {
        "positions": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        "indices": [(0, 1, 2), (0, 2, 3)],
    }
    result = mod.hard_edge_split(planet)
    assert result["hard_edges"] == 0
    assert result["vertex_count"] == 4
    assert result["split_count"] == 0


def test_hard_fold_splits_shared_edge():
    # 两个 90 度相交的四边形共享一条边
    fold = {
        "positions": [
            (0, 0, 0), (1, 0, 0),
            (0, 1, 0), (1, 1, 0),
            (0, 1, -1), (1, 1, -1),
        ],
        "indices": [(0, 1, 3), (0, 3, 2), (2, 3, 5), (2, 5, 4)],
    }
    result = mod.hard_edge_split(fold)
    assert result["hard_edges"] == 1
    # 共享棱上的两个顶点各自被复制一份
    assert result["split_count"] == 2


def test_normal_expand_has_no_cracks_on_shared_edges():
    assert mod.normal_expand()["cracks"] == 0
    smooth = {
        "positions": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        "indices": [(0, 1, 2), (0, 2, 3)],
    }
    assert mod.normal_expand(smooth)["cracks"] == 0
    # 立方体的硬边表现为“预期的开口”，不是裂缝
    expanded = mod.normal_expand()
    assert expanded["open_seams"] == 12 and expanded["cracks"] == 0


# --- 2. 宽度按视锥换算 ---------------------------------------------------

def test_width_fov_keeps_screen_pixels_constant():
    info = mod.metrics(3.0)
    assert info["constant"] is True
    assert info["max_deviation_px"] <= 1e-9
    for sample in info["samples"]:
        assert _approx(sample["recovered_px"], 3.0, 1e-9)


def test_width_fov_compensates_narrow_fov():
    narrow = mod.width_fov(2.0, fov_y_deg=30.0, screen_height=1080, distance=10.0)
    wide = mod.width_fov(2.0, fov_y_deg=90.0, screen_height=1080, distance=10.0)
    # 窄 FOV 下同一像素对应的世界尺寸更小
    assert narrow["world_width"] < wide["world_width"]
    assert _approx(
        narrow["world_width"] / wide["world_width"],
        math.tan(math.radians(15.0)) / math.tan(math.radians(45.0)),
    )
    assert _approx(narrow["fov_scale"], math.tan(math.radians(15.0)) / math.tan(math.radians(30.0)))
    assert _approx(wide["fov_scale"], math.tan(math.radians(45.0)) / math.tan(math.radians(30.0)))
    assert mod.width_fov(2.0, fov_y_deg=60.0)["fov_scale"] == pytest.approx(1.0)


def test_width_scales_with_resolution_and_distance():
    low = mod.width_fov(2.0, screen_height=720, distance=1.0)["world_width"]
    high = mod.width_fov(2.0, screen_height=1440, distance=1.0)["world_width"]
    assert _approx(high, low / 2.0)
    near = mod.width_fov(2.0, distance=5.0)["world_width"]
    far = mod.width_fov(2.0, distance=10.0)["world_width"]
    assert _approx(far, 2.0 * near)


def test_thickness_clamps_to_visible_range():
    thin = mod.thickness(0.1)
    thick = mod.thickness(500.0, max_px=64.0)
    assert thin["clamped"] is True and _approx(thin["thickness_px"], 1.0)
    assert thick["clamped"] is True and _approx(thick["thickness_px"], 64.0)
    normal = mod.thickness(4.0)
    assert normal["clamped"] is False and _approx(normal["thickness_px"], 4.0)


def test_width_fov_rejects_bad_fov():
    for bad in (0.0, 180.0, -30.0):
        with pytest.raises(ValueError):
            mod.width_fov(2.0, fov_y_deg=bad)


def test_normal_expand_prefers_provided_normals():
    smooth = {
        "positions": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        "indices": [(0, 1, 2), (0, 2, 3)],
        "normals": [(0, 0, 1)] * 4,
    }
    result = mod.normal_expand(smooth)
    assert result["normal_source"] == "provided"
    assert result["cracks"] == 0
    # 外扩方向就是提供的 +Z，位移大小等于换算后的世界宽度
    assert _approx(result["positions"][0][2], result["width_world"])


# --- 3. 描边参与深度测试 --------------------------------------------------
def test_outline_hidden_behind_wall():
    hidden = mod.occluded_outline(5.0, 2.0)
    assert hidden["visible"] is False
    assert hidden["depth_test"] is True
    assert hidden["depth_write"] is False


def test_outline_visible_when_in_front():
    visible = mod.occluded_outline(1.0, 2.0)
    assert visible["visible"] is True


def test_occlusion_sequence_reports_ratio():
    result = mod.occluded_outline(1.0, [0.5, 2.0, 3.0])
    assert result["visibility"] == [False, True, True]
    assert _approx(result["visible_ratio"], 2.0 / 3.0)


def test_depth_bias_moves_towards_camera_and_does_not_pierce_wall():
    # 偏置只消除自身深度冲突，不会让被墙挡住的描边穿出来
    assert mod.occluded_outline(1.0001, 1.0)["visible"] is False
    assert mod.occluded_outline(1.0001, 1.0, depth_bias=0.001)["visible"] is True
    assert mod.occluded_outline(5.0, 2.0, depth_bias=1e-3)["visible"] is False


def test_occluded_outline_reversed_z():
    assert mod.occluded_outline(0.9, 0.8, depth_func="greater_equal")["visible"] is True
    assert mod.occluded_outline(0.7, 0.8, depth_func="greater_equal")["visible"] is False


# --- 4. 两条描边路径互斥 --------------------------------------------------

def test_dual_path_never_both_enabled():
    assert mod.dual_path(1)["path"] == "geometry"
    assert mod.dual_path(0)["path"] == "post"
    assert mod.dual_path(0, 0)["path"] == "none"
    with pytest.raises(ValueError):
        mod.dual_path(1, 1)


# --- 5. 统一深度比较函数 --------------------------------------------------

def test_depth_compare_semantics():
    assert mod.depth_compare(0.5, 0.5, "less_equal") is True
    assert mod.depth_compare(0.5, 0.5, "less") is False
    assert mod.depth_compare(0.5, 0.5, "greater_equal") is True
    assert mod.depth_compare(0.5, 0.5, "greater") is False
    with pytest.raises(ValueError):
        mod.depth_compare(0.5, 0.5, "lequal")


def test_default_pipeline_is_depth_consistent():
    assert mod.depth_consistency()["consistent"] is True
    assert mod.audit()["passed"] is True


def test_mixed_depth_baselines_are_reported():
    passes = [
        {"name": "outline_a", "path": "geometry", "depth_func": "less_equal",
         "depth_test": True, "depth_write": False, "reversed_z": False},
        {"name": "outline_b", "path": "geometry", "depth_func": "greater_equal",
         "depth_test": True, "depth_write": False, "reversed_z": True},
    ]
    result = mod.depth_consistency(passes)
    assert result["consistent"] is False
    assert len(result["conflicts"]) >= 2


def test_audit_catches_the_five_failure_modes():
    broken = {
        "passes": [
            {"name": "outline_geo", "path": "geometry", "depth_test": False,
             "depth_write": True, "depth_func": "less_equal"},
            {"name": "outline_post", "path": "post", "depth_test": True,
             "depth_write": False, "depth_func": "less_equal",
             "depth_reconstruction": False},
        ]
    }
    result = mod.audit(broken)
    assert result["passed"] is False
    assert result["checks"]["outline_depth_test"] is False
    assert result["checks"]["outline_depth_write_off"] is False
    assert result["checks"]["single_path"] is False
    assert result["checks"]["post_depth_reconstruction"] is False


def test_linearize_depth_is_monotonic_and_round_trips():
    samples = [0.0, 0.1, 0.5, 0.9, 1.0]
    values = [mod.linearize_depth(d, near=0.1, far=1000.0) for d in samples]
    assert values == sorted(values)
    assert _approx(values[0], 0.1)
    assert _approx(values[-1], 1000.0)
    # 非线性深度上等间距的差值在线性空间并不等距（这正是直接比较出错的原因）
    assert abs(values[1] - values[0]) < abs(values[2] - values[1])
    with pytest.raises(ValueError):
        mod.linearize_depth(1.5)


def test_depth_consistency_and_audit_agree_on_pass_count():
    for count in (1, 2, 3, 5):
        assert mod.depth_consistency(count)["pass_count"] == count
        assert mod.audit(count)["pass_count"] == count


# --- 6. 后处理描边按深度重建 ----------------------------------------------

def test_screen_space_reconstructs_linear_depth():
    result = mod.screen_space(1)
    assert result["reconstruction"] == "linearized_depth"
    linear = result["linearized_depth"]
    # 近处方块重建后的线性深度明显小于背景
    assert linear[3][3] < linear[0][0]
    assert result["edge_count"] > 0


def test_screen_space_radius_controls_thickness():
    thin = mod.screen_space(1)
    thick = mod.screen_space(2)
    assert thick["dilated_count"] > thin["dilated_count"]
    assert thin["radius"] == 1 and thick["radius"] == 2


def test_screen_space_normal_edges():
    flat = [[(0.0, 0.0, 1.0) for _ in range(4)] for _ in range(4)]
    normal_only = mod.screen_space(
        0, depth_buffer=[[0.5] * 4 for _ in range(4)], normals_buffer=flat
    )
    assert normal_only["edge_count"] == 0
    bent = [row[:] for row in flat]
    bent[2][2] = (1.0, 0.0, 0.0)
    bent_only = mod.screen_space(
        0, depth_buffer=[[0.5] * 4 for _ in range(4)], normals_buffer=bent
    )
    assert bent_only["edge_count"] > 0


def test_screen_space_rejects_bad_buffers():
    with pytest.raises(ValueError):
        mod.screen_space(1, depth_buffer=[])
    with pytest.raises(ValueError):
        mod.screen_space(1, depth_buffer=[[0.5, 0.5], [0.5]])


# --- 其它入口：模板、透明、确定性、校验、回退 ------------------------------

def test_stencil_writes_before_testing():
    info = mod.stencil(1, compare="not_equal")
    assert info["order"] == "write_then_test"
    assert info["depth_write"] is False
    with pytest.raises(ValueError):
        mod.stencil(256)
    with pytest.raises(ValueError):
        mod.stencil(1, compare="equals")


def test_transparent_contributes_but_does_not_write_depth():
    glass = mod.transparent(0.4)
    assert glass["contributes_outline"] is True
    assert glass["writes_depth"] is False
    assert glass["depth_test_source"] == "opaque_only"
    assert mod.transparent(1.0)["writes_depth"] is True
    with pytest.raises(ValueError):
        mod.transparent(1.5)


def test_determinism_is_stable_and_seeded():
    first = mod.determinism(42, count=4)
    again = mod.determinism(42, count=4)
    assert first["offsets"] == again["offsets"]
    assert first["stable"] is True
    assert all(-0.5 <= o < 0.5 for o in first["offsets"])
    assert mod.determinism(42, count=4, previous=first["offsets"])["stable"] is True
    assert mod.determinism(42, count=4, previous=[0.0] * 4)["stable"] is False
    assert mod.determinism(43, count=4)["offsets"] != first["offsets"]


def test_validation_flags_each_bad_parameter():
    assert mod.validation()["valid"] is True
    cases = [
        dict(width_px=0.0),
        dict(fov_y_deg=180.0),
        dict(screen_height=-1),
        dict(depth_func="bogus"),
        dict(near=10.0, far=1.0),
    ]
    for kwargs in cases:
        result = mod.validation(**kwargs)
        assert result["valid"] is False and result["errors"]


def test_fallback_prefers_provided_normals_and_detects_degenerate():
    mesh = {
        "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        "indices": [(0, 1, 2)],
        "normals": [(0, 0, 2), (0, 0, 2), (0, 0, 2)],
    }
    provided = mod.fallback(mesh)
    assert provided["source"] == "provided"
    assert provided["fallback_used"] is False
    assert all(_approx(n[2], 1.0) for n in provided["normals"])

    mesh["normals"] = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    degenerated = mod.fallback(mesh)
    assert degenerated["source"] == "generated_degenerate"
    assert degenerated["fallback_used"] is True
    assert all(_approx(n[2], 1.0) for n in degenerated["normals"])
