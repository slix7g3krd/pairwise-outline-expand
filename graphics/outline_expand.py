"""轮廓描边外扩（outline expansion）核心模块。

本模块实现两类描边路径，并修复以下六类缺陷：

1. 硬边断裂 —— 法线外扩时顶点法线在硬边处被平均，挤出方向斜向，
   硬边两侧被撕开。``hard_edge_split`` 按平滑组（相邻面法线夹角）
   拆分顶点，硬边两侧各用独立法线；``normal_expand`` 再做外扩。
2. 宽度随 FOV 跳变 —— 线宽写死为世界单位，相机 FOV 一变屏幕宽度就变。
   ``width_fov`` 按视锥换算：像素宽度 -> 世界宽度，并给出 FOV 缩放比；
   ``metrics``/``thickness`` 验证换算后屏幕像素宽度恒定。
3. 描边穿透墙体 —— 描边 pass 不参与深度测试。``occluded_outline``
   让描边片元与场景深度比较，被遮挡的轮廓不显示。
4. 双线叠加 —— 几何描边与后处理描边同时开启会画出两条线。
   ``dual_path`` 强制两条路径互斥。
5. 多 Pass 深度基准不一致 —— 前后景描边颠倒。``depth_consistency``
   与 ``depth_compare`` 统一所有 pass 的深度比较函数与方向约定。
6. 后处理描边浮在表面 —— ``screen_space`` 先线性化重建深度，
   再在屏幕空间按像素半径做边缘检测与膨胀。

所有入口都容忍“只给一个数值”的简化调用，便于单元测试与快速试验；
数值参数会做合法性校验，非法输入抛 ``ValueError``。
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------

DEFAULT_FOV_Y_DEG = 60.0      # 竖直 FOV（度）
DEFAULT_SCREEN_HEIGHT = 1080  # 屏幕像素高度
DEFAULT_WIDTH_PX = 2.0        # 期望描边宽度（屏幕像素）
DEFAULT_NEAR = 0.1            # 近裁剪面
DEFAULT_FAR = 1000.0          # 远裁剪面
DEFAULT_DISTANCE = 1.0        # 用于把像素宽度换算到世界宽度的参考距离
DEFAULT_HARD_EDGE_DEG = 60.0  # 法线夹角超过该值即视为硬边
DEFAULT_DEPTH_FUNC = "less_equal"
DEPTH_COMPARE_FUNCS = ("less", "less_equal", "greater", "greater_equal")

# 模板 pass 的合法比较函数（图形 API 的常见取值）
STENCIL_COMPARE_FUNCS = (
    "never", "less", "less_equal", "equal", "greater",
    "greater_equal", "not_equal", "always",
)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# 基础数学 / 深度工具
# ---------------------------------------------------------------------------

def _check_positive(name, value):
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def depth_compare(a, b, depth_func=DEFAULT_DEPTH_FUNC):
    """统一的深度比较函数（所有 pass 必须走这里，避免前后景描边颠倒）。

    ``a`` 通常是“新片元深度”，``b`` 是“深度缓冲中的深度”。
    常规 Z（近小远大）常用 ``less_equal``；反转 Z（近大远小）用
    ``greater_equal``。返回 ``a`` 是否通过深度测试。
    """
    if depth_func not in DEPTH_COMPARE_FUNCS:
        raise ValueError(
            f"unsupported depth_func {depth_func!r}, "
            f"expected one of {DEPTH_COMPARE_FUNCS}"
        )
    a = float(a)
    b = float(b)
    if depth_func == "less":
        return a < b
    if depth_func == "less_equal":
        return a <= b
    if depth_func == "greater":
        return a > b
    return a >= b


def linearize_depth(depth, near=DEFAULT_NEAR, far=DEFAULT_FAR):
    """把 [0, 1] 的非线性窗口深度重建为视空间（线性）深度。

    后处理描边如果直接在非线性深度上做阈值比较，近裁剪面附近深度
    变化剧烈、远裁剪面附近几乎不变，会出现近处粗线、远处断线。
    先重建再比较，阈值才有一致含义。
    """
    near = _check_positive("near", near)
    far = _check_positive("far", far)
    if far <= near:
        raise ValueError("far must be greater than near")
    depth = float(depth)
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be within [0, 1], got {depth!r}")
    return near * far / (far - depth * (far - near))


def _world_width(width_px, fov_y_deg, screen_height, distance):
    """像素宽度 -> 给定距离处的世界宽度（透视投影的标准换算）。

    一像素在距离 d 处张开的视角为 ``fov / screen_height``，对应的
    世界尺寸为 ``2 * d * tan(fov / 2) / screen_height``（近似小角度，
    但使用正切使其在宽 FOV 下也准确）。
    """
    width_px = _check_positive("width_px", width_px)
    screen_height = _check_positive("screen_height", screen_height)
    distance = _check_positive("distance", distance)
    fov = float(fov_y_deg)
    if not 0.0 < fov < 180.0:
        raise ValueError(f"fov_y_deg must be within (0, 180), got {fov!r}")
    tan_half = math.tan(math.radians(fov) * 0.5)
    return width_px * 2.0 * distance * tan_half / screen_height


# ---------------------------------------------------------------------------
# 网格工具：立方体、面法线、硬边拆分
# ---------------------------------------------------------------------------

def _cube(size=1.0):
    """返回以原点为中心、棱长为 ``size`` 的立方体网格。

    立方体是硬边的典型例子：每个角点被三个互相垂直的面共享，
    若直接平均法线会得到指向体对角线的斜法线，外扩后角上开裂。
    """
    size = _check_positive("size", size)
    h = size * 0.5
    positions = [
        (-h, -h, -h), (h, -h, -h), (-h, h, -h), (h, h, -h),
        (-h, -h, h), (h, -h, h), (-h, h, h), (h, h, h),
    ]
    quads = [
        (1, 3, 7, 5),  # +X
        (0, 4, 6, 2),  # -X
        (2, 6, 7, 3),  # +Y
        (0, 1, 5, 4),  # -Y
        (4, 5, 7, 6),  # +Z
        (0, 2, 3, 1),  # -Z
    ]
    indices = []
    for a, b, c, d in quads:
        # 用质心方向统一朝外，避免绕序不同导致法线朝内
        for tri in ((a, b, c), (a, c, d)):
            p0, p1, p2 = (positions[i] for i in tri)
            n = _face_normal(p0, p1, p2)
            centroid = tuple(sum(p[i] for p in (p0, p1, p2)) / 3.0 for i in range(3))
            if sum(n[i] * centroid[i] for i in range(3)) < 0.0:
                tri = (tri[0], tri[2], tri[1])
            indices.append(tri)
    return {"positions": positions, "indices": indices}


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _length(a)
    if n < _EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _face_normal(p0, p1, p2):
    """返回单位面法线（退化三角形返回零向量）。"""
    return _normalize(_cross(_sub(p1, p0), _sub(p2, p0)))


def _triangle_area_normal(p0, p1, p2):
    """返回 (面积加权法线, 面积)，用于平滑组内的面积加权平均。"""
    raw = _cross(_sub(p1, p0), _sub(p2, p0))
    area = 0.5 * _length(raw)
    if area < _EPS:
        return (0.0, 0.0, 0.0), 0.0
    return (raw[0] * 0.5, raw[1] * 0.5, raw[2] * 0.5), area


def _as_mesh(mesh):
    """把 ``mesh`` 归一化为 ``{"positions", "indices"}``。

    数值参数视为立方体棱长（方便快速试验）；也接受带 ``positions`` /
    ``indices`` 的字典。
    """
    if mesh is None:
        return _cube(1.0)
    if isinstance(mesh, bool):
        raise TypeError("mesh must be a dict or a number, not bool")
    if isinstance(mesh, (int, float)):
        return _cube(mesh)
    if not isinstance(mesh, dict):
        raise TypeError("mesh must be a dict with positions/indices")
    if "positions" not in mesh or "indices" not in mesh:
        raise ValueError("mesh requires 'positions' and 'indices'")
    positions = [tuple(float(c) for c in p) for p in mesh["positions"]]
    indices = [tuple(int(i) for i in tri) for tri in mesh["indices"]]
    if not positions or not indices:
        raise ValueError("mesh must contain at least one triangle")
    for tri in indices:
        if len(tri) != 3:
            raise ValueError("only triangle meshes are supported")
        if any(i < 0 or i >= len(positions) for i in tri):
            raise ValueError("mesh index out of range")
    return {"positions": positions, "indices": indices}


class _UnionFind:
    """极小并查集，用于按顶点聚合属于同一平滑组的面。"""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 固定取较小 id 为根，保证结果与输入顺序无关
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def _split_mesh(mesh, angle_threshold_deg):
    """按平滑组拆分顶点。

    返回 ``(positions, normals, indices, hard_edges)``：

    * 同一个顶点上，只有通过“软边”（相邻面法线夹角 <= 阈值）相连
      的面才共享一个输出顶点，其法线为这些面的面积加权平均；
    * 硬边两侧的面各自成组，使用各自的面法线，外扩时不会互相拉扯。
    """
    threshold = float(angle_threshold_deg)
    if not 0.0 <= threshold <= 180.0:
        raise ValueError("angle_threshold_deg must be within [0, 180]")
    positions = mesh["positions"]
    indices = mesh["indices"]
    cos_threshold = math.cos(math.radians(threshold))

    face_normals = []
    face_area_normals = []
    for a, b, c in indices:
        n = _face_normal(positions[a], positions[b], positions[c])
        area_n, _ = _triangle_area_normal(positions[a], positions[b], positions[c])
        face_normals.append(n)
        face_area_normals.append(area_n)

    incident = defaultdict(list)
    edge_faces = defaultdict(list)
    incident_edges = defaultdict(list)
    for f, (a, b, c) in enumerate(indices):
        for v in (a, b, c):
            incident[v].append(f)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edge_faces[key].append(f)
            incident_edges[u].append(key)
            incident_edges[v].append(key)

    # 统计硬边：恰好两个相邻面且夹角超过阈值
    hard_edges = 0
    for faces in edge_faces.values():
        if len(faces) == 2:
            if _dot(face_normals[faces[0]], face_normals[faces[1]]) < cos_threshold - 1e-12:
                hard_edges += 1

    new_positions = []
    new_normals = []
    vertex_sources = []          # 新顶点索引 -> 原顶点索引
    vertex_of = {}               # (原顶点, 面) -> 新顶点索引

    for v in sorted(incident):
        faces = sorted(incident[v])
        uf = _UnionFind(faces)
        for key in incident_edges[v]:
            fs = sorted(edge_faces[key])
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    if _dot(face_normals[fs[i]], face_normals[fs[j]]) >= cos_threshold - 1e-12:
                        uf.union(fs[i], fs[j])
        groups = defaultdict(list)
        for f in faces:
            groups[uf.find(f)].append(f)
        for root in sorted(groups):
            acc = [0.0, 0.0, 0.0]
            for f in groups[root]:
                an = face_area_normals[f]
                acc[0] += an[0]
                acc[1] += an[1]
                acc[2] += an[2]
            normal = _normalize(tuple(acc))
            if normal == (0.0, 0.0, 0.0):
                normal = face_normals[groups[root][0]]
            index = len(new_positions)
            new_positions.append(positions[v])
            new_normals.append(normal)
            vertex_sources.append(v)
            for f in groups[root]:
                vertex_of[(v, f)] = index

    new_indices = []
    for f, tri in enumerate(indices):
        new_indices.append(tuple(vertex_of[(v, f)] for v in tri))

    return new_positions, new_normals, new_indices, hard_edges, vertex_sources


def hard_edge_split(mesh=1.0, angle_threshold_deg=DEFAULT_HARD_EDGE_DEG):
    """硬边按顶点法线拆分。

    ``mesh`` 可以是 ``{"positions", "indices"}`` 字典，也可以是数值
    （视为立方体棱长，默认 1）。返回拆分后的位置 / 法线 / 索引及统计：

    * ``source_vertex_count``：输入顶点数；
    * ``vertex_count``：拆分后顶点数；
    * ``split_count``：新增顶点数（硬边造成的复制量）；
    * ``hard_edges``：法线夹角超过阈值的边数。

    立方体（阈值 60 度）所有 12 条棱都是硬边，8 个角点各拆成 3 个
    顶点，故 ``vertex_count == 24``、``split_count == 16``。
    """
    data = _as_mesh(mesh)
    positions, normals, indices, hard_edges, vertex_sources = _split_mesh(
        data, angle_threshold_deg
    )
    return {
        "positions": positions,
        "normals": normals,
        "indices": indices,
        "vertex_sources": vertex_sources,
        "source_vertex_count": len(data["positions"]),
        "vertex_count": len(positions),
        "split_count": len(positions) - len(data["positions"]),
        "hard_edges": hard_edges,
        "angle_threshold_deg": float(angle_threshold_deg),
    }


# ---------------------------------------------------------------------------
# 宽度换算：像素 <-> 世界，FOV 自适应
# ---------------------------------------------------------------------------

def width_fov(
    width_px=DEFAULT_WIDTH_PX,
    *,
    fov_y_deg=DEFAULT_FOV_Y_DEG,
    screen_height=DEFAULT_SCREEN_HEIGHT,
    distance=DEFAULT_DISTANCE,
    ref_fov_y_deg=DEFAULT_FOV_Y_DEG,
):
    """按视锥把屏幕像素宽度换算为世界宽度。

    只要相机 FOV / 距离 / 分辨率变化，世界宽度就同步变化，因此
    屏幕上的像素宽度保持不变（不会在 FOV 变化时突然变细）。

    返回字典：

    * ``world_width``：``distance`` 处对应的世界宽度；
    * ``ndc_width``：归一化设备坐标下的宽度（只看像素与分辨率）；
    * ``fov_scale``：相对参考 FOV 的正切比例，用于把按参考 FOV
      制作的描边参数迁移到当前 FOV。
    """
    ref = float(ref_fov_y_deg)
    if not 0.0 < ref < 180.0:
        raise ValueError(f"ref_fov_y_deg must be within (0, 180), got {ref!r}")
    world_width = _world_width(width_px, fov_y_deg, screen_height, distance)
    screen_height = _check_positive("screen_height", screen_height)
    fov_scale = math.tan(math.radians(float(fov_y_deg)) * 0.5) / math.tan(
        math.radians(ref) * 0.5
    )
    return {
        "width_px": float(width_px),
        "screen_height": float(screen_height),
        "fov_y_deg": float(fov_y_deg),
        "distance": float(distance),
        "world_width": world_width,
        "ndc_width": 2.0 * float(width_px) / float(screen_height),
        "fov_scale": fov_scale,
    }


def thickness(
    width_px=DEFAULT_WIDTH_PX,
    *,
    screen_height=DEFAULT_SCREEN_HEIGHT,
    distance=DEFAULT_DISTANCE,
    fov_y_deg=DEFAULT_FOV_Y_DEG,
    min_px=1.0,
    max_px=64.0,
):
    """描边厚度：像素宽度钳制到 ``[min_px, max_px]`` 后给出世界尺寸。

    线宽以屏幕像素定义，但过细会闪烁、过粗会糊住物体，因此统一钳制。
    """
    min_px = _check_positive("min_px", min_px)
    max_px = _check_positive("max_px", max_px)
    if max_px < min_px:
        raise ValueError("max_px must be >= min_px")
    requested = float(width_px)
    clamped_px = min(max(requested, min_px), max_px)
    info = width_fov(
        clamped_px,
        fov_y_deg=fov_y_deg,
        screen_height=screen_height,
        distance=distance,
    )
    return {
        "requested_px": requested,
        "thickness_px": clamped_px,
        "clamped": clamped_px != requested,
        "world_width": info["world_width"],
        "ndc_width": info["ndc_width"],
    }


def metrics(
    width_px=DEFAULT_WIDTH_PX,
    *,
    distances=(1.0, 2.0, 5.0, 10.0),
    fovs=(45.0, 60.0, 90.0),
    screen_height=DEFAULT_SCREEN_HEIGHT,
):
    """验证“宽度按视锥换算”后的屏幕像素宽度恒定。

    对每个 (距离, FOV) 组合，把像素宽度换算到世界宽度，再按投影
    反算回像素宽度，理论上恒等于 ``width_px``。返回每个采样及其
    最大偏差；``constant`` 表示所有采样偏差都在浮点容差内。
    """
    if not distances:
        raise ValueError("distances must not be empty")
    if not fovs:
        raise ValueError("fovs must not be empty")
    samples = []
    max_deviation = 0.0
    for distance in distances:
        for fov in fovs:
            world = _world_width(width_px, fov, screen_height, distance)
            tan_half = math.tan(math.radians(float(fov)) * 0.5)
            recovered_px = (
                world * float(screen_height) / (2.0 * float(distance) * tan_half)
            )
            deviation = abs(recovered_px - float(width_px))
            max_deviation = max(max_deviation, deviation)
            samples.append(
                {
                    "distance": float(distance),
                    "fov_y_deg": float(fov),
                    "world_width": world,
                    "recovered_px": recovered_px,
                }
            )
    return {
        "width_px": float(width_px),
        "samples": samples,
        "max_deviation_px": max_deviation,
        "constant": max_deviation <= 1e-6 * max(1.0, abs(float(width_px))),
    }


# ---------------------------------------------------------------------------
# 几何描边路径：法线外扩
# ---------------------------------------------------------------------------

def _resolve_normals(mesh, angle_threshold_deg):
    """拆分网格并解析每个输出顶点的法线，返回 ``(split, normals, source)``。

    * 输入带合法 ``normals`` 时优先使用，并按拆分映射到输出顶点；
    * 缺失或出现零向量（退化）时，用几何面法线按平滑组重建。

    ``source`` 为 ``provided`` / ``generated`` / ``generated_degenerate``。
    """
    data = _as_mesh(mesh)
    split = hard_edge_split(data, angle_threshold_deg=angle_threshold_deg)
    provided = None
    if isinstance(mesh, dict) and mesh.get("normals") is not None:
        raw = list(mesh["normals"])
        if len(raw) != len(data["positions"]):
            raise ValueError("normals count must match positions count")
        raw = [_normalize(tuple(float(c) for c in n)) for n in raw]
        if any(n == (0.0, 0.0, 0.0) for n in raw):
            source = "generated_degenerate"
        else:
            provided = raw
            source = "provided"
    else:
        source = "generated"

    if provided is not None:
        # 拆分会把一个原顶点复制到多个输出顶点，法线必须跟着映射，
        # 否则外扩后法线与顶点错位。
        normals = [provided[src] for src in split["vertex_sources"]]
    else:
        normals = split["normals"]
    return split, normals, source


def normal_expand(
    mesh=1.0,
    *,
    width_px=DEFAULT_WIDTH_PX,
    distance=DEFAULT_DISTANCE,
    fov_y_deg=DEFAULT_FOV_Y_DEG,
    screen_height=DEFAULT_SCREEN_HEIGHT,
    angle_threshold_deg=DEFAULT_HARD_EDGE_DEG,
):
    """沿拆分后的顶点法线做外扩（inverted hull）。

    硬边顶点先按法线拆分，再各自沿法线外扩，硬边两侧不再共享被平均
    的斜法线；软边两侧共享同一输出顶点，外扩后仍然缝合。若网格自带
    顶点法线则优先使用（并经拆分映射），否则按面法线重建。

    返回外扩后的网格，以及 ``cracks``（软边两侧外扩后位置不一致的
    边数，正确实现应为 0）与 ``open_seams``（硬边数，属预期开口）。
    """
    split, normals, normal_source = _resolve_normals(mesh, angle_threshold_deg)
    world_width = _world_width(width_px, fov_y_deg, screen_height, distance)
    source_positions = split["positions"]
    indices = split["indices"]
    expanded = [
        (
            p[0] + n[0] * world_width,
            p[1] + n[1] * world_width,
            p[2] + n[2] * world_width,
        )
        for p, n in zip(source_positions, normals)
    ]

    # 同一输出边被两个面共享时，两侧端点必须来自同一个输出顶点，
    # 外扩后位置自然一致。硬边在拆分时已复制成不同顶点、边不再共享，
    # 因此共享边上的位置若不一致即为真正的裂缝。相邻面的绕序相反，
    # 比较端点位置时按无序配对。
    edge_vertices = defaultdict(list)
    for a, b, c in indices:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_vertices[(min(u, v), max(u, v))].append((u, v))
    cracks = 0
    for endpoints in edge_vertices.values():
        if len(endpoints) < 2:
            continue
        ref_u, ref_v = endpoints[0]
        ref = sorted((expanded[ref_u], expanded[ref_v]))
        for u, v in endpoints[1:]:
            got = sorted((expanded[u], expanded[v]))
            if any(
                any(abs(g - w) > _EPS for g, w in zip(got_pt, ref_pt))
                for got_pt, ref_pt in zip(got, ref)
            ):
                cracks += 1
    return {
        "positions": expanded,
        "normals": normals,
        "indices": indices,
        "vertex_count": len(expanded),
        "width_world": world_width,
        "width_px": float(width_px),
        "cracks": cracks,
        "open_seams": split["hard_edges"],
        "hard_edges": split["hard_edges"],
        "split_count": split["split_count"],
        "normal_source": normal_source,
    }


def fallback(mesh=1.0, *, angle_threshold_deg=DEFAULT_HARD_EDGE_DEG):
    """法线缺失 / 退化时的回退策略。

    硬边拆分依赖顶点法线；当网格未提供法线，或提供的法线出现零向量
    时，直接用几何面法线按平滑组重建，避免外扩方向塌陷或随机闪烁。

    返回 ``source``：``provided`` / ``generated`` / ``generated_degenerate``，
    以及规整后的单位法线。
    """
    split, normals, source = _resolve_normals(mesh, angle_threshold_deg)
    return {
        "normals": normals,
        "source": source,
        "fallback_used": source != "provided",
        "normal_count": len(normals),
        "split_count": split["split_count"],
    }


# ---------------------------------------------------------------------------
# 深度：遮挡与一致性
# ---------------------------------------------------------------------------

def occluded_outline(
    object_depth=DEFAULT_DISTANCE,
    scene_depth=None,
    *,
    depth_bias=0.0,
    depth_func=DEFAULT_DEPTH_FUNC,
):
    """描边片元参与深度测试：被墙体挡住的物体不显示描边。

    ``object_depth`` 是描边片元深度，``scene_depth`` 是深度缓冲中
    不透明场景的深度（缺省为远裁剪面）。通过统一比较函数判定描边
    是否可见；对序列输入返回逐项可见性与可见比例。

    说明：``depth_bias`` 为正时把测试深度朝相机方向挪一点（常规 Z
    是减、反转 Z 是加），用来消除描边壳与自身表面的深度冲突；它不会
    让真正被墙挡住的描边穿透——偏置量远小于景物深度差。
    """
    if scene_depth is None:
        scene_depth = DEFAULT_FAR
    bias = float(depth_bias)
    # 朝相机方向：常规 Z 减小，反转 Z 增大
    towards_camera = 1.0 if depth_func in ("greater", "greater_equal") else -1.0
    test_depth = float(object_depth) + bias * towards_camera
    if isinstance(scene_depth, (list, tuple)):
        visibility = [
            depth_compare(test_depth, d, depth_func) for d in scene_depth
        ]
        visible_ratio = (
            sum(1 for v in visibility if v) / len(visibility)
            if visibility
            else 0.0
        )
        return {
            "visible": any(visibility),
            "visibility": visibility,
            "visible_ratio": visible_ratio,
            "object_depth": float(object_depth),
            "depth_bias": float(depth_bias),
            "depth_func": depth_func,
            "depth_test": True,
            "depth_write": False,
        }
    scene_depth = float(scene_depth)
    visible = depth_compare(test_depth, scene_depth, depth_func)
    return {
        "visible": visible,
        "object_depth": float(object_depth),
        "scene_depth": scene_depth,
        "test_depth": test_depth,
        "depth_bias": float(depth_bias),
        "depth_func": depth_func,
        "depth_test": True,
        "depth_write": False,
    }


def _default_pipeline(extra_passes=0):
    """默认描边管线：只走几何描边一条路径，全部 pass 深度基准一致。"""
    passes = [
        {
            "name": "stencil_write",
            "path": "geometry",
            "depth_test": True,
            "depth_write": True,
            "depth_func": DEFAULT_DEPTH_FUNC,
            "reversed_z": False,
        },
        {
            "name": "geometry_outline",
            "path": "geometry",
            "depth_test": True,
            "depth_write": False,
            "depth_func": DEFAULT_DEPTH_FUNC,
            "reversed_z": False,
        },
    ]
    for i in range(max(0, int(extra_passes))):
        passes.append(
            {
                "name": f"geometry_outline_{i + 1}",
                "path": "geometry",
                "depth_test": True,
                "depth_write": False,
                "depth_func": DEFAULT_DEPTH_FUNC,
                "reversed_z": False,
            }
        )
    return {"passes": passes, "active_path": "geometry"}


def _pass_list(passes):
    """把 ``None`` / pass 数量 / pass 列表统一成 pass 列表。

    整数表示“总共多少个 pass”：1 只保留模板写入，2 为默认两 pass，
    大于 2 时追加额外的几何描边 pass。
    """
    if passes is None:
        return _default_pipeline()["passes"]
    if isinstance(passes, bool):
        raise TypeError("passes must be a list, int or None")
    if isinstance(passes, int):
        if passes <= 0:
            raise ValueError("pass count must be positive")
        base = _default_pipeline()["passes"]
        if passes <= len(base):
            return base[:passes]
        return _default_pipeline(extra_passes=passes - len(base))["passes"]
    if isinstance(passes, (list, tuple)):
        return list(passes)
    raise TypeError("passes must be a list, int or None")


def depth_consistency(
    passes=None,
    *,
    depth_func=DEFAULT_DEPTH_FUNC,
    reversed_z=False,
):
    """统一多 Pass 的深度比较函数与方向约定。

    ``passes`` 可以是 pass 列表、pass 数量（整数）或 ``None``（默认
    两 pass 管线）。每个 pass 的 ``depth_func`` / ``reversed_z`` 必须
    与统一基准一致，否则前后景描边的深度排序会颠倒。

    返回 ``consistent`` 以及每一条冲突描述，便于在测试/断言中定位。
    """
    if depth_func not in DEPTH_COMPARE_FUNCS:
        raise ValueError(f"unsupported depth_func {depth_func!r}")
    pass_list = _pass_list(passes)

    conflicts = []
    for index, p in enumerate(pass_list):
        if not isinstance(p, dict):
            conflicts.append(f"pass[{index}] is not a dict")
            continue
        func = p.get("depth_func", DEFAULT_DEPTH_FUNC)
        if func != depth_func:
            conflicts.append(
                f"pass[{index}] ({p.get('name', '?')}) depth_func={func!r} "
                f"!= unified {depth_func!r}"
            )
        if bool(p.get("reversed_z", reversed_z)) != bool(reversed_z):
            conflicts.append(
                f"pass[{index}] ({p.get('name', '?')}) reversed_z mismatch"
            )
        if not p.get("depth_test", True):
            conflicts.append(
                f"pass[{index}] ({p.get('name', '?')}) has depth test disabled"
            )
    return {
        "consistent": not conflicts,
        "depth_func": depth_func,
        "reversed_z": bool(reversed_z),
        "pass_count": len(pass_list),
        "conflicts": conflicts,
    }


def audit(pipeline=None):
    """审计描边管线是否满足六项修复的约束。

    检查项：描边 pass 开启深度测试、关闭深度写入；所有 pass 深度
    函数与方向一致；仅启用一条描边路径；后处理路径声明了深度重建。
    ``pipeline`` 为 ``None`` / pass 数量 / 管线字典。整数语义与
    :func:`depth_consistency` 一致（表示 pass 总数）。
    """
    if pipeline is None:
        pipeline = _default_pipeline()
    elif isinstance(pipeline, bool):
        raise TypeError("pipeline must be a dict, int or None")
    elif isinstance(pipeline, int):
        pipeline = {"passes": _pass_list(pipeline)}
    if not isinstance(pipeline, dict) or "passes" not in pipeline:
        raise ValueError("pipeline must be a dict with 'passes'")

    passes = pipeline["passes"]
    issues = []
    checks = {}

    outline_passes = [p for p in passes if "outline" in str(p.get("name", ""))]
    checks["outline_depth_test"] = bool(outline_passes) and all(
        p.get("depth_test", True) for p in outline_passes
    )
    if not checks["outline_depth_test"]:
        issues.append("outline pass must participate in depth test")

    checks["outline_depth_write_off"] = all(
        not p.get("depth_write", True) for p in outline_passes
    )
    if not checks["outline_depth_write_off"]:
        issues.append("outline pass must not write depth")

    funcs = {p.get("depth_func", DEFAULT_DEPTH_FUNC) for p in passes}
    checks["uniform_depth_func"] = len(funcs) <= 1
    if not checks["uniform_depth_func"]:
        issues.append(f"inconsistent depth functions: {sorted(funcs)}")

    reversed_z = {bool(p.get("reversed_z", False)) for p in passes}
    checks["uniform_depth_direction"] = len(reversed_z) <= 1
    if not checks["uniform_depth_direction"]:
        issues.append("inconsistent depth direction (reversed_z)")

    checks["single_path"] = len({p.get("path") for p in passes}) == 1
    if not checks["single_path"]:
        issues.append("geometry and post-process outline paths are not exclusive")

    checks["post_depth_reconstruction"] = all(
        p.get("depth_reconstruction", False)
        for p in passes
        if p.get("path") == "post"
    )
    if not checks["post_depth_reconstruction"]:
        issues.append("post-process outline must reconstruct depth")

    return {
        "passed": not issues,
        "checks": checks,
        "issues": issues,
        "pass_count": len(passes),
    }


# ---------------------------------------------------------------------------
# 路径互斥 / 模板
# ---------------------------------------------------------------------------

def dual_path(use_geometry=1, use_post=None):
    """几何描边与后处理描边两条路径互斥，避免叠加成双线。

    ``use_geometry`` 与 ``use_post`` 同时为真时抛 ``ValueError``；
    只给一个参数时，另一条路径自动关闭。两条都关闭时返回 ``none``。
    """
    geometry = bool(use_geometry)
    if use_post is None:
        post = not geometry
    else:
        post = bool(use_post)
    if geometry and post:
        raise ValueError(
            "geometry and post-process outline paths are mutually exclusive; "
            "enable exactly one"
        )
    if geometry:
        path = "geometry"
    elif post:
        path = "post"
    else:
        path = "none"
    return {
        "path": path,
        "geometry": geometry,
        "post": post,
        "mutually_exclusive": True,
    }


def stencil(
    stencil_ref=1,
    *,
    mask=0xFF,
    compare="not_equal",
    write_value=1,
    write_mask=0xFF,
):
    """模板描边配置：先写模板，再只在模板不匹配处绘制外扩描边。

    ``stencil_ref`` / ``mask`` / ``write_*`` 为 0..255 的整数；
    ``compare`` 取图形 API 的常见比较函数。返回的 ``order`` 明确
    模板写入必须先于描边测试，防止重叠物体互相覆盖。
    """
    for name, value in (
        ("stencil_ref", stencil_ref),
        ("mask", mask),
        ("write_value", write_value),
        ("write_mask", write_mask),
    ):
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
        if not 0 <= int(value) <= 255:
            raise ValueError(f"{name} must be within [0, 255]")
    if compare not in STENCIL_COMPARE_FUNCS:
        raise ValueError(
            f"unsupported stencil compare {compare!r}, "
            f"expected one of {STENCIL_COMPARE_FUNCS}"
        )
    return {
        "stencil_ref": int(stencil_ref),
        "mask": int(mask),
        "compare": compare,
        "write_value": int(write_value),
        "write_mask": int(write_mask),
        "order": "write_then_test",
        "depth_write": False,
        "depth_test": True,
    }


# ---------------------------------------------------------------------------
# 后处理描边：按深度重建
# ---------------------------------------------------------------------------

def _synthetic_depth_buffer(size=8):
    """构造一个小的合成深度缓冲：背景深处有一个近处方块。

    深度值为 [0, 1] 的非线性窗口深度，便于演示“先重建再检测”。
    """
    if size < 3:
        raise ValueError("buffer size must be at least 3")
    background = 0.98   # 约等于远处墙面
    foreground = 0.40   # 近处方块
    lo = size // 4
    hi = size - lo
    buffer = [[background for _ in range(size)] for _ in range(size)]
    for y in range(lo, hi):
        for x in range(lo, hi):
            buffer[y][x] = foreground
    return buffer


def _dilate(mask, radius):
    """方形结构元的二值膨胀，半径以像素计，保持线宽在屏幕空间恒定。"""
    if radius <= 0:
        return [row[:] for row in mask]
    height = len(mask)
    width = len(mask[0])
    out = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            for yy in range(max(0, y - radius), min(height, y + radius + 1)):
                for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                    out[yy][xx] = 1
    return out


def screen_space(
    radius=1,
    *,
    depth_buffer=None,
    normals_buffer=None,
    near=DEFAULT_NEAR,
    far=DEFAULT_FAR,
    depth_threshold=0.05,
    normal_edge_threshold=0.9,
    dilate=True,
):
    """后处理描边：先按视锥重建深度，再做屏幕空间边缘检测与膨胀。

    直接在非线性深度上取差值会在近/远裁剪面附近误判；这里先把
    ``[0, 1]`` 窗口深度线性化，再用“相对深度差 > 阈值”判定深度
    边缘；若给出法线缓冲，相邻法线点积低于阈值也判为边缘（覆盖
    硬边/斜面折角）。最后按像素半径膨胀，得到屏幕空间恒定线宽。

    参数 ``radius`` 为膨胀半径（像素）。返回边缘掩码、边缘数量与
    重建信息；未提供 ``depth_buffer`` 时使用内置合成缓冲。
    """
    if isinstance(radius, bool) or int(radius) != radius:
        raise ValueError("radius must be an integer pixel count")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if depth_buffer is None:
        depth_buffer = _synthetic_depth_buffer()
    if not depth_buffer or not depth_buffer[0]:
        raise ValueError("depth_buffer must be a non-empty 2D sequence")
    height = len(depth_buffer)
    width = len(depth_buffer[0])
    if any(len(row) != width for row in depth_buffer):
        raise ValueError("depth_buffer must be rectangular")

    linear = [
        [linearize_depth(d, near=near, far=far) for d in row]
        for row in depth_buffer
    ]

    normals = None
    if normals_buffer is not None:
        if len(normals_buffer) != height:
            raise ValueError("normals_buffer must match depth_buffer size")
        normals = [[_normalize(tuple(map(float, n))) for n in row] for row in normals_buffer]
        if any(len(row) != width for row in normals):
            raise ValueError("normals_buffer must match depth_buffer size")

    raw_mask = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            la = linear[y][x]
            depth_edge = False
            if x + 1 < width:
                lb = linear[y][x + 1]
                denom = max(min(la, lb), _EPS)
                if abs(la - lb) / denom > depth_threshold:
                    depth_edge = True
            if not depth_edge and y + 1 < height:
                lb = linear[y + 1][x]
                denom = max(min(la, lb), _EPS)
                if abs(la - lb) / denom > depth_threshold:
                    depth_edge = True
            normal_edge = False
            if normals is not None:
                na = normals[y][x]
                if x + 1 < width and _dot(na, normals[y][x + 1]) < normal_edge_threshold:
                    normal_edge = True
                elif y + 1 < height and _dot(na, normals[y + 1][x]) < normal_edge_threshold:
                    normal_edge = True
            if depth_edge or normal_edge:
                raw_mask[y][x] = 1

    mask = _dilate(raw_mask, radius) if dilate else raw_mask
    edge_count = sum(sum(row) for row in raw_mask)
    return {
        "mask": mask,
        "raw_mask": raw_mask,
        "edge_count": edge_count,
        "dilated_count": sum(sum(row) for row in mask),
        "radius": radius,
        "buffer_size": (width, height),
        "reconstruction": "linearized_depth",
        "depth_threshold": float(depth_threshold),
        "normal_edge_used": normals is not None,
        "linearized_depth": linear,
    }


# ---------------------------------------------------------------------------
# 校验 / 确定性 / 透明
# ---------------------------------------------------------------------------

def validation(
    width_px=DEFAULT_WIDTH_PX,
    *,
    fov_y_deg=DEFAULT_FOV_Y_DEG,
    screen_height=DEFAULT_SCREEN_HEIGHT,
    distance=DEFAULT_DISTANCE,
    angle_threshold_deg=DEFAULT_HARD_EDGE_DEG,
    depth_func=DEFAULT_DEPTH_FUNC,
    near=DEFAULT_NEAR,
    far=DEFAULT_FAR,
):
    """描边参数合法性校验，返回 ``valid`` 与全部错误信息。

    非法值不会被静默钳制：零宽度、越界 FOV、错误深度函数、近远裁剪
    面倒置等都直接报告，避免参数错误在渲染时表现为难以排查的视觉
    缺陷。
    """
    errors = []

    def check_positive(name, value):
        try:
            if not float(value) > 0.0:
                errors.append(f"{name} must be positive")
        except (TypeError, ValueError):
            errors.append(f"{name} must be a number")

    check_positive("width_px", width_px)
    check_positive("screen_height", screen_height)
    check_positive("distance", distance)
    check_positive("near", near)
    check_positive("far", far)
    try:
        if not 0.0 < float(fov_y_deg) < 180.0:
            errors.append("fov_y_deg must be within (0, 180)")
    except (TypeError, ValueError):
        errors.append("fov_y_deg must be a number")
    try:
        if not 0.0 <= float(angle_threshold_deg) <= 180.0:
            errors.append("angle_threshold_deg must be within [0, 180]")
    except (TypeError, ValueError):
        errors.append("angle_threshold_deg must be a number")
    if depth_func not in DEPTH_COMPARE_FUNCS:
        errors.append(f"depth_func must be one of {DEPTH_COMPARE_FUNCS}")
    try:
        if float(far) <= float(near):
            errors.append("far must be greater than near")
    except (TypeError, ValueError):
        pass

    return {
        "valid": not errors,
        "errors": errors,
        "params": {
            "width_px": float(width_px) if _is_number(width_px) else width_px,
            "fov_y_deg": fov_y_deg,
            "screen_height": screen_height,
            "distance": distance,
            "angle_threshold_deg": angle_threshold_deg,
            "depth_func": depth_func,
            "near": near,
            "far": far,
        },
    }


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _stable_unit(seed, index):
    """与进程无关的确定性伪随机数，落在 [0, 1)。

    不能用内置 ``hash()``（受 PYTHONHASHSEED 影响）或全局随机数：
    描边抖动/采样顺序必须跨帧、跨进程一致。
    """
    digest = hashlib.sha256(f"{int(seed)}:{int(index)}".encode("ascii")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(1 << 64)


def determinism(seed=1, *, count=1, previous=None):
    """确定性：同样的种子与序号永远得到同样的抖动序列。

    返回 ``offsets``（以像素为中心的 [-0.5, 0.5) 抖动）、``stable``
    （与上一帧结果一致）以及 ``hashing`` 方法说明。使用 SHA-256 而
    非内置 ``hash()``，跨进程结果稳定。
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    offsets = [round(_stable_unit(seed, i) - 0.5, 12) for i in range(count)]
    if previous is not None:
        prev = list(previous)
        stable = len(prev) == len(offsets) and all(
            abs(a - b) <= 1e-12 for a, b in zip(offsets, prev)
        )
    else:
        stable = True
    return {
        "seed": seed,
        "offsets": offsets,
        "stable": stable,
        "previous_provided": previous is not None,
        "hashing": "sha256",
    }


def transparent(alpha=1.0, *, depth_write=None):
    """透明物体的描边参与方式。

    透明片元不写深度、也不作为遮挡深度来源；但只要有覆盖率
    （alpha > 0）就参与描边掩码。显式传入 ``depth_write`` 时按其
    覆盖默认策略，便于测试指定行为。返回覆盖率与深度写入策略。
    """
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        raise ValueError("alpha must be a number within [0, 1]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    opaque = alpha >= 1.0
    contributes = alpha > 0.0
    if depth_write is None:
        writes = opaque
    elif isinstance(depth_write, bool):
        writes = depth_write
    else:
        raise ValueError("depth_write must be a bool or None")
    return {
        "alpha": alpha,
        "opaque": opaque,
        "contributes_outline": contributes,
        "writes_depth": writes,
        "depth_test_source": "opaque_only",
        "blend": "premultiplied_alpha" if not opaque else "opaque",
    }
