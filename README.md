# pairwise-outline-expand

## 领域

游戏、图形与多媒体

## 主题

轮廓描边外扩

## 说明

硬边断裂、宽度换算与深度一致。

初始状态下 `graphics/outline_expand.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
