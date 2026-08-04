# Milestone M3 Report — v0.4.0-M3

**Date:** 2026-08-04
**Commit:** `9f3c5d7`
**Tests:** 247 passed
**Coverage:** 82%
**Scientific:** 16/16 passed

## Completed Features

- VoxelGrid: 分块(32³)多属性体素存储
- VoxelChunk: 逐块属性数组/活跃计数
- 稀疏模式: 仅写入时分配块
- 掩膜系统: active≠void≠invalid(零值不歧义)
- apply_mask(): RockMask批量应用
- 世界坐标↔体素索引互转
- 静态内存估算+危险分辨率警告
- 236个测试通过

## Incomplete Features

- DFN-体素求交(M4)

## Known Issues

- 全网格apply_mask逐体素循环效率低
- 仅支持int32属性

## Scientific Validation Results

- ✅ **SV-01**: 倾向倾角↔法向量往返转换 — 最大误差 = 2.84e-14
- ✅ **SV-02**: 已知法向量值验证 — [-0. -0.  1.]; [-0. -1.  0.]; [-1. -0.  0.]
- ✅ **SV-03**: Fisher方向分布统计 — dd=45.27°(err=0.27°), dip=29.97°(err=0.03°)
- ✅ **SV-04**: 单裂隙穿过立方体 — True
- ✅ **SV-05**: 倾斜裂隙穿过多个体素 — 25个体素求交: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
- ✅ **SV-06**: 固定种子复现性 — fractures: 4451 vs 4451, P32: 1.003439 vs 1.003439
- ✅ **SV-07**: 两裂隙求交 — True (相交); False (不相交)
- ✅ **SV-08**: 目标P32与实际P32 — achieved_p32=1.0034, error=0.34%
- ✅ **SV-09**: 不同种子不同实现 — seed42: 4451f, P32=1.0034; seed99: 4451f, P32=1.0034; positions_differ=True, p32_close=True
- ✅ **SV-10**: 两节理组生成 — Set1: 5024f, Set2: 5024f, total P32=1.0051
- ✅ **SV-11**: 体素网格世界↔索引往返 — (5, 7, 9)
- ✅ **SV-12**: 多边形面积计算 — 100.000000; 6.000000
- ✅ **SV-13**: 掩膜激活计数 — 125个体素激活 (共1000个)
- ✅ **SV-14**: 裂隙连通性 — 相交: 1边, 1簇; 远离: 0边, 2簇
- ✅ **SV-15**: 连通簇分离 — 2个簇, 大小: [1, 1]
- ✅ **SV-16**: 危险分辨率检测 — is_dangerous=True, reason='1,000,000,000,000 total voxels > 1 billion — extreme resolution'

## How to Reproduce

```bash
git checkout v0.4.0-M3
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
pytest tests/ -v
```