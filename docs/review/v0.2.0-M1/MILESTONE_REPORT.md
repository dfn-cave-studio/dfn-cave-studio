# Milestone M1 Report — v0.2.0-M1

**Date:** 2026-08-04
**Commit:** `43ec169`
**Tests:** 247 passed
**Coverage:** 82%
**Scientific:** 16/16 passed

## Completed Features

- Borehole/Collar/Survey/FractureObservation/RQDInterval模型
- BoreholeSurvey: 最小曲率法3D轨迹计算
- BoreholeCollection: CSV导入/重复检测/角度验证/质量报告
- StructuralDomain: 全局/盒/多边形/断层缓冲边界
- StructuralDomainCollection: 优先级空间点查域
- RockMask: 盒/面基掩膜; ExcavationMask: 开挖掩膜
- SurfaceModel: 三角网重心插值elevation_at()
- FractureMechanicalProperties: Mohr-Coulomb/刚度/剪胀
- MechanicalPropertyTemplate/Library: 优先级多条件匹配
- Project模型: 统一容器JSON序列化+版本迁移
- ProjectStore: 原子保存/备份/自动保存/脏标记
- RecentProjectsManager: 持久化最近项目(最多20)
- MainWindow: 项目CRUD集成(新/开/存/另存)/未保存提示
- 项目树动态显示 + 示例数据(5钻孔/14裂隙/1示例项目)
- 63个新测试(共200个通过)

## Incomplete Features

- DFN 3D渲染(M2)

## Known Issues

- 钻孔CSV导入仅模型层无UI对话框
- SurfaceModel大网格未优化
- 无HDF5支持

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
git checkout v0.2.0-M1
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
pytest tests/ -v
```