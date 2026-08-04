# Scientific Validation Report — v0.3.0-M2

## SV-01: 倾向倾角↔法向量往返转换

| Field | Value |
|-------|-------|
| **Input** | 8组(dip_dir,dip)测试用例 |
| **Random Seed** | N/A (deterministic) |
| **Expected** | 往返误差 < 1e-10 |
| **Actual** | 最大误差 = 2.84e-14 |
| **Absolute Error** | 2.842171e-14 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 1e-10 |
| **Passed** | ✅ YES |

## SV-02: 已知法向量值验证

| Field | Value |
|-------|-------|
| **Input** | dip_dir=0,dip=0; dip_dir=0,dip=90; dip_dir=90,dip=90 |
| **Random Seed** | N/A |
| **Expected** | [0,0,1]; [0,-1,0]; [-1,0,0] |
| **Actual** | [-0. -0.  1.]; [-0. -1.  0.]; [-1. -0.  0.] |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 1e-12 |
| **Passed** | ✅ YES |

## SV-03: Fisher方向分布统计

| Field | Value |
|-------|-------|
| **Input** | mean_dd=45°, mean_dip=30°, kappa=50, n=5000 |
| **Random Seed** | 42 |
| **Expected** | 均值偏差 < 10° |
| **Actual** | dd=45.27°(err=0.27°), dip=29.97°(err=0.03°) |
| **Absolute Error** | 2.663899e-01 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 10.0 |
| **Passed** | ✅ YES |

## SV-04: 单裂隙穿过立方体

| Field | Value |
|-------|-------|
| **Input** | center=(0.5,0.5,0.5), normal=(0,0,1), r=10, box=[0,1]³ |
| **Random Seed** | N/A |
| **Expected** | True (相交) |
| **Actual** | True |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## SV-05: 倾斜裂隙穿过多个体素

| Field | Value |
|-------|-------|
| **Input** | 45°倾角裂隙, r=3, 5×5体素网格 |
| **Random Seed** | N/A |
| **Expected** | 多个体素检测到求交 |
| **Actual** | 25个体素求交: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)] |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## SV-06: 固定种子复现性

| Field | Value |
|-------|-------|
| **Input** | master_seed=42, 1节理组, P32=1.0 |
| **Random Seed** | 42 |
| **Expected** | 两次生成结果完全一致 |
| **Actual** | fractures: 4451 vs 4451, P32: 1.003439 vs 1.003439 |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 1e-10 |
| **Passed** | ✅ YES |

## SV-07: 两裂隙求交

| Field | Value |
|-------|-------|
| **Input** | 两正交裂隙在原点相交(r=5); 两裂隙相距很远(r=1,d=17.3) |
| **Random Seed** | N/A |
| **Expected** | True (相交); False (不相交) |
| **Actual** | True (相交); False (不相交) |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## SV-08: 目标P32与实际P32

| Field | Value |
|-------|-------|
| **Input** | target_p32=1.0, 50m³模型, fixed r=3.0m |
| **Random Seed** | 42 |
| **Expected** | P32 ≈ 1.0 (±5%) |
| **Actual** | achieved_p32=1.0034, error=0.34% |
| **Absolute Error** | 3.439303e-03 |
| **Relative Error** | 0.003439 |
| **Tolerance** | 0.05 |
| **Passed** | ✅ YES |

## SV-09: 不同种子不同实现

| Field | Value |
|-------|-------|
| **Input** | master_seed=42 vs 99, 同节理组配置 (固定尺寸r=3m) |
| **Random Seed** | 42, 99 |
| **Expected** | 位置不同(不同种子) 且 P32统计特征一致(误差<2%) |
| **Actual** | seed42: 4451f, P32=1.0034; seed99: 4451f, P32=1.0034; positions_differ=True, p32_close=True |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.02 |
| **Passed** | ✅ YES |

## SV-10: 两节理组生成

| Field | Value |
|-------|-------|
| **Input** | Set1(P32=0.5)+Set2(P32=0.5), 50m³ |
| **Random Seed** | 42 |
| **Expected** | 两组裂隙均有生成, set_id正确 |
| **Actual** | Set1: 5024f, Set2: 5024f, total P32=1.0051 |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## SV-11: 体素网格世界↔索引往返

| Field | Value |
|-------|-------|
| **Input** | 20³ grid, cell=5m, origin=(10,20,30), test point (5,7,9) |
| **Random Seed** | N/A |
| **Expected** | 往返后索引完全一致 (5,7,9) |
| **Actual** | (5, 7, 9) |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## SV-12: 多边形面积计算

| Field | Value |
|-------|-------|
| **Input** | 10×10正方形; 3×4×5直角三角形 |
| **Random Seed** | N/A |
| **Expected** | 100.0; 6.0 |
| **Actual** | 100.000000; 6.000000 |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 1e-10 |
| **Passed** | ✅ YES |

## SV-13: 掩膜激活计数

| Field | Value |
|-------|-------|
| **Input** | 10³ grid(50m³), mask=25³, cell=5m |
| **Random Seed** | N/A |
| **Expected** | 部分体素激活(不是全0也不是全1000) |
| **Actual** | 125个体素激活 (共1000个) |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## SV-14: 裂隙连通性

| Field | Value |
|-------|-------|
| **Input** | 两正交相交裂隙; 两远离裂隙 |
| **Random Seed** | N/A |
| **Expected** | 1条边(连通); 0条边(不连通) |
| **Actual** | 相交: 1边, 1簇; 远离: 0边, 2簇 |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## SV-15: 连通簇分离

| Field | Value |
|-------|-------|
| **Input** | 两个远离的裂隙 |
| **Random Seed** | N/A |
| **Expected** | 2个独立簇 |
| **Actual** | 2个簇, 大小: [1, 1] |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## SV-16: 危险分辨率检测

| Field | Value |
|-------|-------|
| **Input** | 1000m³模型, 0.1m体素 → 10^9体素 |
| **Random Seed** | N/A |
| **Expected** | is_dangerous=True |
| **Actual** | is_dangerous=True, reason='1,000,000,000,000 total voxels > 1 billion — extreme resolution' |
| **Absolute Error** | 0.000000e+00 |
| **Relative Error** | 0.000000 |
| **Tolerance** | 0.0 |
| **Passed** | ✅ YES |

## Summary

- **Total cases:** 16
- **Passed:** 16
- **Failed:** 0