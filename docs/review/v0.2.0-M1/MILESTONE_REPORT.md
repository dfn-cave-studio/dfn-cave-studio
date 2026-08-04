# Milestone M1 Report

**Version:** v0.2.0-M1
**Date:** 2026-08-04
**Status:** COMPLETE

## Completed Features

- Borehole/Collar/Survey/FractureObservation/RQDInterval模型
- BoreholeSurvey: 最小曲率法轨迹计算
- BoreholeCollection: CSV导入/重复检测/角度验证/质量报告
- StructuralDomain: 全局/盒/多边形/断层缓冲 边界类型
- StructuralDomainCollection: 优先级空间查询
- RockMask: 盒/面基掩膜 含点/AABB测试
- ExcavationMask: 开挖区域掩膜
- SurfaceModel: 三角网 重心插值elevation_at()
- FractureMechanicalProperties: 莫尔-库仑/刚度/剪胀
- MechanicalPropertyTemplate/Library: 优先级多条件匹配
- Project模型: 统一容器 JSON序列化 版本迁移
- ProjectStore: 原子保存/备份/自动保存/脏标记
- RecentProjectsManager: 持久化最近项目
- MainWindow: 项目CRUD(新建/打开/保存/另存)/未保存提示
- 项目树动态显示
- 示例数据(5钻孔/14裂隙/1示例项目)
- 63个新测试(共200个)

## Test Results

- **Total:** 200
- **Passed:** 200
- **Failed:** 0
- **Coverage:** 79.0%

## Known Issues

- 钻孔CSV导入仅模型层无UI对话框
- SurfaceModel高程查询大网格未优化
- 无HDF5支持

## How to Run

```bash
cd d:/claudeprj
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python -m dfn_cave_studio.app
```
