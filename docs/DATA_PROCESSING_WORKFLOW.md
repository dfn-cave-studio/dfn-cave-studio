# 数据处理流程（多类型观测第一阶段）

```text
CSV/XLSX
  → 预览与字段映射
  → 选择fracture观测模式/间距单位
  → BoreholeDatabase Raw（原值与来源）
  → Formal / Excluded / Pending
  → 数据质量与显式结构域区间关联
  ├─ 全局完整产状/旧GLOBAL_DIP_ONLY → 既有M8–M11路径
  ├─ 区间间距 → 沿孔P10派生记录（不合成裂隙）
  ├─ 孔轴夹角 → BOREHOLE_RELATIVE定位记录（待后续方向补全）
  ├─ RQD/RMR → 既有通用标量场IDW/Kriging路径
  └─ 空间产状点 → 显示与标准访问接口（保持UNASSIGNED）
```

所有区间深度都是measured depth。空间坐标必须由collar和survey轨迹计算，不能把测深当作Z。跨结构域区间保留原始记录，并附加多个半开关联段；关联段不是新增独立观测。第一阶段的RQD/RMR标量拟合与验证会保守排除跨多个结构域或含未分配段的数据库样本，并在结果provenance报告原因；不会将其放入`domain_id=None`拟合组，也不会按中点强行归域。取消或失败回滚本次数据库、RMR/RQD标量样本投影及被暂时失效的结果。

第一阶段实现导入、审计、存储、显示表格和标准访问接口。Phase 2A 另外提供一个独立的“钻孔裂隙随机补全”入口：

```text
P普通代表产状 + Z摄像代表产状
  → 轴向极点加权聚类（K由用户指定）
  → local_set_id映射到稳定global_set_id
P普通组/RANDOM间距
  → 按P测点选择空间邻点（每个入选点展开全部局部分量）
  → contribution=(1/spacing)×distance_kernel
  → 局部分量概率归一化并向Global Set聚合
钻孔区间平均间距 + survey轨迹
  → N ~ Poisson(L/S)
  → 半开区间内均匀测深
  → 真实轨迹XYZ + 方向抽样
  → 独立列式BoreholeFractureRealization（多实现）
```

Z记录只约束方向，不参与强度。没有RANDOM行表示`NOT_REPORTED`，不是随机裂隙为零；只有用户明确确认才成为`REPORTED_ABSENT`。没有附近P强度支持的区间会被阻止并报告，不使用默认等概率。Phase 2A结果不写回Formal observations、不自动启动M9；将确认实现送入M9的适配器、原始方向补全、连通性和块度仍未实现。
