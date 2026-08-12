# DFN Cave Studio v0.9.0-M9 使用说明书

## M9 新增操作（步骤 8–11）

M9 在 M8 七步流程之后增加四步。开始前确认数据质量完成、Validation Holdout 已锁定、Formal 裂隙/结构域/节理组可用，且 Voxel Analysis Domain 与 dx/dy/dz 已确认。Validation 孔只用于最后评价。

1. `Fracture Density Model`：选择固定长度或结构域区间、GLOBAL_CONSTANT/IDW 和随机种子，点击 `Calculate P10 / P32`。半开区间避免边界重复计数；低可观测性不输出不稳定 P32。
2. `Fracture Size Distribution`：只有真实 radius/diameter/trace_length/mapped_length 才允许自动拟合；否则设置 FIXED、UNIFORM 或三种截断分布并保留 ASSUMED/USER_DEFINED 来源。当前自动截断分布拟合必须视为 EXPERIMENTAL：截断边界使用样本极值，截断对数正态尚非完整截断似然优化。界面和项目文件保存收敛状态、优化器消息和样本量。演示固定半径 2 m 始终标记为 ASSUMED。不会从 aperture、RQD 或 set_id 推导尺寸。
3. `First Voxel Parameter Field`：后台按块生成，可显示进度并取消。NO_DATA、TRUE_ZERO、OUTSIDE_MODEL 和 MODELED_VALUE 独立保存；本步骤不生成显式裂隙面。
4. `Validation`：从参数场提取预测 P32，按留出孔真实局部轨迹换算预测 P10，报告 MAE、RMSE、Bias、可用时的 R²/相关系数；样本不足显示 INSUFFICIENT_VALIDATION。

`.dfnproj` 保存 M9 设置、P10/P32、尺寸模型、压缩参数场数组、Validation 结果及工作流。可导出 P10/P32 CSV、密度/尺寸 JSON、参数场 NPZ/VTI 和 Validation CSV/JSON。本版本不含 M10 显式 DFN。

> 适用版本：v0.9.0-M9（开发完成、等待外部审查）
> 文档语言：简体中文
> 适用平台：当前以 Windows 源代码运行环境为主

## 1. 软件用途与当前范围

DFN Cave Studio 是面向地下矿山与岩体裂隙研究的离散裂隙网络（DFN）建模软件。

当前 v0.9.0-M9 包含完整 M8 基础，并增加局部 DFN 参数场和第一次体素化：

1. 建立可持续维护的项目级钻孔数据库；
2. 对钻孔数据进行质量检查、修正、排除和审计；
3. 留出独立的 Validation 钻孔；
4. 维护钻孔结构域区间并识别节理组；
5. 定义体素分析域和 DFN 生成域；
6. 检查完整钻孔轨迹和观测点是否越界；
7. 设置体素尺寸、查看网格数量与内存估算，并进行轻量三维预览；
8. 将数据库、质量问题、工作流和空间设置保存到 `.dfnproj` 项目文件。
9. 计算 Calibration 钻孔的区间 P10 和方向修正 P32；
10. 拟合真实裂隙尺寸或维护明确标注的先验尺寸模型；
11. 生成 GLOBAL_CONSTANT 或 IDW 的第一次体素化输入参数场；
12. 使用留出的 Validation 钻孔进行独立误差评价并导出 M9 结果。

M9 **不包含**以下 M10 及后续功能：条件显式 DFN、第二次体素化、正式 3DEC/PFC 输出、机器学习训练和块度分析。RQD 不会被直接换算为 P10 或 P32。

主菜单中可能仍显示部分历史功能或后续功能入口。这些入口不代表 M10—M12 科学流程已经完成；用户应以左侧 11 步工作流为主。

## 2. 坐标和单位约定

软件内部采用 SI 单位和右手坐标系：

| 项目 | 约定 |
|---|---|
| X | Easting，向东为正 |
| Y | Northing，向北为正 |
| Z | Elevation，向上为正 |
| 长度、深度、体素尺寸 | 米（m） |
| 输入/界面角度 | 度（°） |
| 方位角或倾向 | 从北顺时针，范围 `[0, 360)` |
| 测斜倾角 | 范围 `[-90, 90]` |
| 裂隙倾角 | 从水平面量起，范围 `[0, 90]` |

导入前应确认所有文件采用同一坐标参考和同一长度单位。当前 M8 不自动进行坐标系转换。

## 3. 启动软件

在项目根目录打开 PowerShell，执行：

```powershell
.\.venv\Scripts\python.exe -m dfn_cave_studio.app
```

如果尚未创建虚拟环境，可按项目 `README.md` 安装依赖后启动。

启动后主要界面包括：

- 左侧 `M9 Workflow`：M8 基础七步和 M9 四步工作流导航；
- `Project Explorer`：当前项目、钻孔、边界和体素摘要；
- `Borehole Database / 钻孔数据库`：数据库查看和维护面板；
- 中央三维视图：钻孔、边界和网格预览；
- 底部日志区：显示加载、保存和操作信息。

## 4. 项目文件

### 4.1 新建项目

选择 `File → New Project`，或按 `Ctrl+N`。

新项目尚无保存路径。第一次选择 `Save Project` 或按 `Ctrl+S` 时，软件会自动进入 `Save As`。

### 4.2 保存项目

推荐保存为：

```text
项目名称.dfnproj
```

`.dfnproj` 会保存：

- Raw、Formal、Excluded、Pending 数据；
- 来源文件、来源行号和导入时间；
- 数据质量问题及其处理状态；
- 修改历史和排除原因；
- Formal 钻孔投影；
- Validation Holdout；
- 结构域区间和节理组；
- 工作流状态；
- 模型边界、体素设置和 DFN 生成域。

普通保存会在覆盖原文件前保留 `.bak` 备份。已经保存过且存在未保存修改的项目可按配置自动保存；从未选择过保存路径的新项目不会自动保存。

### 4.3 打开项目

选择 `File → Open Project` 或按 `Ctrl+O`。支持：

- `.dfnproj`：当前推荐的 ZIP 项目格式；
- `.dfncs`：历史 JSON 项目格式。

也可从 `File → Recent Projects` 打开最近项目。v0.7.0-M7 项目在打开时会迁移到 M8 数据仓库；建议迁移后另存为 `.dfnproj`。

### 4.4 关闭软件

存在未保存修改时，关闭窗口会询问：

- `Save`：保存后关闭；
- `Discard`：放弃未保存修改并关闭；
- `Cancel`：返回软件。

## 5. M8 七步工作流

左侧工作流包含：

1. 钻孔数据库；
2. 数据质量；
3. Validation Holdout；
4. 钻孔结构域；
5. 节理组；
6. 模型边界；
7. 体素网格预览与确认。

单击工作流条目即可打开相应功能。

状态含义：

| 状态 | 含义 |
|---|---|
| `NOT_STARTED` | 尚未开始 |
| `READY` | 依赖已满足，可以执行 |
| `HAS_ISSUES` | 存在 Pending 或尚未解决的 ERROR |
| `COMPLETED` | 已完成并确认 |
| `STALE` | 上游数据已改变，需要重新检查或计算 |

已排除且具有明确原因、并经过确认的历史记录属于已处理审计结果，不会永久阻止工作流。

## 6. 第一步：钻孔数据库

### 6.1 打开数据库

可通过以下任一入口打开：

- 单击工作流 `1. 钻孔数据库`；
- 选择 `Data → Borehole Database`。

数据库面板支持：

- 按钻孔查看；
- 按数据类型筛选；
- 按 Raw、Formal、Excluded、Pending 状态查看；
- 关键词搜索；
- 表格排序；
- 查看原始来源、来源行号、当前值和排除原因；
- 独立导入、编辑和软删除记录。

### 6.2 四种数据状态

| 状态 | 说明 |
|---|---|
| Raw | 所有成功进入数据库的原始记录；原始值不静默修改 |
| Formal | 已关联且满足正式使用条件的记录；下游计算只使用这部分数据 |
| Excluded | 因无效、冲突或人工决定而排除，但仍完整保留供审计 |
| Pending | 当前找不到对应 collar 等，暂时无法关联；补充数据后可重新分类 |

`project.borehole_collection` 是 Formal 数据的下游投影，不是第二份原始数据库。

### 6.3 支持的数据表

五类数据可以独立、任意顺序、分批和多次导入，不要求一次提供全部文件。

| 数据类型 | 推荐标准字段 | 常用兼容字段 |
|---|---|---|
| `collars` | `borehole_id, collar_x, collar_y, collar_z, final_depth` | `hole_id, easting, northing, elevation, total_depth` |
| `surveys` | `hole_id, measured_depth, azimuth, dip` | `borehole_id, depth` |
| `fractures` | `hole_id, depth, dip_direction, dip, set_id` | `borehole_id, measured_depth` |
| `rqd` | `hole_id, from_depth, to_depth, rqd` | `borehole_id, rqd_value` |
| `domain_intervals` | `hole_id, from_depth, to_depth, domain_id` | `borehole_id` |

`set_id` 可为空；非空时必须能解释为整数。文件支持 CSV、XLSX 和 XLS。

### 6.4 独立导入操作

1. 在数据库面板点击 `Import / Append…`；
2. 选择 `Data type`；
3. 选择提交方式：
   - `Append`：追加新记录，完全相同的重复记录跳过；
   - `Replace formal table`：替换该类正式表，同时保留审计历史；
4. 选择数据文件；
5. 点击 `Preview`，核对前 100 行；
6. 在 `Field mapping` 中将源字段名改成标准字段名；
7. 点击 `Stage This Import`；
8. 查看 raw、formal、excluded、pending 和 duplicates 统计；
9. 可以继续暂存其他文件；
10. 点击 `OK` 提交全部暂存操作，或点击 `Cancel` 完整回滚本次对话框中的修改。

如果先导入 surveys、fractures、RQD 或 domain intervals，而对应 collar 尚不存在，记录会进入 Pending。以后导入对应 collar 后，数据库会自动尝试重新关联。

### 6.5 编辑与删除

编辑记录：

1. 在表格中选中记录；
2. 点击 `Edit selected…`；
3. 编辑 JSON 形式的当前值；
4. 确认后保存。

编辑只改变清洗后的当前值，原始值和修改历史仍保留。

删除记录采用软删除：选中记录后点击 `Delete selected…`，记录转入 Excluded，不会从 Raw 审计层消失。

## 7. 第二步：数据质量

### 7.1 打开和查看问题

单击工作流 `2. 数据质量`。窗口顶部可以按以下条件筛选：

- 数据类型；
- 严重程度；
- 问题状态。

问题表包含：钻孔编号、来源文件、来源行、字段、原值、当前值、建议值和原因。底部摘要显示：

```text
Raw | Formal | Excluded | Pending | Unresolved ERROR | Quality confirmed
```

### 7.2 自动检查范围

当前会检查：

- collars：必填字段、坐标、终孔深度；
- surveys：深度、数字类型、方位角、倾角和重复测点；
- fractures：深度、倾向、倾角和 `set_id`；
- RQD：值域、区间顺序、孔深范围及同孔重叠；
- domain intervals：深度、`domain_id`、孔深范围及同孔重叠；
- 非 collar 记录引用未知钻孔；
- Excluded 记录是否具有明确原因并得到确认。

### 7.3 自动修正

当前只有两类问题允许安全自动修正：

- survey 方位角修正为 `azimuth % 360`，使其进入 `[0, 360)`；
- survey 倾角限制到 `[-90, 90]`。

操作方式：

- 选中问题后点击 `Accept selected auto-fix`；
- 或点击 `Accept all auto-fixes` 一次性应用所有可自动修正项。

同一测站的多个字段会合并处理。每个字段都会写入包含修正前值、修正后值、问题编号、来源和时间的修改历史。重复运行检查或重复点击不会重复应用同一修正。

### 7.4 人工处理

- `Manual edit record…`：修改一条记录的当前值；
- `Exclude record…`：排除无法修正的记录，必须填写原因；
- `Retry / reclassify`：修正关联条件后尝试恢复 Excluded 或 Pending；
- `Confirm selected exclusions`：确认选中的有理由排除记录；
- `Confirm all documented exclusions`：确认全部已有明确原因的排除记录；
- `Re-run checks`：重新执行全部质量检查；
- `Export JSON…` / `Export CSV…`：导出质量报告。

### 7.5 完成数据质量

只有同时满足以下条件，才能点击 `Confirm data quality complete`：

```text
Raw > 0
Pending = 0
Unresolved ERROR = 0
```

已确认且原因明确的 Excluded 记录可以保留，不阻止完成。完成后：

```text
clean = COMPLETED
holdout = READY
```

点击质量窗口的 `OK` 只保存当前对话框操作并关闭，不会自动打开 Validation Holdout。点击 `Cancel` 会将本次对话框中的数据库、问题和工作流修改全部回滚。

## 8. 第三步：Validation Holdout

Validation Holdout 用于将钻孔划分为：

- Calibration：用于节理组识别及后续拟合；
- Validation：独立保留，不参与拟合。

操作步骤：

1. 单击工作流 `3. Validation Holdout`；
2. 选择方法：
   - `Manual selection`：手动选择；
   - `Random (fixed seed)`：按固定种子随机选择；
   - `Stratified by domain`：当前界面会提示结构域条件，并暂用随机选择；
3. 手动模式下选择钻孔，点击 `Mark as Calibration` 或 `Mark as Validation`；
4. 随机模式下设置 Validation 比例和随机种子；
5. 检查 Calibration、Validation 和 Total 数量；
6. 点击 `Lock Holdout Split`；
7. 点击 `OK` 提交。

节理组识别前必须锁定 Holdout。修改上游钻孔数据后，应重新检查拆分是否仍然有效。

## 9. 第四步：钻孔结构域

单击工作流 `4. 钻孔结构域` 打开编辑器。

可以：

- 新增或删除结构域；
- 设置结构域名称、ID、颜色和备注；
- 将某钻孔的深度区间分配给结构域；
- 删除已分配区间；
- 查看各结构域的钻孔区间。

区间必须满足：

```text
0 <= from_depth < to_depth <= borehole final_depth
```

同一钻孔的结构域区间不得重叠。M8 只维护钻孔区间约束，不生成三维结构域体。

如果项目没有 domain interval 数据，本步骤不是使用数据库、清洗或设置边界的前置条件。

## 10. 第五步：节理组识别

单击工作流 `5. 节理组`。

前提条件：

- 存在有效 Formal fractures；
- Validation Holdout 已锁定；
- 至少存在一个 Calibration 钻孔。

支持两种模式：

- `Mode A: Use imported set_id from CSV`：使用导入的 `set_id`；
- `Mode B: Auto-identify`：使用球面 K-Means 自动识别。

自动模式需设置节理组数量和固定随机种子。结果表显示：

- Set ID 和名称；
- 平均倾向、平均倾角；
- Fisher Kappa；
- Count；
- 来源。

结果摘要会分别显示 Calibration fractures used 和 Validation fractures excluded。Validation 裂隙不得参与聚类。点击 `OK` 后才将结果写入项目。

## 11. 第六步：模型边界

单击工作流 `6. 模型边界`。本步骤只确认 Voxel Analysis Domain，不同时完成体素网格步骤。

### 11.1 自动边界

选择 `Automatic from full trajectories and observations`，设置 `Automatic outward margin`。系统根据完整钻孔轨迹和有效观测点范围计算边界，再向外扩展指定距离。

### 11.2 手动边界

选择 `Manual`，输入：

```text
X min / X max
Y min / Y max
Z min / Z max
```

点击 `Check Coverage` 可查看：

- 超界钻孔数量；
- 超界轨迹点数量；
- 超界观测点数量；
- 各方向最大超出距离；
- 受影响钻孔；
- 推荐扩展范围。

如果边界不足，软件会建议扩展到推荐边界。若确实需要保留不足边界，必须勾选明确允许裁剪；系统不会静默删除或裁剪数据库记录。

点击 `OK` 后只有：

```text
bounds = COMPLETED
voxel_grid = READY
```

## 12. 第七步：体素网格预览与确认

完成模型边界后，单击工作流 `7. 体素网格预览与确认`。

### 12.1 体素尺寸

分别设置：

- `dx`：X 方向尺寸；
- `dy`：Y 方向尺寸；
- `dz`：Z 方向尺寸。

三者可以不同。系统使用 `ceil` 计算 `nx、ny、nz`，保证体素网格完整覆盖分析域。

摘要显示：

- `nx × ny × nz`；
- 总体素数量；
- 有 mask 时的有效体素数量；
- 主要字段内存估算；
- 大网格警告；
- DFN Generation Domain 范围。

体素状态区分：无数据、真实零值和模型外空间。

### 12.2 DFN 生成域缓冲

设置：

- `Voxel buffer layers`；
- `Maximum fracture radius`。

每个方向的默认缓冲依据为：

```text
max(最大裂隙半径, 缓冲层数 × 对应方向体素尺寸)
```

系统保证：

```text
DFN Generation Domain 包含 Voxel Analysis Domain
```

M8 只计算、显示并保存这个范围，不在此步骤生成条件 DFN。

### 12.3 三维预览

可设置：

- 是否显示抽样网格线框；
- 网格线透明度；
- X/Y/Z 切片方向；
- 切片位置百分比。

点击 `3D Preview` 后显示：

- Voxel Analysis Domain 外框；
- DFN Generation Domain 外框；
- 完整钻孔轨迹；
- 裂隙观测点；
- 坐标轴；
- 抽样网格线和切片；
- 醒目标记的超界数据。

预览不会创建或绘制大型网格的每一个体素边框。点击 `OK` 后才令 `voxel_grid = COMPLETED`。

## 13. 推荐的完整操作流程

```text
New Project
→ 打开钻孔数据库
→ 分批导入 collars / surveys / fractures / rqd / domain_intervals
→ 检查 Raw / Formal / Excluded / Pending
→ 打开数据质量
→ 接受安全自动修正
→ 人工处理其余 ERROR 和 Pending
→ 确认有明确原因的 Excluded
→ Confirm data quality complete
→ 设置并锁定 Validation Holdout
→ 检查或编辑钻孔结构域
→ 识别节理组
→ 确认模型边界
→ 设置并预览体素网格和 DFN 生成域
→ 保存为 .dfnproj
→ 关闭并重新打开，复核数量与工作流状态
```

建议每完成一个重要步骤后保存项目。

## 14. 演示数据

仓库中的示例文件位于：

```text
examples/m7_demo/
```

包含 collars、surveys、fractures、rqd 和 domain intervals 五类文件，可用于熟悉独立导入和数据质量流程。

当前演示数据导入后的审计基线为：

```text
Raw = 284
Formal = 272
Excluded = 12
Pending = 0
```

其中 fractures：

```text
Raw = 83
Formal = 80
Excluded = 3
```

接受安全测斜自动修正后，Formal survey stations 为 148，所有正式测站满足：

```text
0 <= azimuth < 360
-90 <= dip <= 90
```

这些数量仅是演示数据验收结果，生产代码不会依赖或硬编码这些数值。

## 15. 数据追溯原则

每条数据库记录均保留：

- 唯一 `record_id`；
- `source_file`；
- `source_row`；
- `imported_at`；
- `original_values`；
- 当前清洗值；
- Raw/Formal/Excluded/Pending 状态；
- 排除原因；
- 修改来源和修改历史。

请勿通过编辑原始 CSV 后覆盖项目的方式隐藏问题。推荐追加新批次或使用有审计记录的编辑、排除和恢复操作。

## 16. 常见问题

### 16.1 只导入 surveys 后为什么都是 Pending？

survey 需要通过 `hole_id` 关联 collar。先导入 survey 是允许的；之后导入相应 collars，系统会自动尝试关联。

### 16.2 为什么 Excluded 不会从 Raw 中消失？

Raw 是不可静默修改的审计层。Excluded 表示不参与正式计算，不表示删除原始记录。

### 16.3 为什么确认了 Excluded，数量仍然存在？

确认只表示该排除记录已有明确原因并完成审阅。它仍必须保留用于审计，但不再计入未解决 ERROR。

### 16.4 为什么不能完成数据质量？

检查底部摘要。必须满足 Pending 为 0 且 Unresolved ERROR 为 0。WARNING 不一定阻止完成，但应由用户检查其科学影响。

### 16.5 为什么节理组按钮不能得到结果？

应先锁定 Validation Holdout，并确认至少有一个 Calibration 钻孔和有效 Formal fractures。

### 16.6 修改数据库后为什么后续步骤变成 STALE？

这表示上游数据变化使已有结果可能失效。重新进入受影响步骤并确认即可。软件只应使真正依赖该数据的步骤失效。

### 16.7 为什么体素数非常大？

体素总数约为 `nx × ny × nz`。减小体素尺寸会按三个方向共同放大数量和内存需求。应先查看摘要和警告，再确认网格。

### 16.8 为什么第一次按 Ctrl+S 会弹出保存位置？

新项目尚无路径，因此普通保存会自动转入 Save As。这是正常行为。

## 17. 使用限制与科学注意事项

- Validation 钻孔必须完全排除在节理组拟合和未来 M9 参数拟合之外；
- 不应将 RQD 直接等同于 P10 或 P32；
- 高裂隙密度不等于必然可崩；
- 相邻体素存在裂隙不等于裂隙网络真实贯通；
- M8 网格预览不是显式 DFN 生成结果；
- 自动修正仅应用于明确、安全且可追溯的字段转换；
- 任何自动或人工修改都应在质量报告和项目修改历史中可追溯；
- 正式研究前应保存、重开并复核数据库计数、Holdout、节理组和空间设置。

## 18. 故障排查建议

遇到问题时依次检查：

1. 底部日志区是否有错误信息；
2. 数据库 Raw/Formal/Excluded/Pending 数量；
3. 数据质量中的 Unresolved ERROR；
4. 字段映射和数值单位是否正确；
5. `hole_id` 是否与 collars 完全一致；
6. 工作流是否显示 STALE；
7. 手动边界是否覆盖完整轨迹；
8. 项目是否已保存为 `.dfnproj`；
9. 重启软件并重新打开项目后问题是否仍然存在。

报告问题时建议同时提供：软件版本、项目文件格式、操作步骤、日志信息、相关来源文件名和 `source_row`。不要发送无法脱敏的敏感工程数据。
