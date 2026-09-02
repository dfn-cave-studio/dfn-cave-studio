# DFN Cave Studio v0.11.0-M11.1 中文使用说明书

> 中文为说明主体，界面控件保留英文原名，便于按界面查找。适用版本：已发布的 `v0.11.0-M11.1`（M11.1 Exact Second Voxelization）。

## 1. 软件用途与版本边界

DFN Cave Studio 用于从钻孔数据库建立可审计、可复现的离散裂隙网络（DFN）模型。当前版本支持：钻孔数据维护与清洗、Calibration/Validation 留出、结构域与节理组、M9 输入参数体素场、M10 多尺度显式 DFN、M11.1 精确圆盘—体素面积求交和三维 P32 显示。

当前版本**尚不支持**裂隙—裂隙求交图、连通簇、边界贯通、渗流、块体切割/块度、力学分析、Kriging 或正式 3DEC/PFC 导出。菜单中出现历史或未来入口，不等于相应科学功能已经完成。

## 2. 安装、启动与语言

在项目根目录创建 Python 3.12+ 环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m dfn_cave_studio.app
```

也可安装包后运行 `dfn-cave-studio`。首次启动时，中文系统默认简体中文，其他系统默认 English。通过 `Settings → Language → 简体中文 / English` 切换；偏好保存于 QSettings，不写入 `.dfnproj`。计算或保存期间语言切换会被暂时禁用。切换语言不重算模型、不改变项目 dirty 状态，也不改变内部字段名、图层 ID、随机种子或科学数组。

## 3. 主窗口

- `File`：`New Project`、`Open Project…`、`Save Project`、`Save Project As…`、`Recent Projects`。
- `Data → Borehole Database…`：项目级钻孔数据库。
- `Voxel → Voxel Settings…`：体素网格设置入口。
- `DFN → Joint Set Manager…`、`Explicit DFN Generation…`。
- `Visualization`：重置及标准相机方向。
- 左侧 `Project` 与工作流：按 1–13 步显示依赖状态。
- 中央 3D 视窗：钻孔、边界、M9/M10/M11 session 图层。
- 右侧 `Properties` 或 `M11 Visualization`；底部 `Log` 和进度。

图层、色标、交互平面和 Box Widget 是会话显示状态；保存项目不会序列化 VTK Actor，也不会保存相机状态。

## 4. 坐标、单位和角度

| 项目 | 约定 |
|---|---|
| 坐标系 | 右手系；X=Easting，Y=Northing，Z=Elevation（向上） |
| 长度/深度/半径/体素尺寸 | m |
| 面积 | m² |
| P10、P32 | m⁻¹；P32 的维度为 m²/m³ |
| UI/导入导出角度 | degree（°）；内部三角函数使用 radian |
| dip direction | 从北顺时针，`0 ≤ dip_direction < 360` |
| fracture dip | 从水平面量起，`0 ≤ dip ≤ 90` |
| survey dip | 钻孔轨迹倾角，允许 `[-90, 90]`；负值通常表示向下 |

所有输入文件必须使用同一坐标参考和长度单位；当前版本不自动执行坐标系或单位换算。

## 5. 项目生命周期

1. `File → New Project` 建立空项目。
2. 首次点击 `Save Project` 会进入 `Save Project As…`；推荐扩展名 `.dfnproj`。
3. 保存采用原子替换；验证失败时原文件保持不变，项目仍为 dirty。
4. Autosave 每 60 秒检查一次；大型生成、渲染或保存期间会避免并发保存。
5. `Open Project…` 与 `Recent Projects` 使用同一加载流程，恢复数据库、workflow、M9/M10/M11 科学状态和空间配置。
6. New/Open 会清除旧项目的 session-only M9/M10/M11 Actor、色标和 Widget。

上游数据或科学配置改变时，只将真实依赖的步骤标为 `STALE`。仅改变透明度、切片、颜色等显示参数不会使结果失效。

## 6. 准备输入数据

支持 CSV、XLSX 和 XLS。五类钻孔数据可独立、任意顺序、分批导入：Collars、Surveys、Fractures、RQD、Domain intervals。推荐用 UTF-8 CSV，首行是唯一表头，一行一个记录；不要混合单位、坐标系或多个孔号拼写。

详细字段、别名和校验见 [数据字典与术语](DATA_DICTIONARY_AND_GLOSSARY.md)。可直接练习 `examples/m7_demo/`；M10 确定性结构面示例位于 `examples/m10_demo/deterministic_structures.csv`。`examples/m10_demo/` 不是完整 M7–M11 项目，完整流程仍需先导入五类钻孔数据并完成 M9。

### 6.1 Collars

最少提供孔号、X/Y/Z 和终孔深度。常用映射是 `hole_id/easting/northing/elevation/total_depth`，也支持 `HoleName/East/North/RL/HoleLength` 和标准字段 `borehole_id/collar_x/collar_y/collar_z/final_depth`。普通 `depth` 不会自动映射为 `final_depth`。

### 6.2 Surveys

每行是一个测斜站：孔号、`measured_depth`、`azimuth`、`dip`。未知 Collar 的记录进入 Pending，后续导入 Collar 后自动关联。轨迹按深度排序，缺少 0 m 测站时使用 Collar 方位补起点，使用 Minimum Curvature 方法插值；尾部缺站时沿最后测站方向延伸到终孔深度。

### 6.3 Fractures

最少提供孔号、测深和倾角。`dip_direction` 可为空：

- Full orientation：有 `dip_direction + dip`，可进入法向量、Fisher 和球面聚类。
- Dip only：只保留 dip，不伪造方位；仍是 Formal 密度证据，在有明确 `set_id` 时可参与相应 P10 计数，但不参与三维方向拟合或条件化。

`set_id`、aperture、filling、fracture_type、confidence 为可选信息。Aperture 不是裂隙半径，不能用于尺寸模型。

### 6.4 RQD

每行是 `[from_depth, to_depth]` 区间及 0–100 的 RQD。软件检查区间、孔深和重叠。RQD 只是辅助数据；不会直接换算为 P10、P32、RMR 或裂隙半径。

### 6.5 Domain intervals

每行给出孔号、起止测深和 `domain_id`，可选 `domain_name`。同孔区间重叠会被报告。内部公共边界采用半开区间 `[from_depth, to_depth)`，钻孔最后区间允许包含终点，以避免边界裂隙重复归属。

### 6.6 Deterministic structures

在 M10 对话框使用 `Import Deterministic CSV`。字段包括 `structure_id, center_x, center_y, center_z, dip_direction, dip, radius, structure_type`，`domain_id` 和 `set_id` 可选。它表示参数化圆盘，不是 STL/OBJ 曲面。完全域外结构面会明确警告，默认不参与当前模型显示与统计但保留导入记录；相交圆盘按 Generation Domain 裁剪，软件不会移动中心或修改半径。

## 7. 导入、预览和字段映射

1. 打开 `Borehole Database / 钻孔数据库`，点击 `Import / Append…`。
2. 选择 Data Type 和文件，点击 `Preview`。
3. 在 Mapping 表检查 `Source Field` 与 `Standard Field`。自动识别忽略大小写、首尾空格、UTF-8 BOM，并把空格/连字符规范为下划线。
4. 每个 `Standard Field` 下拉框仍可手动调整。一个源列匹配多个标准字段，或一个标准字段有多个候选列时，必须人工解决冲突。
5. 选择 `Append`、`Replace` 或 Cancel。Cancel 不修改项目；有实际提交才标记 dirty。

Preview 和 Import 使用同一映射。每条记录保存 `record_id`、原表头、规范化表头、字段映射、源文件、源行、导入时间、原值和修改历史。重复文件不会静默追加相同记录。

## 8. 钻孔数据库与数据质量

`Borehole Database` 可按数据类型、孔号及 Raw/Formal/Excluded/Pending 查看、搜索、排序、编辑、软删除和追加。四个状态含义：

- Raw：所有原始导入审计记录；原值不可静默改写。
- Formal：下游计算使用的清洗值。
- Excluded：有明确原因的排除记录，仍保留审计链。
- Pending：尚未关联 Collar 等暂不能正式分类的记录。

点击工作流 `2. 数据质量` 打开 `M8 Data Quality / 数据质量`：

1. `Re-run checks` 运行完整质量检查。
2. 按类型、Severity、Status 筛选 issue。
3. `Accept selected auto-fix` 或 `Accept all auto-fixes` 接受建议修正；不会未经同意修改 Formal 值。
4. 必要时 `Manual edit record…`，或 `Exclude record…` 并填写原因。
5. `Retry / reclassify` 尝试恢复 Pending/Excluded。
6. `Confirm all documented exclusions` 确认有明确原因的排除记录。
7. `Export JSON…` / `Export CSV…` 导出质量报告。
8. 只有 `Pending=0` 且 unresolved ERROR=0 时，`Confirm data quality complete` 才可完成。

自动建议包括 Survey azimuth `% 360`、Survey dip 限制到 `[-90,90]`，以及 fracture direction `360→0`。每个接受操作写入 ModificationEvent（before、after、issue_id、source、时间）。已确认 Excluded 是处理结果，不会永久阻止工作流。

## 9. Validation Holdout

打开 `3. Validation Holdout`：可用 `Manual`、`Random fixed seed` 或 `Stratified` 分配孔；也可用 `Mark as Calibration` / `Mark as Validation`。检查后点击 `Lock Holdout Split`。

Calibration 用于拟合方向、密度、尺寸和条件 DFN；Validation 只用于独立验证，不参与拟合或 M10 条件化。解锁或改变划分会按依赖规则使下游结果失效。

## 10. 结构域与节理组

### 10.1 Structural Domain Editor

在 `4. 钻孔结构域` 查看或维护区间。Domain 是地质结构域；不要与 `Voxel Analysis Domain` 或 `DFN Generation Domain` 的空间边界混淆。

### 10.2 Joint Set Identification

在 `5. 节理组` 选择：

- `Mode A: Use imported set_id`：按 Formal `set_id` 统计。
- `Mode B: Auto-identify (spherical K-Means)`：对 Calibration 的 Full orientation 法向量进行轴向球面 K-Means++；`n` 与 `−n` 等价，固定 seed 可复现，空簇确定性重新初始化。请求 K 组必须产生 K 个非空组；不同轴向方向不足时会明确拒绝。

Validation 和 Dip-only 记录不进入球面拟合。所有参与的 Calibration Full orientation 只分配一次；Count 总和应等于参与聚类的样本数。可在 `Joint Set Manager…` 检查/维护名称、颜色和模型参数。

## 11. 模型边界与体素网格

### 11.1 Model Boundary

打开 `6. 模型边界`。Automatic 根据完整钻孔轨迹、Survey station 和有效观测点范围加外扩距离；Manual 输入 XYZ min/max。`Check Coverage` 报告越界孔、轨迹点、观测点、各方向最大超距和受影响孔。边界确认只完成 `bounds`。

### 11.2 Voxel Grid Preview and Confirmation

打开 `7. 体素网格预览与确认`，设置独立 `dx/dy/dz`。软件按 `ceil(extent/spacing)` 计算 `nx/ny/nz`，确保完整覆盖；同时估算体素数和内存。确认后才完成 `voxel_grid`。

必须满足 `DFN Generation Domain ⊇ Voxel Analysis Domain`。Generation Domain 缓冲用于 M10 生成和裁剪；只改变它不会清空 M9 密度/尺寸。透明度、切片方向和切片位置只是预览设置。

## 12. M9 Fracture Density Model

打开 `8. Fracture Density Model`：

1. `Interval mode` 选 `Fixed length` 或 `Domain intervals`。固定区间在结构域边界处分割，每个输出区间只属于一个域，采样总长度守恒。
2. `Spatial model` 选 `Global constant` 或 `IDW`。
3. IDW 可设 `IDW power`、`Search radius`、`Min/Max neighbours`、`Distance scale X/Y/Z` 和 `Use labelled domain-global fallback`。
4. 设 `Random seed`、`Fisher Monte Carlo samples` 和低可观测阈值，点击 `Calculate P10 / P32`。

区间 P10 为裂隙计数除以有效轨迹长度。方向修正 P32 使用 Calibration 数据的 Poisson-MLE 和 Fisher 方向曝光；低可观测或方向不足会给出明确状态，不输出伪可靠值。Validation 区间会计算观测量，但不进入拟合。

## 13. M9 Fracture Size Distribution

打开 `9. Fracture Size Distribution`。只有真实 `radius`、`diameter`、`trace_length`、`mapped_length` 可以点击 `Fit real calibration measurements`；diameter 会换算为半径。没有尺寸观测时使用 `Apply user-defined size model`，选择 fixed、uniform、truncated lognormal/power law/exponential，并标为 `ASSUMED` 或 `USER_DEFINED`。

当前自动截断分布拟合是 `EXPERIMENTAL`，不是严格已验证 MLE；项目保存收敛状态、优化器消息和样本量。固定半径示例应保持 `ASSUMED`。

## 14. M9 First Voxel Parameter Field

打开 `10. First Voxel Parameter Field`，点击 `Generate parameter field`。每个体素保存 Domain、方向、Kappa、组概率和 P32 等输入参数。`Global constant` 在域内使用常量；`IDW` 采用各向异性距离、邻域数量和搜索半径。关闭 fallback 且范围内无邻居时结果为 NO_DATA，不会伪造为 0。

Cell state：

- `OUTSIDE_MODEL`：模型外。
- `NO_DATA`：没有足够信息。
- `TRUE_ZERO`：有支持证据且真实值为 0，必须与 No Data 区分。
- `MODELED_VALUE`：有建模值。
- `EXCAVATION`：开挖/掩膜外。

### M9 显示和导出

选择 Field、Axis、Slice 和 Opacity 后点击 `Render slice`。`Rendered Layers / 已渲染图层` 支持 Show/Hide、Remove、Clear Current Slice 和 Clear All M9 Layers；同一配置会替换，不会叠加。无效单元不生成显示几何，TRUE_ZERO 保留；opacity=1 时有限区域应完全遮挡后方。

`Export M9 package` 输出 P10/P32 CSV、密度和尺寸 JSON、Validation CSV/JSON，以及存在参数场时的 NPZ 与 VTI。`Save PNG` 仅保存当前截图。

## 15. M9 Validation

打开 `11. Validation`，将留出孔的观测 P10 与参数场预测比较，报告可用的 MAE、RMSE、Bias、R²/相关性；样本不足时显示 `INSUFFICIENT_VALIDATION`。这一步不得反向调整拟合。

## 16. M10 Explicit DFN Generation

打开 `12. Explicit DFN Generation` 或 `DFN → Explicit DFN Generation…`。

### 16.1 生成设置

- `Base seed`、Realizations、`CPU workers`：同一输入和 seed 可复现；实现 seed 按保存的派生策略生成。
- `Condition Calibration FULL_ORIENTATION observations`：保留穿过 Calibration 观测点的条件裂隙；Validation 和 Dip-only 不条件化。
- `Import Deterministic CSV`：加入确定性圆盘。
- `Generate Batch`：后台批量生成；Cancel 协作停止且不提交部分结果。

随机裂隙在 MODELED_VALUE 体素内按 Domain/Set 的 P32、Fisher 方向和尺寸分布生成。每体素每组的期望数为：

```text
λ = P32 × V / (π E[R²])
N ~ Poisson(λ)
```

必须使用 `πE[R²]`，不能使用 `π(E[R])²`。权威几何为 `center + normal + radius`；边界裁剪多边形仅为实际裁剪裂隙保存。

### 16.2 多尺度设置

尺寸类互斥：SMALL `R < r_sm`，MEDIUM `r_sm ≤ R < r_ml`，LARGE `R ≥ r_ml`。

- `Threshold Mode: Auto`：用尺寸分布的面积加权 CDF；默认目标贡献约 10%/60%/30%。这是数值分辨率建议，不是地质分类标准。
- `Manual`：用户给定两个半径阈值，必须 `0 ≤ r_sm < r_ml`。
- 默认 Generate MEDIUM/LARGE，SMALL 不显式生成。未显式生成的可建模贡献进入逐体素逐组 `P32_subgrid`，不会被删除。
- 方向无效的目标进入 `p32_unresolved_orientation`，不伪装为显式或 subgrid。

目标预算满足：

```text
P32_target = P32_explicit_target + P32_subgrid + P32_unresolved_orientation
```

Poisson 单次实现会波动；软件不通过修改最后一条裂隙强行等于目标。

### 16.3 显示与导出

`All Fractures – LOD` 使用 GPU glyph/LOD 显示全部显式裂隙而非科学抽样；`Exact Geometry` 只适合选定范围或较小数量。`Color By` 支持 Joint Set、Domain、Source、Size Class；类别显隐只影响显示。DFN 图层清理不会删除钻孔、边界或 M9 图层。

选择 realization 后点击 `Export Selected`，输出：

- `fractures.csv`、`conditioned_fractures.csv`、`deterministic_structures.csv`；
- `realization_summary.json`、`realization_summary.csv`；
- `fractures.npz`；
- `fractures.vtp`。

这些是通用审计格式，不是正式 3DEC/PFC 文件。

仓库审计基准曾在特定机器、固定项目与配置下测得约 721,786 条裂隙的生成、保存、重开和 LOD 数据；这些数值只用于版本审计，不是对其他硬件、网格、尺寸分布或项目的性能保证。实际运行前始终以对话框的当前 Expected Count、final/peak memory、render buffer 和 save temporary space 估算为准。

## 17. M11.1 Exact Second Voxelization

打开 `13. Exact Second Voxelization`，选择 M10 realization 并启动计算。算法对参数化圆盘与候选体素 AABB 求真实面积交；点接触和线接触面积为 0。内部共享面采用半开所有权，外边界包含，避免重复归属。结果以稀疏三列保存：`fracture_ordinal`、`voxel_flat_index`、`intersection_area`。

```text
P32_explicit_intersection = Σ(圆盘与体素真实相交面积) / 体素体积
P32_total = P32_explicit_intersection + P32_subgrid
```

`p32_unresolved_orientation` 单独保留，**不**混入 `P32_total`。M9 的 P32 是输入参数场/中心归属预估；M11.1 是几何相交回算，两者含义不可互换。Cancel 只取消当前任务，不提交半成品，不覆盖已有有效结果。

## 18. M11 Visualization

右侧 `M11 Visualization` 支持 Field：`P32 explicit intersection`、`P32 subgrid`、`P32 total`；可选 All Joint Sets 或单组。

### 18.1 显示模式

- `Voxel Cells`：一个有效体素一个颜色，是默认审计视图。
- `Outer Surface Cloud`：从有效体素体积提取真实外围面，不是包围盒。
- `Orthogonal Sections`：X/Y/Z 固定法向切片，滑块、数值框和 3D 手柄双向同步。
- `Arbitrary Section`：输入 Origin 和归一化 Normal；拒绝零向量/非有限值，可平移旋转。
- `Plane Cutaway`：保留平面一侧，`Flip Side` 切换；从有效体积裁剪后提取外围和新暴露内部面。
- `Display Clipping Box — retains box interior`：保留 Box 内有效体积。`Reset to Model Bounds` 使用权威体素 edge；`Snap Box to Voxel Faces` 默认开启；Box Widget 与六个数值框双向同步。

Box/Cutaway 只是显示裁剪，不改变科学计算范围。NO_DATA、OUTSIDE_MODEL、EXCAVATION 和非有限字段不生成几何；TRUE_ZERO 是有效科学值，必须显示。

### 18.2 Exact、Smooth、色标和网格

- `Exact Cell Colours`：直接使用 cell scalar，是权威体素显示。
- `Smooth Display`：先应用有效 mask，再把 cell scalar 转为 point scalar；仅为视觉插值，不修改结果，不是 Kriging。
- `Global Range` 在同字段/同组的不同图层间复用全局范围；`Manual Range` 必须 min < max。
- `Continuous Gradient` 或 `Discrete Bands`；非有限值通过不生成几何保持空白。
- `Show Grid Lines` 仅影响显示；关闭时不把 VTK 裁剪三角线当作体素网格。

### 18.3 图层管理

`M11 Rendered Layers` 支持 Visible、Opacity、Show/Hide、Remove、`Clear Current View`、`Clear All M11`（计算对话框中的同类按钮标为 `Clear All M11 Layers`）。实时 Slice/Cutaway/Box 使用固定 slot 替换；`Keep Snapshot` 才新增快照。同一配置重复渲染不会累积 Actor 或色标。删除最后一个共享色标的可见图层后，相应 M11 色标才删除；其他模块的 Actor/色标不受影响。

仓库中的 162,150 体素云图数据是 CPU-side VTK geometry/no-op plotter 基准，不代表真实 OpenGL 帧率。显卡驱动、屏幕分辨率、可见图层和网格线都会影响交互体验。

## 19. 保存、重开和结果失效

`.dfnproj` 保存数据库 Raw/Formal/Excluded/Pending、provenance、holdout、workflow、空间设置、M9 参数与数组、M10 columnar 几何和质量预算、M11 稀疏求交及 P32 数组。保存前会验证 M10 嵌套配置；非法 Auto share 不会覆盖旧项目文件。

以下是典型失效关系：

- 仅重新打开并原样确认 Voxel Grid：不失效 M9。
- 改 Analysis Domain、dx/dy/dz 或 mask：Parameter Field 和 Validation 变 STALE；不清空 P10/P32 密度拟合或尺寸模型。
- 只改 Generation Domain：影响未来 M10/M11，不清空 M9。
- 重新生成 M10：旧 M11 结果和 M11 session 图层失效。
- 改透明度、颜色、切片、语言：不使科学结果失效，不设置 dirty。

## 20. 从原始数据到 M11.1 的完整操作示例

1. New Project，先导入 `examples/m7_demo/surveys.csv`，确认 Pending 可见。
2. 导入 `collars.csv`，确认 Pending 自动关联；再独立导入 `fractures.csv`、`rqd.csv`、`domain_intervals.csv`。
3. 打开 Data Quality，运行检查，接受安全自动修正，确认有原因的 Excluded，完成质量确认。
4. 建立并锁定 Validation Holdout。
5. 检查 Domain intervals；用 Mode A 或 Mode B 识别 Joint Sets。
6. 确认 Model Boundary 和 Voxel Grid；先使用较粗网格估算内存。
7. 计算 M9 Density；建立 Size Distribution；生成 First Voxel Parameter Field；运行 Validation。
8. 在 M10 选择 seed、workers、多尺度阈值；可导入 `examples/m10_demo/deterministic_structures.csv`；检查预计数量/内存后 Generate Batch。
9. 保存 `.dfnproj` 并重开，核对 realization、数组和 workflow。
10. 运行 M11.1 Exact Second Voxelization，检查 conservation 和 per-set/aggregate P32。
11. 在 M11 Visualization 检查 Voxel Cells、Surface、Sections、Cutaway 和 Box Cutaway。
12. 保存项目；如需交换数据，使用 M9 package 和 M10 `Export Selected`。当前没有 M11 独立文件导出器。

演示文件用于操作验证，不代表真实地质参数，也不能直接作为生产模型结论。

## 21. 常见问题

| 现象 | 检查与处理 |
|---|---|
| 导入提示 Missing required fields | 在 Preview 检查 Mapping；确认没有把普通 depth 当 final_depth；解决候选冲突。 |
| Survey 是 Pending | 先或随后导入同孔 Collar；再 Retry/reclassify。 |
| Data Quality 无法完成 | 必须 Pending=0 且 unresolved ERROR=0；有原因且已确认的 Excluded 不阻止。 |
| Joint Set 少于请求 K | 有效且不同的 Full orientation 轴向数量必须至少为 K；Dip-only 不参与聚类。 |
| P32 为 NO_DATA | 检查方向模型、Calibration 支持、IDW 半径/邻居、Domain 和 cell state。不要把 NO_DATA 当 0。 |
| M10 数量或内存很大 | 检查 P32、体素体积、E[R²]、显式尺寸类、realizations/workers；先用估算表。 |
| M10 Auto share 报错 | 必须 `0 < small share < cumulative medium/large share < 1`；Manual 模式检查半径阈值。 |
| Cancel 后正在停止 | 后台与子进程在安全批次边界协作退出；确认结束前不能再次计算。 |
| opacity=1 仍有空洞 | 仅 NO_DATA/模型外/非有限区应透出；TRUE_ZERO 应遮挡。确认选中的实际图层。 |
| Box Reset 报越界 | 当前版本应精确使用 Analysis Domain 的体素 edge；若仍出现，请记录项目和边界值，不要向内缩小代替修复。 |
| 保存后图层消失 | 正常：图层和 Widget 是 session-only；科学数组和设置应恢复。 |

## 22. 科学解释限制

- P10 是线密度，P32 是面密度；二者不能按固定常数互换。
- RQD、RMR、joint spacing、P10、P32 是不同概念；本版本不计算 RMR，也不从 RQD 直接推导 P32。
- M9 P32 是基于钻孔、方向修正和空间模型的输入场；M11.1 P32 是显式圆盘真实相交面积回算。
- `P32_subgrid` 是未显式生成但保留的可建模小尺度目标；`p32_unresolved_orientation` 是方向未知预算，不能相加冒充 `P32_total`。
- 相邻体素都有裂隙不代表裂隙网络连通；高 P32 不代表必然可崩。
- Smooth Display、IDW 和未来 Kriging 含义不同；当前没有 Kriging。
- 单次 Poisson realization 的 realized P32 会围绕 target P32 波动；固定 seed 只保证复现，不消除不确定性。
- 输出必须结合数据覆盖、量测偏差、结构域假设、尺寸来源和 Validation 结果解释。

## 23. 相关文档

- [数据处理流程](DATA_PROCESSING_WORKFLOW.md)
- [数据字典与术语](DATA_DICTIONARY_AND_GLOSSARY.md)
- [科学规范](../SCIENTIFIC_SPEC.md)
- [M11.1 审查材料](review/v0.11.0-M11.1/)
- [GitHub Release](https://github.com/dfn-cave-studio/dfn-cave-studio/releases/tag/v0.11.0-M11.1)
