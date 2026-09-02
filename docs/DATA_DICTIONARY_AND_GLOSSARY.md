# DFN Cave Studio 数据字典与术语（v0.11.0-M11.1）

本文列出当前代码实际接受的钻孔/结构面字段、常见别名、验证规则和科学术语。标准字段名、枚举、JSON/NPZ key、ID 和单位不会随界面语言变化。

## 1. 通用导入规则

| 项目 | 规则 |
|---|---|
| 文件格式 | CSV、XLSX、XLS |
| 表头规范化 | 去首尾空格和 UTF-8 BOM；case-insensitive；空格和连字符转 `_`；连续 `_` 合并 |
| 一行含义 | 一条对应类型的观测或区间记录 |
| 缺失值 | 空白及常见 null token 作为缺失；数值 0 不是缺失 |
| Provenance | record_id、source_file、source_row、imported_at、原值、字段映射、状态、修改来源和 ModificationEvent |
| 数据状态 | Raw、Formal、Excluded、Pending |
| 坐标 | X=Easting，Y=Northing，Z=Elevation/RL；m |
| 角度 | UI/IO 为 degree；内部计算为 radian |

同一源列不能静默映射到两个标准字段；同一标准字段若存在多个候选列，必须人工确认。Required 检查在字段映射完成后执行。

## 2. Collars

**支持格式**：CSV/XLSX/XLS。**每行**：一个钻孔孔口及终孔信息。

| 标准字段 | 必填 | 类型/单位 | 常用别名 | 缺失与验证 | 产生对象/下游用途 |
|---|---:|---|---|---|---|
| `borehole_id` | 是 | string | `hole_id`, `holeid`, `borehole`, `HoleName`, `hole_name`, `bh_id`, `hole`, `id` | 非空；项目内用于关联 | Collar/Borehole；所有关联表 |
| `collar_x` | 是 | float, m | `easting`, `east`, `x`, `East`, `x_coord` | 有限数值 | X 坐标、轨迹、边界 |
| `collar_y` | 是 | float, m | `northing`, `north`, `y`, `North`, `y_coord` | 有限数值 | Y 坐标、轨迹、边界 |
| `collar_z` | 是 | float, m | `elevation`, `rl`, `z`, `RL`, `elev`, `z_coord` | 有限数值 | Z 高程、轨迹、边界 |
| `final_depth` | 是 | float, m | `total_depth`, `hole_length`, `depth_total`, `HoleLength`, `length`, `eoh` | `>0` | 终孔深度、区间边界 |
| `azimuth` | 否 | float, ° | `az`, `bearing`, `azm` | 缺 Survey 起点时作为初始方向 | 轨迹起始方向 |
| `dip` | 否 | float, ° | `inclination`, `incl`, `dip_angle` | Collar/Survey 语境，通常 `[-90,90]` | 轨迹起始方向 |
| `domain_id` | 否 | integer | — | 可空；不替代 Domain intervals | 辅助标签 |

普通列名 `depth` **不会**自动映射到 `final_depth`，避免把观测深度误当终孔深度。

## 3. Surveys

**每行**：一个钻孔测斜站。

| 标准字段 | 必填 | 类型/单位 | 常用别名 | 缺失与验证 | 产生对象/下游用途 |
|---|---:|---|---|---|---|
| `borehole_id` | 是 | string | `hole_id`, `bh_id` | 未知孔先 Pending | SurveyStation、关联 Borehole |
| `measured_depth` | 是 | float, m | `depth`, `md`, `downhole_depth` | `0 ≤ MD ≤ final_depth`；同孔重复 MD 报告 | Minimum Curvature 轨迹 |
| `azimuth` | 是 | float, ° | `az`, `bearing` | 质量服务建议 `%360`；用户接受后写历史 | 轨迹方向 |
| `dip` | 是 | float, ° | `inclination`, `incl` | 质量服务建议限制到 `[-90,90]` | 轨迹方向 |

Survey dip 是**钻孔轴线**相对水平面的倾角，可为负；它不是 fracture dip。Formal 数据库值与实际 SurveyStation 使用值完全一致，不在投影时静默归一化。

## 4. Fractures

**每行**：一个沿钻孔测深定位的裂隙观测。

| 标准字段 | 必填 | 类型/单位 | 常用别名 | 缺失与验证 | 产生对象/下游用途 |
|---|---:|---|---|---|---|
| `borehole_id` | 是 | string | `hole_id`, `bh_id` | 未知孔 Pending | FractureObservation |
| `measured_depth` | 是 | float, m | `depth`, `md`, `from` | `0 ≤ depth ≤ final_depth` | 轨迹空间位置、P10 |
| `dip_direction` | 否 | float, ° | `dipdir`, `dd`, `dip_dir` | 可缺失；`0≤dd<360`；360 可审计修正为 0 | Full orientation/Fisher/Joint Set |
| `dip` | 是 | float, ° | `dip_angle` | `0≤dip≤90` | 倾角、方向/密度证据 |
| `set_id` | 否 | integer | `set`, `joint_set_id`, `family`, `fracture_set_id` | 可空；非整数排除/issue | Mode A、按组 P10/P32 |
| `aperture` | 否 | numeric | `aperture_mm`, `opening` | 单位必须由数据来源说明；当前不作尺寸半径 | 审计属性，不用于 R |
| `filling` | 否 | string | `infill`, `fill`, `filling_type` | 可空 | 审计属性 |
| `fracture_type` | 否 | string | `type`, `frac_type`, `discontinuity_type` | 可空 | 审计属性 |
| `confidence` | 否 | string/numeric | `certainty`, `quality` | 可空 | 审计属性 |
| `radius` | 否 | float, m | — | `>0`，真实量测才用于尺寸 |
| `diameter` | 否 | float, m | — | `>0`，读取后半径=diameter/2 |
| `trace_length` | 否 | float, m | — | `>0` | 尺寸模型候选 |
| `mapped_length` | 否 | float, m | — | `>0` | 尺寸模型候选 |

### 4.1 Full orientation 与 Dip only

- `dip_direction` 和 `dip` 都存在：FULL_ORIENTATION，可转换为轴向法向量。
- `dip_direction` 缺失但 dip 有效：DIP_ONLY，仍保留为 Formal；不会填入随机方位，不参与 Fisher、球面 K-Means 或 M10 条件化。
- 数值 `0` 是正北方向，不是缺失。

## 5. RQD intervals

**每行**：一个钻孔岩芯质量区间。

| 标准字段 | 必填 | 类型/单位 | 常用别名 | 缺失与验证 | 产生对象/下游用途 |
|---|---:|---|---|---|---|
| `borehole_id` | 是 | string | `hole_id`, `bh_id` | 未知孔 Pending | RQDInterval |
| `from_depth` | 是 | float, m | `from`, `top`, `start_depth` | `0≤from<to≤final_depth` | 区间起点 |
| `to_depth` | 是 | float, m | `to`, `bottom`, `end_depth` | 同孔重叠会报告 | 区间终点 |
| `rqd` | 是 | float, % | `rqd_value`, `rqd_pct`, `rqd_percent` | `0≤RQD≤100` | 辅助质量数据 |
| `core_recovery` | 否 | float, % | `recovery`, `rec`, `cr` | 可空；建议 0–100 | 辅助属性 |

RQD 不直接转换为 P10/P32；当前工作流也不计算 RMR。

## 6. Domain intervals

**每行**：钻孔上的一个结构域区间。

| 标准字段 | 必填 | 类型/单位 | 常用别名 | 缺失与验证 | 产生对象/下游用途 |
|---|---:|---|---|---|---|
| `borehole_id` | 是 | string | `hole_id`, `bh_id` | 未知孔 Pending | DomainInterval |
| `from_depth` | 是 | float, m | `from`, `top`, `start_depth` | `0≤from<to≤final_depth` | Domain 归属 |
| `to_depth` | 是 | float, m | `to`, `bottom`, `end_depth` | 同孔重叠会报告 | Domain 归属 |
| `domain_id` | 是 | integer | `domain`, `structural_domain_id` | 必须可解析为整数 | M9/M10 分域模型 |
| `domain_name` | 否 | string | `domain_label`, `formation` | 可空 | 显示/审计 |

相邻区间内部按 `[from,to)` 处理，最后区间可包含终点；公共边界的 fracture 只归属一次。

## 7. Deterministic structures CSV

**每行**：一个参数化确定性圆盘。

| 字段 | 必填 | 类型/单位 | 验证与语义 |
|---|---:|---|---|
| `structure_id` | 是 | string | 唯一来源标识 |
| `center_x/y/z` | 是 | float, m | 圆盘中心；不自动移动 |
| `dip_direction` | 是 | float, ° | `[0,360)`，从北顺时针 |
| `dip` | 是 | float, ° | `[0,90]`，从水平面量起 |
| `radius` | 是 | float, m | `>0` |
| `structure_type` | 是 | string | 用户来源类别 |
| `domain_id` | 否 | integer | 模型归属/审计 |
| `set_id` | 否 | integer | 保留真实组号；缺失为明确无组状态 |

该格式不代表任意曲面；当前不导入 STL/OBJ 为确定性裂隙。

## 8. 主要项目对象与数组

| 名称 | 类型/位置 | 含义 |
|---|---|---|
| BoreholeDatabase | 项目唯一审计源 | Raw/Formal/Excluded/Pending、issues、history |
| BoreholeCollection | Formal 下游投影 | 科学服务读取的 Collar/Survey/Fracture/RQD/Domain |
| P10Interval | M9 state | 孔/域/组/role 的计数、轨迹长度、P10 |
| P32Estimate | M9 state | Calibration 方向修正 P32、曝光和 eligibility |
| FractureSizeModel | M9 state | distribution、参数、bounds、source、fit status |
| Parameter field arrays | M9 state | cell_state、domain_id、总/分组 P32、方向、Kappa 等 |
| M10Realization | M10 state | config/seed/quality/provenance + columnar geometry arrays |
| `center` | `(N,3)` float | 圆盘中心 XYZ |
| `normal` | `(N,3)` float | 单位法向量，轴向等价 |
| `radius` | `(N,)` float | 圆盘半径 m |
| `size_class` | uint8 | 0 UNKNOWN、1 SMALL、2 MEDIUM、3 LARGE |
| `source_code` | uint8 | Stochastic/Conditioned/Deterministic 的紧凑编码 |
| `p32_subgrid_set_*` | float32 voxel arrays | 未显式生成的可建模尺度 P32 |
| `p32_unresolved_orientation` | voxel/quality data | 方向未知预算；不进入 total |
| M11 sparse triples | ordinal/index/area | 有正面积的裂隙—体素求交 |
| `p32_explicit_intersection` | M11 voxel array | 真实相交面积/体素体积 |
| `p32_total` | M11 voxel array | explicit intersection + subgrid |

## 9. 体素状态

| 状态 | 内部 code | 含义 | 显示/建模规则 |
|---|---:|---|---|
| OUTSIDE_MODEL | 0 | 模型外 | 不生成、不显示 |
| NO_DATA | 1 | 信息不足 | 不生成；透明靠无几何，不填 0 |
| TRUE_ZERO | 2 | 有支持的真实 0 | 不生成随机裂隙，但作为有效 0 显示 |
| MODELED_VALUE | 3 | 有可靠模型值 | 可用于 M10/M11 |
| EXCAVATION | 4 | 开挖/Mask 排除 | 不生成、不显示 |

## 10. 密度、方向和尺寸术语

| 术语 | 定义 | 本版本中的作用/限制 |
|---|---|---|
| Joint spacing | 相邻裂隙沿某采样线的间距 | 不等同 P10 的倒数于所有情形 |
| P10 | 单位测线长度的裂隙交点数，m⁻¹ | 钻孔区间观测 |
| P32 | 单位体积裂隙面积，m²/m³=m⁻¹ | M9 输入估计；M11 几何回算 |
| RQD | 长岩芯段占比指标，% | 辅助变量，不直接换算 P32 |
| RMR | 岩体质量分级 | 当前未实现计算 |
| Fisher distribution | 球面方向分布 | mean direction + Kappa |
| Kappa | Fisher 集中参数 | 越大表示方向越集中；不是置信度本身 |
| Axial orientation | `n` 与 `−n` 表示同一裂隙面 | Joint Set 距离使用 `abs(dot)` |
| E[R²] | 半径平方期望 | `E[area]=πE[R²]`，不是 `π(E[R])²` |
| Target P32 | 生成配置的期望面积密度 | 与单次 realized P32 区分 |
| Realized P32 | Poisson 实现的生成面积密度 | 有随机波动 |
| Intersection-derived P32 | 真实圆盘—体素面积回算 | M11.1 权威局部几何结果 |
| P32_subgrid | 未显式生成但保留的可建模面积密度 | 加入 M11 P32_total |
| unresolved orientation | 缺可靠方向而无法生成的预算 | 单独报告，不混入 total/subgrid |

## 11. 显示术语

| English control | 中文含义 | 科学语义 |
|---|---|---|
| Voxel Cells | 体素单元 | Exact cell scalar 审计显示 |
| Outer Surface Cloud | 外围云图 | 有效体素体积的真实外表面 |
| Orthogonal Sections | 正交剖面 | X/Y/Z 固定法向平移切片 |
| Arbitrary Section | 任意剖面 | 用户原点和法向平面 |
| Cutaway / Clip Plane | 平面剖切 | 保留一侧并显示新暴露内部面 |
| Box Cutaway | 盒裁切 | 保留 Box 内有效体积 |
| Snap Box to Voxel Faces | 吸附体素面 | 显示边界对齐权威 voxel edges |
| Exact Cell Colours | 精确单元颜色 | 直接 cell scalar，不插值 |
| Smooth Display | 平滑显示 | 仅视觉 point interpolation，不是 Kriging |
| Scalar Bar | 色标 | session-only；内部 ID 不随语言改变 |
| LOD | 层次细节显示 | 显示全部显式裂隙的 GPU 表示，不是科学抽样 |

## 12. 易混淆概念

- **Structural Domain** 是地质分区；**Voxel Analysis Domain** 是体素计算范围；**DFN Generation Domain** 是更大的显式裂隙生成/裁剪范围。
- **Survey azimuth/dip** 描述钻孔轴；**fracture dip direction/dip** 描述裂隙面。
- **M9 preliminary/center-assigned P32** 是生成输入场；**M11 intersection-derived P32** 是圆盘跨体素真实面积回算。
- **NO_DATA** 不是 0；**TRUE_ZERO** 不是透明区。
- **DIP_ONLY** 是受限证据，不是随机补全过的 FULL_ORIENTATION。
- **IDW** 是当前空间插值；Smooth Display 是视觉插值；Kriging 尚未实现。
- **邻接体素**不等于裂隙网络连通；**高 P32** 不等于保证可崩。
