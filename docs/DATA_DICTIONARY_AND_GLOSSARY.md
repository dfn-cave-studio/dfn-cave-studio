# 数据字典与术语（多类型观测第一阶段）

内部坐标固定为X=Easting、Y=Northing、Z=Elevation，长度使用m；输入/界面角度使用°。

| 数据表/模式 | 必填字段 | 可选字段 | 科学含义 |
|---|---|---|---|
| collars | hole_id, easting, northing, elevation, total_depth | azimuth, dip, domain_id | 孔口与终孔深度；domain_id空值保持未指定 |
| surveys | hole_id, measured_depth, azimuth, dip | — | 沿孔测斜，决定真实三维轨迹 |
| fractures/full_orientation | hole_id, measured_depth, dip_direction, dip | set_id | 一条全局完整产状裂隙观测 |
| fractures/interval_spacing | hole_id, from_depth, to_depth, fracture_spacing | set_id | 区间沿孔平均间距；派生P10但不是P32 |
| fractures/axis_plane_angle | hole_id, measured_depth, axis_plane_angle | set_id | 裂隙面与局部孔轴锐夹角，参考系为BOREHOLE_RELATIVE |
| rqd | hole_id, from_depth, to_depth, rqd | core_recovery | 连续指标0–100，不直接换算P32 |
| rmr | hole_id, from_depth, to_depth, rmr | quality_flag | 连续分值0–100；等级类别不可直接按连续值插值 |
| domain_intervals | hole_id, from_depth, to_depth, domain_id | domain_name | 钻孔结构域区间，采用明确区间关联 |
| orientation_points | point_id, x, y, z, dip, dip_direction | observation_id, domain_id, set_id, site_id, source, quality, observation_kind | 有位置和产状的点观测，不代表有限面积结构面 |

`Raw`保留源值；`Formal`是可供对应下游使用的已接受记录；`Excluded`保留原因；`Pending`表示尚无collar关联。`derived_p10`为间距倒数（m⁻¹），不得解释为实际裂隙整数计数或P32。`observation_kind=MEASURED`表示单条实测，`DOMINANT_SUMMARY`表示人工汇总优势方向，两者不会混作同一种原始计数。

第一阶段未实现：从孔轴夹角补全全局方向、观测影响权重、无钻孔点的Calibration/Validation分组、将点产状自动转成条件裂隙或确定性圆盘。
