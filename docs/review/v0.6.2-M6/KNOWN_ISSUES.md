# Known Issues

- M6导入UI需后续里程碑补全字段映射与LAS预览
- 确定性裂隙STL/OBJ导入未实现
- 体素3D渲染需pyvista.UniformGrid集成
- 大DFN(>5万裂隙)连通性O(N²)需空间索引
- 几何连通性与力学连通性需M11区分
- 项目仍使用.dfncs JSON格式，.dfnproj ZIP格式待实现
- 体素裁剪面积使用64边形近似(误差<0.1%)
- GUI测试需显示服务器(CI中设置QT_QPA_PLATFORM=offscreen)