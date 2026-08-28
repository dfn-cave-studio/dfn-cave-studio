# M10人工验收资料

1. 打开一个已经完成M9 Density、Size、First Voxel Parameter Field和Voxel Grid的`.dfnproj`。
2. 点击`12. Explicit DFN Generation`。
3. 导入`deterministic_structures.csv`；该文件是参数化圆盘，不是STL/OBJ曲面。
4. 查看预计裂隙数和内存，设置`base seed=42`、`realizations=1`，点击`Generate Batch`。
5. 分别渲染全部、按节理组、条件裂隙和确定性结构面；重复渲染同一分组不会增加同名图层。
6. 测试隐藏/显示、透明度、颜色、Remove、Clear Current和Clear All。M9切片、钻孔、边界和坐标轴不得被清除。
7. 点击OK后保存`.dfnproj`，关闭并重开，核对实现ID、seed、数量和质量摘要。
8. 在M10窗口选择实现并导出，重新读取CSV、JSON、NPZ和VTP。
9. 将`base seed`改为43生成第二次结果，几何应改变；相同输入和seed重新生成应逐项一致。

质量报告中的局部P32状态为`CENTER_ASSIGNED_PRELIMINARY`。M11才会通过真实裂隙—体素求交回算局部P32。
