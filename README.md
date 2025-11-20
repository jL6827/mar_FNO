```markdown
# FNO2d for ocean point data → field reconstruction

说明
- 该项目使用原始 CSV（processed_data_mean.csv）作为唯一数据源，不修改原始文件。
- 为满足严格的 2D FNO（需要规则 2D 网格），对原始点集合按 (date, depth_level) 做三维插值 (lon,lat,depth) -> (uo,vo,so,thetao)，生成规则经纬度切片（每个切片就是一个训练样本）。
- 模型输入通道：lon_norm, lat_norm, t_norm, depth_norm（4 通道）。输出：uo, vo, so, thetao（4 通道）。
- 数据拆分：对生成的样本（date × depth_level）随机划分，85% 训练 / 15% 测试。
- 支持 GPU（CUDA），可以通过 --device 参数指定。

文件列表
- train_fno2d.py：主脚本，负责数据处理、训练、评估、示例重建与输出。
- requirements.txt：依赖清单。
- README.md：本文件。

快速开始
1. 创建并激活虚拟环境（你已说明已创建好虚拟环境）。
2. 安装依赖：
   pip install -r requirements.txt

3. 把你的 processed_data_mean.csv 放在当前目录（或给 --csv 指定路径）。

4. 运行训练（示例，使用 GPU）：
   python train_fno2d.py --csv processed_data_mean.csv --nx 64 --ny 64 --nz 8 --epochs 120 --batch 8 --device cuda

   常用参数：
   - --nx / --ny: 网格分辨率（lat 行数，lon 列数）
   - --nz: 每个日期上要插值的深度层数（模型将以每层为单独样本训练）
   - --min_points: 若某日观测点少于该值则跳过该日
   - --epochs / --batch / --lr
   - --modes1/--modes2/--width: FNO 超参数
   - --test_frac: 测试集占比（默认 0.15）
   - --device: cuda 或 cpu

输出
- 训练过程会保存最佳模型到 --save_model（默认 fno2d_ocean.pth）。
- 训练结束会保存示例文件：
  - sample_input.npy, sample_target.npy, sample_pred.npy（用于快速检查）
  - example_uo_pred.png（示例切片的 uo 目标 vs 预测 对比）
- 若需完整整场重建（所有深度层），可以在脚本基础上用保存的模型循环构造所有 depth_level 的输入并推断。

改进建议（可选）
- 如果想提高训练质量，可在 preprocess 中加标准化/均值方差归一化，或对输出做标准化并在损失里加回归到物理量的项。
- 若你有密集的深度层观测，也可以将 FNO 扩展为 3D FNO（更复杂但能直接处理 x,y,z）以避免分层插值。
- 可以加入物理约束损失（涡度、质量守恒等）以引入物理先验。

```