# FastLIO2 With VoxelMapPlus And STD
![image](https://github.com/liangheming/VoxelMapPlus_FASTLIO2/blob/main/resources/demo.png)

## 主要工作
1. 基于[FASTLIO2](https://github.com/hku-mars/FAST_LIO)和[VoxelMap](https://github.com/hku-mars/VoxelMap)
2. 参考[VoxelMapPlus](https://github.com/uestc-icsp/VoxelMapPlus_Public)
3. 目前暂时支持MID_360的传感器
4. 平面主要还是用均值和法向量进行参数化，而不是使用(ax+by+z+d=0,x+by+cz+d=0,ax+y+cz+d=0)建模
5. voxel与voxel之间主要基于法向量夹角(余弦相似度)，以及原点到平面距离(欧式距离)这两个参数进行融合
6. 地图的内存管理采用LRU(last recently used)的规则，优先释放最长时间没有使用的voxel
7. 重写了[STD](https://github.com/hku-mars/STD)的代码，去除最大特征个数的限制，简化了部分代码，去除对ceres的依赖。
8. 基于[STD](https://github.com/hku-mars/STD) 完成回环检测和相应的位姿图优化功能；
## 环境说明
```text
系统版本: ubuntu20.04
机器人操作系统: ros1-noetic
```

## 编译依赖
1. livox_ros_driver2
2. pcl
3. sophus
4. eigen

### 1.安装 LIVOX-SDK2
```shell
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd ./Livox-SDK2/
mkdir build
cd build
cmake .. && make -j
sudo make install
```

### 2.安装 livox_ros_driver2
```shell
mkdir -r ws_livox/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git ws_livox/src/livox_ros_driver2
cd ws_livox/src/livox_ros_driver2
./build.sh ROS1
```

### 3. 安装Sophus
```
git clone https://github.com/strasdat/Sophus.git
cd Sophus
mkdir build && cd build
cmake .. -DSOPHUS_USE_BASIC_LOGGING=ON
make
sudo make install
```
**新的Sophus依赖fmt，可以在CMakeLists.txt中添加add_compile_definitions(SOPHUS_USE_BASIC_LOGGING)去除，否则会报错**

## DEMO 数据
```text
链接: https://pan.baidu.com/s/1eEnc5fWYCQJxVCaEw7buiA?pwd=cwd4 提取码: cwd4 
--来自百度网盘超级会员v7的分享
```

## 启动脚本
建图线程
```shell
roslaunch voxel_plus mapping.launch
rosbag play your_bag.bag
```
建图加回环
```
roslaunch std_matcher pgo.launch
rosbag play your_bag.bag
```

## 支持 sensor_msgs/PointCloud2 的雷达
`mapping.yaml` 里的 `lidar_type` 决定订阅的消息类型:
- `livox`(默认): `livox_ros_driver2::CustomMsg`
- `pointcloud2`: `sensor_msgs/PointCloud2`, 按字段名解析 x/y/z/intensity, 并且**自动探测逐点时间字段**
  (`timestamp`/`time`/`t`/`offset_time`/`time_offset`/`offset`, 支持绝对时间与秒/毫秒/微秒/纳秒相对时间),
  换算成毫秒写入 `curvature`, 走标准去畸变流程; 没有逐点时间字段时, 整帧按
  `header.stamp + cloud_time_offset` 时刻的刚性快照处理(建议取半帧周期的 0.05s)。

### 示例: 港口双雷达数据 `2026-10-10-10-56-17_0.bag`
雷达 `/lidar_preprocessor/meta_cloud`(10Hz, base_link, 无逐点时间), IMU `/ins_driver/imu`(100Hz, 加速度单位是 g),
配置见 `voxel_plus/config/mapping.yaml`。bag 自带 `/tf`(map->base_link) 与车端定位, 播放时建议只播需要的两个话题:
```shell
roslaunch voxel_plus mapping.launch
rosbag play /home/lcp/data/bag/221/10-10-1/2026-10-10-10-56-17_0.bag \
  --topics /lidar_preprocessor/meta_cloud /ins_driver/imu
```
该数据下节点 CPU 占用约 0.2 核(462 帧/46.5s), 轨迹与 bag 自带的 `/localization/ins` 相比,
44m 行程内的位移误差约 0.1~0.3m。

### 坐标系约定 (建图 / 建图+回环 都要注意)
LIO 输出 `lio_map -> body`。**LIO 的 `map_frame` 不能取 `map`**: `pgo_node` 会把
`/lio_node/cloud_with_odom` 的 `header.frame_id`(即 LIO 的 map_frame)当作局部帧, 再广播
"自己的 map_frame(std_matcher/config/config.yaml 里配置的全局帧, 默认 map) -> 该局部帧"; 两者同名就会发出
`map -> map` 的非法变换, tf 一直报
`Ignoring transform from authority ... with frame_id and child_frame_id "map" because they are the same`。
所以 LIO 用 `lio_map`, pgo 用 `map`, rviz 的 Fixed Frame 设成 `lio_map`(纯建图模式下它就存在,
回环模式下它是 `map` 的子节点, 两种模式都能显示)。
另外 bag 自带的 `/tf` 里有 `map -> base_link`(车端定位), 播放时建议只播需要的两个话题以免混淆。

## 适配 KITTI 数据 (2011_09_30_0018)

这个 bag 与常规 LIO 输入差别较大，做了三件事：

1. **合成 IMU**：bag 里没有任何 IMU 测量，只有 `/pose_imu`(10Hz 位姿)。新增节点 `pose_to_imu`
   把位姿合成成 100Hz 的 `sensor_msgs/Imu`：位姿序列做局部二次拟合得到速度/加速度（比直接差分抗噪），
   姿态 slerp 插值，角速度取段内常值，再按 `f = R^T(a - g)` 得到比力。
2. **用方位角做去畸变**：点云没有逐点时间字段（只有 x/y/z/intensity），但它是旋转式雷达、每帧正好一圈
   （126957/64≈1983 个方位步），所以按**每个点自身的方位角**推算扫描内时刻：
   `t = 方位角(相对首点, 折叠到一帧内)/360 × 扫描周期`，配置项 `point_time_from_azimuth: true`。
   实测（119m 行程、7~8m/s）误差 mean 0.21m → 0.14m，max 0.78m → 0.43m。
3. **初速度提示**：常速度对加速度计不可观测，起步就在动的数据靠点云学速度会滞后几十米。
   `pose_to_imu` 额外把位姿导数得到的速度发到 `<imu_topic>/velocity`，LIO 用 `velocity_topic` 接收
   并作为初速度。加上它之后，起步即与真值一致（不加则前 5 秒速度从 0 爬到 7m/s，累积 ~22m 偏差）。

启动：
```shell
roslaunch voxel_plus mapping_kitti.launch
rosbag play /home/lcp/data/bag/2011_09_30_0018/2011_09_30_0018.bag
```
> 注意该 bag 里话题名没有前导斜杠（`velodyne_points` 等），播放时 ROS 会解析成 `/velodyne_points`，正常可用。
> `pose_to_imu` 的输出比位姿晚约 0.5s（对称窗口拟合不引入滞后，代价是延迟），LIO 会跟着延迟处理，不影响离线建图。

实测（`-s 20 -u 45` 一段，119m 行程，车速 6~8m/s）：与 `/pose_imu` 真值相比
**位移误差 mean 0.14m / max 0.43m**，CPU 占用约 **0.3 核**。

配置见 `voxel_plus/config/mapping.yaml`（原 221 港口数据的配置另存为 `mapping_port.yaml`）。
关键项：`p_il = [0.811, -0.307, 0.803]`（tf 里 `imu->velodyne` 的平移，1.2m 力臂不能忽略）、
`gravity_align: false`（车体基本水平，避免被行驶加速度带偏）、`na/ng` 按合成 IMU 实测噪声给。

## 已知问题修复记录
1. `voxel_map.cpp` 平面合并公式缺少括号, 实际算成了 `tm0*mean + tm1*mean/(tm0+tm1)`, 平面均值/法向量会
   不断向 0 塌缩(静止场景几十秒后残差可达米级), 已修正为加权平均。
2. `buildResidual` 的 3sigma 残差门限过于乐观(约0.2m), 位姿滞后时会把关键点全部剔除导致状态卡死,
   现在门限取 `max(3sigma, max_residual)`(默认0.5m), 仍能剔除动态物体等大残差。
3. `residual_info` 容量固定 10000, 点云点数超过时会越界写内存(直接段错误), 已按需扩容。
4. `undistortCloud` 里 `data_group.last_imu = package.imus.back()` 在 IMU 数据没覆盖到该帧点云时
   (bag 开头常见)会越界, 已加保护, 并在节点里直接丢弃没有 IMU 覆盖的点云帧。
5. MAP_INIT 只建一帧地图就开始解算: 此时地图平面很少(bag 开头单帧只有几十个有效点), 稀疏地图上的
   虚假残差会在单帧内注入假速度(pos-vel 耦合放大 v=dp/dt), 位置直接跳 1m 并锁死。现在先用
   `map_init_frames`(默认5)帧累积地图再开始解算。
6. `update_size_threshes` 与代码里的 `update_size_thresh` 拼写不一致导致该参数被忽略, 已修正(旧写法仍兼容)。
7. `opti_max_iter` 之前从未传给 IESKF, 现已生效。
## 特别感谢
1. [FASTLIO2](https://github.com/hku-mars/FAST_LIO)
2. [VoxelMap](https://github.com/hku-mars/VoxelMap)
3. [VoxelMapPlus](https://github.com/uestc-icsp/VoxelMapPlus_Public)
4. [STD](https://github.com/hku-mars/STD)

## 对于STD检测回环的一些想法
1. 描述子提取不够稳定(网格的划分，网格遍历的顺序，甚至是特征值的调用方式，都会导致类似场景的描述子提取的不一致性);
2. 无论是STD还是SC，大多利于360的平放的lidar，对斜放的，小FOV的lidar都不是很友好;
3. 整体上召回率还是较低(对于回环优化来说是一件好事);
4. 还是存在误检的情况,误检测的问题还是需要多方校验的，谨慎进行回环优化[LTAOM](https://github.com/hku-mars/LTAOM) 有相关优化，但是个人感觉代价较大;