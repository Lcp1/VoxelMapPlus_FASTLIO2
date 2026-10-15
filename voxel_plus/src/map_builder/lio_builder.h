#pragma once
#include "ieskf.h"
#include "commons.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include "voxel_map.h"

namespace lio
{
    // 前端状态机: IMU 初始化 -> 首帧建图 -> 正常定位建图, 由 LIOBuilder::process() 按帧推进
    enum LIOStatus
    {
        IMU_INIT,   // 攒 IMU 求平均, 估计重力方向/零偏并初始化 IESKF(见 LIOBuilder::initializeImu)
        MAP_INIT,   // 用首帧点云调用 VoxelMap::build() 建初始地图, 本帧不做滤波更新
        LIO_MAPPING // 正常帧: 去畸变 -> 降采样 -> IESKF 迭代更新 -> VoxelMap::update()

    };

    // 前端全部可调参数, 由 lio_node.cpp 从 rosparam 读入后交给 LIOBuilder::loadConfig()
    struct LIOConfig
    {
        int opti_max_iter = 5;          // IESKF::update() 的迭代次数上限
        double na = 0.01;               // 加速度计测量噪声(填进过程噪声 Q)
        double ng = 0.01;               // 陀螺仪测量噪声(填进过程噪声 Q)
        double nba = 0.0001;            // 加速度计零偏随机游走噪声(填进 Q)
        double nbg = 0.0001;            // 陀螺仪零偏随机游走噪声(填进 Q)
        int imu_init_num = 20;          // IMU 初始化至少需要累积的帧数
        Eigen::Matrix3d r_il = Eigen::Matrix3d::Identity(); // LiDAR 到 IMU 的旋转外参 R_IL
        Eigen::Vector3d p_il = Eigen::Vector3d::Zero();     // LiDAR 到 IMU 的平移外参 P_IL
        bool gravity_align = true;      // 是否用静止段平均加速度对齐初始姿态(见 initializeImu)
        bool estimate_ext = false;      // 是否把外参也当作状态量在线估计(见 sharedUpdateFunc)

        double scan_resolution = 0.1;   // 当前帧点云降采样体素边长; <= 0 表示不降采样
        double voxel_size = 0.5;        // 地图体素边长, 传给 VoxelMap
        int update_size_thresh = 10;    // 传给 VoxelGrid 的周期重算点数阈值(update_point_thresh)
        int max_point_thresh = 100;     // 传给 VoxelGrid 的单体素点数上限
        double plane_thresh = 0.01;     // 平面判定阈值, 传给 VoxelMap/VoxelGrid

        double ranging_cov = 0.04;      // 测距噪声, 作为 calcBodyCov() 的 range_inc
        double angle_cov = 0.1;         // 测角噪声(单位: 度), 作为 calcBodyCov() 的 degree_inc

        double merge_thresh_for_angle = 0.1;     // 平面合并夹角阈值(1 - cos), 覆盖 VoxelGrid 的同名静态变量
        double merge_thresh_for_distance = 0.04; // 平面合并间距阈值, 覆盖 VoxelGrid 的同名静态变量
        int map_capacity = 100000;      // 地图体素数量上限, 超出后按 LRU 淘汰
    };

    /**
     * @brief 跨帧解算上下文: IMU 与预积分位姿缓存、过程噪声、逐点残差缓存
     */
    struct LIODataGroup
    {
        IMUData last_imu;   // 上一帧最后一个 IMU 采样, 作为本帧预积分的起点
        std::vector<IMUData> imu_cache; // 本帧参与预积分的 IMU 序列(IMU 初始化阶段先用作初始化数据)
        std::vector<Pose> imu_poses_cache; // 预积分过程中逐 IMU 缓存的位姿, 去畸变时按时间分段插值
        Eigen::Vector3d last_acc = Eigen::Vector3d::Zero();  // 上次预积分末尾的世界系加速度估计(去畸变短时外推用)
        Eigen::Vector3d last_gyro = Eigen::Vector3d::Zero(); // 上次预积分末尾的机体系角速度估计(去畸变短时外推用)
        double last_cloud_end_time = 0.0; // 上一帧点云结束时间, 决定本帧 IMU 积分从哪一时刻开始
        double gravity_norm;    // IMU 初始化中赋值的平均加速度模长(initializeImu 之前未初始化), 用于把加速度归一到 9.81
        kf::Matrix12d Q = kf::Matrix12d::Identity(); // 过程噪声协方差, 四个对角块在 loadConfig() 中填写
        std::vector<ResidualData> residual_info; // 逐点残差缓存(ResidualData 定义见 voxel_map.h), 容量在 loadConfig() 中固定
    };

    /**
     * @brief 前端核心调度类: 点云去畸变、点云坐标系变换, 以及"体素平面地图 <-> IESKF"的桥接
     *
     * 每帧数据流(由 lio_node.cpp 的定时器回调 mainCB 调用 process() 驱动):
     *   process()
     *     -> undistortCloud()  用 IMU 预积分把点云去畸变(同时完成 IESKF 的预测)
     *     -> calcBodyCov() 等  构造 PointWithCov 列表(世界系点坐标 + 点协方差)
     *     -> 首帧: map->build()              批量建立体素平面地图
     *        后续: kf.update() + map->update()  IESKF 迭代更新 + 地图增量插入
     *   kf.update() 每次迭代都会回调 sharedUpdateFunc(), 由它调用 VoxelMap::buildResidual()
     *   组装点-面残差的 H/b 交给 IESKF 求解
     */
    class LIOBuilder
    {
    public:
        /** 默认构造: 各成员取默认值, 真正的配置与建图在 loadConfig() 中完成 */
        LIOBuilder() = default;

        /**
         * 加载配置(由 lio_node.cpp 在构造后调用一次)
         * 内部: 填过程噪声 Q -> 设置降采样器叶节点尺寸 -> 新建 VoxelMap
         *       -> 覆盖 VoxelGrid 的静态合并阈值 -> 注册 IESKF 观测回调 -> 预分配残差缓存
         */
        void loadConfig(LIOConfig &_config);

        /**
         * IMU 初始化(由 process() 在 IMU_INIT 状态下反复调用)
         * 攒够 config.imu_init_num 帧后: 求平均加速度/角速度, 设置外参与零偏初值,
         * 对齐初始姿态与重力方向, 初始化协方差 P; 返回 true 表示初始化完成
         */
        bool initializeImu(std::vector<IMUData> &imus);

        /**
         * 点云去畸变(由 process() 每帧调用)
         * 用 IMU 预积分位姿把每个点补偿到扫描结束时刻的雷达系, 就地覆盖 package.cloud 的坐标
         */
        void undistortCloud(SyncPackage &package);

        /**
         * 每帧主入口(由 lio_node.cpp 的定时器回调 mainCB 调用)
         * 按 status 走状态机: IMU_INIT / MAP_INIT / LIO_MAPPING 三个分支, 见 .cpp 中的实现
         */
        void process(SyncPackage &package);

        /**
         * 点-面残差的观测方程回调
         * 由 IESKF::update() 在每次迭代时调用(经 loadConfig() 中注册的 lambda 转发),
         * 内部调用 VoxelMap::index()/buildResidual() 生成 H/b 输出到 shared_state
         */
        void sharedUpdateFunc(kf::State &state, kf::SharedState &shared_state);

        /** 雷达系点云 -> 世界系: p_w = R_wi * (R_IL * p_l + P_IL) + p_wi, 用当前状态估计 */
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidarToWorld(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

        /** 雷达系点云 -> 机体系(IMU 系): 只施加外参 R_IL/P_IL, 供 lio_node 发布 body_cloud */
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidarToBody(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

    public:
        kf::IESKF kf;                     // 迭代误差状态卡尔曼滤波器(状态 23 维, 定义见 ieskf.h)
        LIOConfig config;                 // 参数副本, 由 loadConfig() 整体赋值
        LIODataGroup data_group;          // 跨帧解算上下文
        LIOStatus status = LIOStatus::IMU_INIT; // 状态机当前状态, lio_node 也据此判断本帧是否发布结果
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidar_cloud; // 当前帧降采样后的点云, sharedUpdateFunc() 使用
        pcl::VoxelGrid<pcl::PointXYZINormal> scan_filter; // 当前帧体素降采样器(scan_resolution > 0 时生效)
        std::shared_ptr<VoxelMap> map;    // 体素平面地图(lio_node 的体素可视化也直接读它)
    };

} // namespace lio
