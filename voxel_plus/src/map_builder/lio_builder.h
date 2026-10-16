#pragma once
#include "ieskf.h"
#include "commons.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include "voxel_map.h"

namespace lio
{
    enum LIOStatus
    {
        IMU_INIT,
        MAP_INIT,
        LIO_MAPPING

    };
    struct LIOConfig
    {
        int opti_max_iter = 5;
        // 初始化阶段帧数: 地图刚建立时平面很少(单帧只有几十个有效点), 稀疏地图上的虚假残差
        // 会通过 pos-vel 耦合在单帧内注入假速度(v=dp/dt)并锁死。这几帧里用强先验把速度状态锁住
        // (位置仍由点云更新, 所以运动中起步也不会把地图拖糊), 之后放开速度由点云序列估计
        int init_frames = 10;
        // 初始化阶段的位置/姿态/速度先验方差
        double init_pos_cov = 1e-3;   // (0.03m)^2
        double init_rot_cov = 1e-3;   // (0.03rad)^2 (重力对齐后的姿态精度)
        double init_vel_cov = 1e-4;   // (0.01m/s)^2 (初始化阶段速度锁)
        double na = 0.01;
        double ng = 0.01;
        double nba = 0.0001;
        double nbg = 0.0001;
        // 协方差下限(方差), 防止静止场景下P塌缩导致状态锁死; 设为0则退化为原始行为
        double min_pos_cov = 1e-4;    // (0.01m)^2
        double min_rot_cov = 1e-6;    // (0.001rad)^2
        double min_vel_cov = 2.5e-3;  // (0.05m/s)^2
        double min_bias_cov = 1e-8;   // (1e-4)^2
        int imu_init_num = 20;
        Eigen::Matrix3d r_il = Eigen::Matrix3d::Identity();
        Eigen::Vector3d p_il = Eigen::Vector3d::Zero();
        bool gravity_align = true;
        bool estimate_ext = false;

        double scan_resolution = 0.1;
        double voxel_size = 0.5;
        int update_size_thresh = 10;
        int max_point_thresh = 100;
        double plane_thresh = 0.01;

        double ranging_cov = 0.04;
        double angle_cov = 0.1;
        double max_residual = 0.5; // 残差门限的绝对下限(米)

        double merge_thresh_for_angle = 0.1;
        double merge_thresh_for_distance = 0.04;
        int map_capacity = 100000;
    };
    struct LIODataGroup
    {
        IMUData last_imu;
        std::vector<IMUData> imu_cache;
        std::vector<Pose> imu_poses_cache;
        Eigen::Vector3d last_acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d last_gyro = Eigen::Vector3d::Zero();
        double last_cloud_end_time = 0.0;
        double gravity_norm;
        int effect_num = 0; // 上一帧参与优化的有效点数(用于状态打印)
        int lio_frame_count = 0;
        kf::Matrix12d Q = kf::Matrix12d::Identity();
        std::vector<ResidualData> residual_info;
    };

    class LIOBuilder
    {
    public:
        LIOBuilder() = default;

        void loadConfig(LIOConfig &_config);

        // 初始化速度提示(机体系): 只有位姿的数据集(如KITTI)里, 常速度对加速度计不可观测,
        // 但位姿导数能直接给出速度, 用它作为滤波器的初速度可消除起步几十米的收敛滞后
        void setVelocityHint(const Eigen::Vector3d &v_body) { velocity_hint = v_body; has_velocity_hint = true; }

        // 时间基准跳变(换了数据源/bag时钟不一致/长时间断流)后重新初始化: 清空缓存并重建地图,
        // 否则 IMU 与点云的时间差会变成巨大的 dt, 直接把状态推到发散
        void reset();

        void setupEstimator();

        bool initializeImu(std::vector<IMUData> &imus);

        void undistortCloud(SyncPackage &package);

        void process(SyncPackage &package);

        void sharedUpdateFunc(kf::State &state, kf::SharedState &shared_state);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidarToWorld(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidarToBody(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud);

    public:
        kf::IESKF kf;
        LIOConfig config;
        LIODataGroup data_group;
        LIOStatus status = LIOStatus::IMU_INIT;
        Eigen::Vector3d velocity_hint = Eigen::Vector3d::Zero();
        bool has_velocity_hint = false;
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr lidar_cloud;
        pcl::VoxelGrid<pcl::PointXYZINormal> scan_filter;
        std::shared_ptr<VoxelMap> map;
    };

} // namespace lio
