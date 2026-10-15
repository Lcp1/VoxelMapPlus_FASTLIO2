#pragma once
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace lio
{
    /**
     * 旋转矩阵 -> 欧拉角(roll, pitch, yaw), 实现见 commons.cpp
     * 约定为 ZYX 顺序(R = Rz(yaw) * Ry(pitch) * Rx(roll)), 即常用的 RPY 定义; 未处理万向锁
     * NOTE: 形参是非 const 引用但函数没有修改它, 且本工程内暂无调用者
     */
    Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot);

    /**
     * 两个点之间的欧氏距离平方, 实现见 commons.cpp
     * 只用 x/y/z 三个字段, 返回平方值(没有开方)
     * NOTE: 本工程内暂无调用者
     */
    float sq_dist(const pcl::PointXYZINormal &p1, const pcl::PointXYZINormal &p2);

    /**
     * @brief 一个 IMU 采样(加速度 + 角速度 + 时间戳)
     */
    struct IMUData
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d acc;  // 加速度计读数
        Eigen::Vector3d gyro; // 陀螺仪读数
        double timestamp;     // 采样时刻, 单位: 秒
        IMUData() = default;
        // NOTE: 第三个参数是非 const 引用(疑似笔误, 值传递 double d 更自然), 传临时量或 const 变量会编译失败
        IMUData(const Eigen::Vector3d &a, const Eigen::Vector3d &g, double &d) : acc(a), gyro(g), timestamp(d) {}
    };

    /**
     * @brief 一帧同步好的数据: 一帧点云 + 覆盖其时间范围的 IMU
     * 由 lio_node.cpp 的 syncPackage() 打包, 再交给 LIOBuilder::process()
     */
    struct SyncPackage
    {
        std::vector<lio::IMUData> imus;  // 落在本帧点云时间范围内的 IMU 采样(按时间升序)
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud; // 一帧点云, curvature 字段存每点相对扫描起点的毫秒偏移
        double cloud_start_time = 0.0;   // 扫描起始时刻(秒)
        double cloud_end_time = 0.0;     // 扫描结束时刻(秒), 由起始时刻加最后一个点的偏移得到
    };

    /**
     * @brief 预积分过程中逐 IMU 缓存的位姿节点, 供去畸变按时间外推使用(见 LIOBuilder::undistortCloud)
     */
    struct Pose
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d acc;   // 该时刻世界系下的加速度估计(已扣加计零偏并计入重力)
        Eigen::Vector3d gyro;  // 该时刻机体系下的角速度估计(已扣陀螺零偏)
        Eigen::Matrix3d rot;   // 姿态 R_wi
        Eigen::Vector3d pos;   // 位置 p_wi
        Eigen::Vector3d vel;   // 速度 v_w
        Pose();
        // NOTE: 初始化列表顺序与成员声明顺序不一致(offset 声明在最后, 会触发 -Wreorder 警告), 无实际影响
        // NOTE: 默认构造函数 Pose() 在本工程中只有声明没有定义(commons.cpp 里没有实现), 一旦被调用会链接报错
        Pose(double t, Eigen::Vector3d a, Eigen::Vector3d g, Eigen::Vector3d v, Eigen::Vector3d p, Eigen::Matrix3d r)
            : offset(t), acc(a), gyro(g), vel(v), pos(p), rot(r) {}
        double offset; // 相对本帧点云扫描起始时刻的秒数, 用作去畸变的时间索引
    };

    /**
     * 计算雷达系下一点的位置协方差(量测噪声模型), 实现见 commons.cpp
     * 由 LIOBuilder::process() 对每个点调用, 求出的协方差随后随状态一起传播到世界系
     */
    void calcBodyCov(Eigen::Vector3d &pb, const double &range_inc, const double &degree_inc, Eigen::Matrix3d &cov);
}
