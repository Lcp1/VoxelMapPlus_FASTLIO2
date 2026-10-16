#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "map_builder/voxel_map.h"
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/MarkerArray.h>

#include "map_builder/voxel_map.h"

/**
 * msg: input
 * out: output lidar
 * filter_num: sample rate
 * blind: filter out near range
 */
void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double range_min, double range_max);

struct CloudConvertConfig
{
    double range_min = 0.5;
    double range_max = 80.0;
    int filter_num = 1;            // 抽稀: 每 filter_num 个点取一个
    double snapshot_offset = 0.05; // 无逐点时间时, 参考时刻 = header.stamp + 该值(秒)
    std::string time_field = "";   // 逐点时间字段名, 空则自动探测
    // 旋转式雷达(如Velodyne)没有逐点时间字段时, 可用点自身方位角推算扫描内时刻:
    // t = 方位角(相对首点, 折叠到一帧内) / 360 * scan_period, 用于去畸变
    bool time_from_azimuth = false;
    double scan_period = 0.1;   // 一帧扫描时长(秒)
    double azimuth_dir = -1.0;  // 方位角随时间的增减: 顺时针(-1, 多数机械雷达)或逆时针(+1)
};

/**
 * PointCloud2 -> PointXYZINormal
 * 逐点时间(若存在, 支持 float/double 及绝对/相对时间)换算成 ms 写入 curvature
 * 无逐点时间时 curvature 全为 0, 整帧按 snapshot_offset 处的刚性快照处理
 * 返回 {帧起始时间, 帧结束时间}, 绝对时间(秒)
 */
std::pair<double, double> msg2pcl(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, const CloudConvertConfig &cfg);

/**
 *
 */
sensor_msgs::PointCloud2 pcl2msg(pcl::PointCloud<pcl::PointXYZINormal>::Ptr inp, const std::string &frame_id, const double &timestamp);

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);

void calcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q);

void calcVectQuation(const Eigen::Vector3d &norm, geometry_msgs::Quaternion &q);

visualization_msgs::MarkerArray voxel2MarkerArray(std::shared_ptr<lio::VoxelMap> map, const std::string &frame_id, const double &timestamp, int max_capacity = 1000000, double voxel_size = 0.2);