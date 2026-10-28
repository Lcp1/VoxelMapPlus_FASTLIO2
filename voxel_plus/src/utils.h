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

#include "map_builder/voxel_map.h" // NOTE: 与第 4 行重复 include(#pragma once 已保证只展开一次, 无实际影响)

/**
 * msg: input
 * out: output lidar
 * filter_num: sample rate
 * blind: filter out near range
 *
 * --- 以下为补充说明(原注释是英文简写, 其中 blind 参数在当前函数签名里已不存在) ---
 * Livox 自定义点云消息(CustomMsg)转 PCL 点云
 *
 * msg        : 输入, livox_ros_driver2 发布的一帧雷达数据
 * out        : 输出, pcl::PointXYZINormal 点云; 函数内部先 clear 再按需填充
 * filter_num : 抽帧间隔, 每 filter_num 个通过线号/回波筛选的点保留 1 个
 * range_min  : 有效点距离下限(米), 距离不大于它的点被丢弃
 * range_max  : 有效点距离上限(米), 距离不小于它的点被丢弃
 *
 * 输出点复用了两个字段: intensity = 反射率, curvature = 点相对本帧起始的偏移时间(毫秒);
 * curvature 会被 syncPackage() 用来推算点云结束时刻, 也被 LIOBuilder::undistortCloud() 用来做运动补偿
 * 调用关系: lio_node.cpp 的 lidarCB() 调用本函数, 是整条前端链路的入口预处理
 */
void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double range_min, double range_max);

/**
 * PCL 点云封装成 ROS 的 sensor_msgs::PointCloud2
 *
 * inp       : 输入点云(pcl::PointXYZINormal)
 * frame_id  : 写进消息 header 的坐标系名
 * timestamp : 绝对时间(秒); 若小于 0 则改用当前 ROS 时间(ros::Time::now())
 * 调用关系: lio_node.cpp 的 publishCloud() 用它发布 body_cloud / world_cloud 话题
 */
sensor_msgs::PointCloud2 pcl2msg(pcl::PointCloud<pcl::PointXYZINormal>::Ptr inp, const std::string &frame_id, const double &timestamp);

/**
 * Eigen 形式的刚体变换转 tf 的 TransformStamped
 *
 * rot            : 父坐标系到子坐标系的旋转矩阵(3x3)
 * pos            : 父坐标系到子坐标系的平移向量
 * frame_id       : 父坐标系名
 * child_frame_id : 子坐标系名
 * timestamp      : 绝对时间(秒)
 * 四元数由旋转矩阵直接构造(Eigen::Quaterniond(rot))
 * 调用关系: lio_node.cpp 的 mainCB() 用它构造 map -> body 的 TF 交给 TransformBroadcaster
 */
geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

/**
 * Eigen 形式的位姿转 nav_msgs::Odometry
 *
 * rot            : 位姿的旋转矩阵(3x3)
 * pos            : 位姿的平移向量
 * frame_id       : 位姿所在坐标系(父)
 * child_frame_id : 载体坐标系(子)
 * timestamp      : 绝对时间(秒)
 * 只填了 pose, 没有填 twist 和协方差
 * NOTE: 本包内没有调用点(名字相近的 eigen2Transform 才被 lio_node.cpp 使用), 属于备用接口
 */
nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp);

/**
 * jet 伪彩色映射: 把标量映射成一条"蓝 -> 青 -> 绿 -> 黄 -> 红"的连续色带
 *
 * v         : 输入标量(此处是体素平面法向量协方差的迹, 调用前已归一化到 [0, 1])
 * vmin/vmax : 截断范围, v 先被截断到该区间
 * r/g/b     : 输出的颜色分量(0~255)
 * 调用关系: utils.cpp 的 voxel2MarkerArray() 用它给平面 marker 上色
 * NOTE: 分段边界 0.1242 / 0.3747 / 0.6253 / 0.8758 是写死的常数, 只有以 vmin=0, vmax=1
 *       调用时截断结果才与这些边界自洽; 换成别的区间调用需要同步修改常数
 */
void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);

/**
 * 由三个坐标轴向量构造四元数
 *
 * x_vec/y_vec/z_vec : 三个轴向量, 依次作为旋转矩阵的三列
 * q                 : 输出四元数
 * 内部把三个向量按列拼成 3x3 矩阵, 再交给 Eigen::Quaterniond 并归一化
 * NOTE: 本包内没有调用点; 若三个输入向量不两两正交, 拼出的矩阵不是旋转矩阵, 结果会有偏差
 */
void calcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q);

/**
 * 由平面法向量构造四元数: 求一个把 +Z 轴转过去与 norm 对齐的最小旋转
 *
 * norm : 平面法向量(内部按 Eigen::Quaterniond::FromTwoVectors 处理, 不必预先归一化)
 * q    : 输出四元数
 * 调用关系: utils.cpp 的 voxel2MarkerArray() 用它给平面 marker 定向, 使 marker 的局部 +Z 轴
 *           与平面法向对齐(配合 scale.z 很小, 看上去就是一张贴在平面上的薄片)
 */
void calcVectQuation(const Eigen::Vector3d &norm, geometry_msgs::Quaternion &q);

/**
 * 体素平面地图转成 rviz 可视化用的 MarkerArray
 *
 * map          : 体素地图, 只读取其 cache_(LRU 链表)与 featmap_(哈希表)
 * frame_id     : marker 所在坐标系名
 * timestamp    : 绝对时间(秒), 写进每个 marker 的 header
 * max_capacity : 本次最多检查并发布的体素平面数量(默认 1000000)
 * voxel_size   : marker 的边长(默认 0.2), lio_node.cpp 调用时传的是 voxel_size * 0.8
 * 只发布"已冻结"的平面体素(is_plane_ 为真且 update_enable_ 为假), 具体实现见 utils.cpp
 * 调用关系: lio_node.cpp 的 voxelTimerCB() 按 5s 周期调用, 结果发到 voxel_map 话题
 */
visualization_msgs::MarkerArray voxel2MarkerArray(std::shared_ptr<lio::VoxelMap> map, const std::string &frame_id, const double &timestamp, int max_capacity = 1000000, double voxel_size = 0.2);