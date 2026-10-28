#include "utils.h"

/**
 * Livox 自定义点云消息(CustomMsg)转 PCL 点云
 *
 * 步骤:
 *   1. 清空输出点云, 按 point_num / filter_num 粗略预留容量
 *   2. 逐点筛选线号与回波标识: 只保留 line < 4(Mid-360 的 4 条扫描线)
 *      且 tag 的 bit4~bit5 为 0x00 或 0x10 的点
 *   3. 抽帧: 对通过第 2 步的点计数, 每 filter_num 个只保留 1 个
 *   4. 距离筛选: 只保留距离平方落在 (range_min^2, range_max^2) 开区间内的点
 *   5. 写入 out, 其中 intensity 存反射率, curvature 存点相对帧起始的偏移时间(纳秒换算成毫秒)
 *
 * 参数含义见 utils.h 的声明注释
 * 调用关系: lio_node.cpp 的 lidarCB() 调用; 输出的 curvature 供 syncPackage() 推算点云结束时刻,
 *           供 LIOBuilder::undistortCloud() 做运动补偿(去畸变)
 */
void livox2pcl(const livox_ros_driver2::CustomMsg::ConstPtr &msg, pcl::PointCloud<pcl::PointXYZINormal>::Ptr out, int filter_num, double range_min, double range_max)
{
    int point_num = msg->point_num;           // 本帧总点数(含后续会被筛掉的点)
    out->clear();
    out->reserve(point_num / filter_num + 1); // 按抽帧间隔粗略预留容量, 避免反复扩容
                                              // NOTE: filter_num <= 0 时这里会除零(filter_num 来自 yaml 的 filter_num 参数, 函数内没做校验)
    uint valid_num = 0;                       // 通过线号/回波筛选的点计数, 只用于抽帧
    for (uint i = 0; i < point_num; i++)
    {
        if ((msg->points[i].line < 4) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) // 线号 + 回波标识双重筛选
        {
            if ((valid_num++) % filter_num != 0)
                continue; // 抽帧: 每 filter_num 个点保留 1 个(计数先自增, 与后面的距离筛选无关)
            pcl::PointXYZINormal p;
            p.x = msg->points[i].x;
            p.y = msg->points[i].y;
            p.z = msg->points[i].z;
            p.intensity = msg->points[i].reflectivity;                 // 反射率
            // curvature 被复用为"点相对本帧起始的偏移时间", 由纳秒换算成毫秒
            p.curvature = msg->points[i].offset_time / float(1000000); // 纳秒->毫秒
            double sq_range = p.x * p.x + p.y * p.y + p.z * p.z;       // 距离平方, 省一次开方
            if (sq_range > (range_min * range_min) && sq_range < range_max * range_max)
            {
                out->push_back(p);                                     // 只保留距离在 (range_min, range_max) 内的点
            }
        }
    }
}

/**
 * PCL 点云封装成 ROS 的 sensor_msgs::PointCloud2
 *
 * 步骤: 用 pcl::toROSMsg 转换字段与点位, 再填 header(时间戳与坐标系名)
 * timestamp < 0 视为"没有可用时间", 用当前 ROS 时间代替
 * 调用关系: lio_node.cpp 的 publishCloud() 调用, 发布 body_cloud / world_cloud 话题
 */
sensor_msgs::PointCloud2 pcl2msg(pcl::PointCloud<pcl::PointXYZINormal>::Ptr inp, const std::string &frame_id, const double &timestamp)
{
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*inp, msg);   // PCL -> ROS: 点位与字段定义一起转换
    if (timestamp < 0)
        msg.header.stamp = ros::Time().now();  // 负时间表示用当前时刻(等价于 ros::Time::now())
    else
        msg.header.stamp = ros::Time().fromSec(timestamp);  // 秒 -> ros::Time
    msg.header.frame_id = frame_id;
    return msg;
}

/**
 * Eigen 形式的刚体变换转 tf 的 TransformStamped
 *
 * 步骤: 填父子坐标系名与时间戳, 平移按分量写入, 旋转矩阵转成四元数后按 w/x/y/z 写入
 * 调用关系: lio_node.cpp 的 mainCB() 用它构造 map(世界) -> body(机体) 的 TF 并广播
 */
geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    geometry_msgs::TransformStamped transform;
    transform.header.frame_id = frame_id;        // 父坐标系(世界系)
    transform.header.stamp = ros::Time().fromSec(timestamp);
    transform.child_frame_id = child_frame_id;   // 子坐标系(机体系)
    transform.transform.translation.x = pos(0);
    transform.transform.translation.y = pos(1);
    transform.transform.translation.z = pos(2);
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);  // 旋转矩阵 -> 四元数

    transform.transform.rotation.w = q.w();
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    return transform;
}

/**
 * Eigen 形式的位姿转 nav_msgs::Odometry
 *
 * 步骤: 填 header(父子坐标系与时间戳), 再填 pose.pose 的位置与姿态四元数;
 *       twist 与 pose.covariance 保持默认值(全零), 本函数不填
 * NOTE: 本包内没有调用点; lio_node.cpp 里虽然 advertise 了 slam_odom 话题, 但从未发布过
 */
nav_msgs::Odometry eigen2Odometry(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    nav_msgs::Odometry odom;
    odom.header.frame_id = frame_id;
    odom.header.stamp = ros::Time().fromSec(timestamp);
    odom.child_frame_id = child_frame_id;
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);  // 旋转矩阵 -> 四元数
    odom.pose.pose.position.x = pos(0);
    odom.pose.pose.position.y = pos(1);
    odom.pose.pose.position.z = pos(2);

    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    return odom;
}

/**
 * jet 伪彩色映射
 *
 * v 先被截断到 [vmin, vmax], 然后按 5 段线性插值给出颜色, 整体走向是
 *   深蓝 -> 蓝 -> 青 -> 绿 -> 黄 -> 红(末端略微变暗)
 * 参数含义见 utils.h 的声明注释
 * 调用关系: voxel2MarkerArray() 用它把平面上色
 */
void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
    r = 255;
    g = 255;
    b = 255;    // NOTE: 这三行赋值在函数末尾会被重新覆盖, 属于残留代码

    if (v < vmin)
    {
        v = vmin;   // 截断到下界
    }

    if (v > vmax)
    {
        v = vmax;   // 截断到上界
    }

    double dr, dg, db;

    if (v < 0.1242)
    {
        db = 0.504 + ((1. - 0.504) / 0.1242) * v;   // 深蓝 -> 蓝: 只有蓝分量在变化
        dg = dr = 0.;
    }
    else if (v < 0.3747)
    {
        db = 1.;
        dr = 0.;
        dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));   // 蓝 -> 青: 绿分量线性上升
    }
    else if (v < 0.6253)
    {
        db = (0.6253 - v) * (1. / (0.6253 - 0.3747));   // 青 -> 绿 -> 黄: 红升蓝降, 绿保持 1
        dg = 1.;
        dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
    }
    else if (v < 0.8758)
    {
        db = 0.;
        dr = 1.;
        dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));   // 黄 -> 红: 绿分量线性下降
    }
    else
    {
        db = 0.;
        dg = 0.;
        dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));  // v 趋向 1 时红分量由 1 回落到 0.504(颜色略变暗)
    }

    r = (uint8_t)(255 * dr);    // 归一化分量 -> 0~255
    g = (uint8_t)(255 * dg);
    b = (uint8_t)(255 * db);
}

/**
 * 由三个坐标轴向量构造四元数
 *
 * 步骤: 三个向量按列拼成旋转矩阵 -> 转四元数 -> 归一化(消除数值误差) -> 填 ROS 四元数
 * NOTE: 本包内没有调用点; 且没有检查输入向量是否两两正交, 传入非正交向量时会得到有偏差的结果
 */
void calcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q)
{
    Eigen::Matrix3d rot;
    rot.col(0) = x_vec;   // 第 1 列 = x 轴向量
    rot.col(1) = y_vec;   // 第 2 列 = y 轴向量
    rot.col(2) = z_vec;   // 第 3 列 = z 轴向量
    Eigen::Quaterniond eq(rot);
    eq.normalize();       // 归一化
    q.w = eq.w();
    q.x = eq.x();
    q.y = eq.y();
    q.z = eq.z();
}

/**
 * 由平面法向量构造四元数(把 +Z 轴转过去与 norm_vec 对齐的最小旋转)
 *
 * 调用关系: voxel2MarkerArray() 用它给平面 marker 定向, marker 的局部 +Z 轴即平面法向
 */
void calcVectQuation(const Eigen::Vector3d &norm_vec, geometry_msgs::Quaternion &q)
{
    Eigen::Quaterniond rq = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(0, 0, 1), norm_vec);  // +Z -> norm_vec 的最小旋转
    q.w = rq.w();
    q.x = rq.x();
    q.y = rq.y();
    q.z = rq.z();
}

/**
 * 体素平面地图转成 rviz 可视化用的 MarkerArray
 *
 * 步骤:
 *   1. 取本次要检查的体素数量 size = min(cache_ 长度, max_capacity), 并预留 markers 容量
 *   2. 遍历 LRU 链表 map->cache_(头部是最久未被淘汰的一端), 只处理"已冻结的平面"体素:
 *      is_plane_ 为真且 update_enable_ 为假(即已经攒够点、平面参数不再变化)
 *   3. 用平面法向量协方差(左上 3x3 block)的迹衡量该平面是否可靠, 截断到 0.25 并归一化后
 *      再做 0.2 次幂拉伸, 最后交给 mapJet 映射成颜色: 迹越小(法向越确定)颜色越偏蓝
 *   4. 每个平面发一个 marker: 位置取体素平面中心, 姿态由平面法向得到(局部 Z 轴对齐法向),
 *      x/y 尺寸为 voxel_size, z 方向很薄(0.01), 因此看上去是一张贴合平面的薄片;
 *      与邻居合并过的体素画成 CUBE, 未合并的画成 CYLINDER, 便于区分
 *
 * 参数含义见 utils.h 的声明注释
 * 调用关系: lio_node.cpp 的 voxelTimerCB() 调用, 结果发布到 voxel_map 话题供 rviz 显示
 */
visualization_msgs::MarkerArray voxel2MarkerArray(std::shared_ptr<lio::VoxelMap> map, const std::string &frame_id, const double &timestamp, int max_capacity, double voxel_size)
{
    visualization_msgs::MarkerArray voxel_plane;
    int size = std::min(static_cast<int>(map->cache_.size()), max_capacity);  // 本次最多检查的体素数

    voxel_plane.markers.reserve(size);  // 预留, 最多也就发布 size 个 marker
    int count = 0;                      // 已发布的 marker 数, 同时用作 marker.id
    for (auto &k : map->cache_)
    {
        if (count >= size)
            break;  // 已经发够数量
        std::shared_ptr<lio::VoxelGrid> grid = map->featmap_[k];  // NOTE: 用的是 operator[], 若该 key 不存在会插入一个空的 shared_ptr, 随后解引用会崩; 这里假定 cache_ 与 featmap_ 始终同步
        if (!grid->is_plane_ || grid->update_enable_)
            continue;   // 跳过非平面体素, 以及还在更新的体素(只可视化已冻结的平面)

        Eigen::Vector3d grid_center = grid->center_;  // 平面中心(即拟合出的点云质心)

        double trace = grid->plane_->cov.block<3, 3>(0, 0).trace();   // 法向量协方差的迹, 越小说明法向越可信
        if (trace >= 0.25)
            trace = 0.25;                   // 截断到 0.25, 避免个别劣质平面污染色带
        trace = trace * (1.0 / 0.25);       // 归一化到 [0, 1]
        trace = pow(trace, 0.2);            // 幂次拉伸: 把低值区抬高, 否则大部分平面都会挤在深蓝端
        uint8_t r, g, b;
        mapJet(trace, 0, 1, r, g, b);       // 映射到 jet 色带
        Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);  // NOTE: 这里除的是 256 而不是 255, 颜色会略偏暗(影响很小)
        double alpha = 0.8;                 // 半透明

        visualization_msgs::Marker plane;
        plane.header.frame_id = frame_id;   // marker 所在坐标系
        plane.header.stamp = ros::Time().fromSec(timestamp);
        plane.ns = "plane";                 // marker 命名空间
        plane.id = count++;                 // 同一命名空间内 id 唯一
        if (!grid->merged_)
            plane.type = visualization_msgs::Marker::CYLINDER;   // 未参与合并: 画圆片
        else
            plane.type = visualization_msgs::Marker::CUBE;       // 参与过合并: 画方块
        plane.action = visualization_msgs::Marker::ADD;
        plane.pose.position.x = grid_center[0];
        plane.pose.position.y = grid_center[1];
        plane.pose.position.z = grid_center[2];
        geometry_msgs::Quaternion q;
        calcVectQuation(grid->plane_->norm, q);   // 姿态: 使 marker 局部 Z 轴对齐平面法向
        plane.pose.orientation = q;
        plane.scale.x = voxel_size;         // 与体素边长同量级
        plane.scale.y = voxel_size;
        plane.scale.z = 0.01;               // 很薄, 视觉上就是贴在平面上的一张片
        plane.color.a = alpha;
        plane.color.r = plane_rgb[0];
        plane.color.g = plane_rgb[1];
        plane.color.b = plane_rgb[2];
        plane.lifetime = ros::Duration();   // 零时长 = 永久有效(不自动消失)
        voxel_plane.markers.push_back(plane);
    }
    return voxel_plane;
}