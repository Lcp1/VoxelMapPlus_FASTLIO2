// 后端节点 pgo_node: 回环检测(STD 描述子) + 位姿图优化(GTSAM ISAM2)
// 数据流: 前端 lio_node 发布"去畸变点云 + 位姿" -> 本节点累积 STD 关键帧 ->
//         在历史关键帧里检索回环 -> 位姿图增量优化 -> 广播校正后的 map->lidar TF,
//         并发布原始轨迹 / 优化轨迹 / 回环可视化
#include <queue>
#include <chrono>
#include <ros/ros.h>
#include "std_manager/descriptor.h"        // STD 描述子提取、回环检索、几何与 ICP 验证
#include "interface/PointCloudWithOdom.h"  // 前端输入消息: 点云 + 位姿
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>

// GTSAM: Pose3 位姿、ISAM2 增量求解器、Values 初值容器、先验因子与相对位姿因子
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
// ROS 消息类型: TF、轨迹、位姿、回环可视化
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

/**
 * @brief 把旋转矩阵 + 平移向量打包成 ROS 的 TransformStamped
 *
 * @param rot            旋转矩阵(3x3), 按正交阵直接使用, 内部不做正交化
 * @param pos            平移向量
 * @param frame_id       父坐标系
 * @param child_frame_id 子坐标系
 * @param timestamp      时间戳(秒), 用前端点云的时间
 * @return 可直接交给 tf2 broadcaster 发送的变换消息
 */
geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    geometry_msgs::TransformStamped transform;
    transform.header.frame_id = frame_id;
    transform.header.stamp = ros::Time().fromSec(timestamp); // 秒 -> ros::Time
    transform.child_frame_id = child_frame_id;
    transform.transform.translation.x = pos(0);
    transform.transform.translation.y = pos(1);
    transform.transform.translation.z = pos(2);
    // 旋转矩阵 -> 四元数; 若传入的 rot 数值上不满足正交性, 得到的四元数也不会是单位四元数
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);

    transform.transform.rotation.w = q.w();
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    return transform;
}

/**
 * @brief 本节点自身的配置(与 STD 内部参数 std_desc::Config 区分开)
 *
 * 每个字段都能被 ROS 私有参数服务器上的同名参数覆盖, 见 loadParams()
 */
struct Config
{
    std::string map_frame = "map";                              // 地图/优化后轨迹所在的全局坐标系
    std::string local_frame = "lidar";                          // TF 子坐标系名(默认值形同虚设, 运行时会被输入消息的 header.frame_id 覆盖, 见 odomCloudCB)
    double ds_size = 0.25;                                      // 单帧点云的体素降采样边长(米)
    int sub_frame_num = 10;                                     // 一个 STD 关键帧累积多少个前端帧
    std::string odom_cloud_topic = "/lio_node/cloud_with_odom"; // 前端"点云 + 位姿"话题
};

/**
 * @brief 订阅回调(生产者)与定时器回调(消费者)之间共享的全部状态
 *
 * 三个队列由 buffer_mutex 保护; 位姿图(initial / graph / isam)和轨迹数组只在定时器回调里访问。
 * NOTE: 锁只覆盖了三个 buffer 队列, current_cloud / current_pose / current_time / key_cloud /
 *       pose_vec 等都在锁外读写。当前 main() 用的是单线程 ros::spin(), 订阅回调与定时器回调
 *       不会并发, 因此暂时不会竞争; 若改成 AsyncSpinner / MultiThreadedSpinner 则会出现数据竞争。
 */
struct DataGroup
{
    std::mutex buffer_mutex;                                       // 保护下面三个队列(生产者=订阅回调, 消费者=定时器回调)
    std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_buffer; // 待处理点云队列
    std::queue<Eigen::Affine3d> pose_buffer;                       // 与 cloud_buffer 一一对应的前端位姿队列
    std::queue<double> time_buffer;                                // 与 cloud_buffer 一一对应的时间戳(秒)队列

    double current_time;                                            // 本帧时间戳(秒)
    Eigen::Affine3d current_pose;                                   // 本帧前端位姿 T_world_body(雷达体系 -> 前端世界系)
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud;             // 本帧点云(处理时会被就地变换到世界系)
    pcl::PointCloud<pcl::PointXYZI>::Ptr key_cloud;                 // 当前 STD 关键帧累积的世界系点云

    size_t cloud_idx = 0;                                           // 已处理的前端帧计数, 同时用作 GTSAM 位姿图的节点键
    std::vector<Eigen::Affine3d> pose_vec;                          // 优化后的位姿, 下标 = 帧号
    std::vector<Eigen::Affine3d> origin_pose_vec;                   // 前端原始位姿, 下标 = 帧号
    std::vector<std::pair<int, int>> loop_container;                // 已检出的回环对(历史关键帧 ID, 当前关键帧 ID), 仅用于可视化

    gtsam::Values initial;                                          // 本次 update 的初值(新增节点的前端位姿)
    gtsam::NonlinearFactorGraph graph;                              // 本次 update 的新增因子(里程计 + 回环), 用完即清空
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise;          // 里程计/先验因子的对角噪声模型
    gtsam::noiseModel::Base::shared_ptr robustLoopNoise;            // 回环因子的鲁棒噪声模型; NOTE 见 initSAM()
    std::shared_ptr<gtsam::ISAM2> isam;                             // 增量式位姿图求解器

    bool has_loop_flag = false;                                     // 本帧是否检出回环: 置位后多跑几次 ISAM2 update 让它充分收敛
};

class PGONode
{
public:
    /**
     * @brief 构造函数: 按固定顺序完成初始化
     *
     * 顺序: loadParams() 读参数 -> 注册订阅 -> 注册发布 -> initSAM() 建位姿图 ->
     *       起 20 Hz 主循环定时器 -> 创建 STD 管理器
     *
     * 注意: 成员按声明顺序初始化, nh 声明在类尾, 所以它在构造函数体内被各方法使用时
     *       已经构造完毕; std_manager 在定时器之后才赋值, 但定时器回调只有 main() 里
     *       调用 ros::spin() 之后才会触发(那时构造函数已返回), 因此不存在空指针问题。
     */
    PGONode() : nh("~")
    {
        loadParams();
        initSubScribers();
        initPublishers();
        initSAM();
        main_loop = nh.createTimer(ros::Duration(0.05), &PGONode::mainLoopCB, this); // 50 ms 周期 = 20 Hz
        std_manager = std::make_shared<std_desc::STDManager>(std_config);            // 拷贝一份 STD 参数给管理器
    }

    /**
     * @brief 读取 ROS 私有参数("~" 命名空间)
     *
     * 分两类:
     *   - 本节点参数(与 Config 字段同名): map_frame / odom_cloud_topic / ds_size / sub_frame_num
     *   - STD 参数(与 std_desc::Config 字段同名), 由 STDManager 拷贝使用
     * nh.param 的第二个参数是"接收值的变量", 第三个参数是参数不存在时的默认值
     */
    void loadParams()
    {
        nh.param<std::string>("map_frame", node_config.map_frame, "map");
        nh.param<std::string>("odom_cloud_topic", node_config.odom_cloud_topic, "/lio_node/cloud_with_odom");
        nh.param<double>("ds_size", node_config.ds_size, 0.25);                  // 单帧降采样体素边长(米)
        nh.param<int>("sub_frame_num", node_config.sub_frame_num, 10);           // 每个 STD 关键帧累积的前端帧数

        nh.param<double>("voxel_size", std_config.voxel_size, 1.0);              // 体素边长(米), 平面拟合的基本单元
        nh.param<int>("voxel_min_point", std_config.voxel_min_point, 10);        // 体素内点数需大于它才参与平面拟合
        nh.param<double>("voxel_plane_thresh", std_config.voxel_plane_thresh, 0.01); // 最小特征值小于它则判为平面
        nh.param<double>("norm_merge_thresh", std_config.norm_merge_thresh, 0.1);    // 相邻体素法向量一致性阈值(1-cos 量级)

        nh.param<double>("proj_2d_resolution", std_config.proj_2d_resolution, 0.25); // 角点投影到 2D 网格的分辨率(米)
        nh.param<double>("proj_min_dis", std_config.proj_min_dis, 0.0001);           // 平面法向竖直分量的下限(用于滤掉近水平面)
        nh.param<double>("proj_max_dis", std_config.proj_max_dis, 5.0);              // 平面法向竖直分量的上限

        nh.param<int>("nms_2d_range", std_config.nms_2d_range, 5);                   // 2D NMS 的网格段尺寸(格数)
        nh.param<double>("nms_3d_range", std_config.nms_3d_range, 2.0);              // 3D NMS 的邻域半径(米)
        nh.param<double>("corner_thresh", std_config.corner_thresh, 10.0);           // 2D 梯度阈值, 大于它才判为角点
        nh.param<int>("max_corner_num", std_config.max_corner_num, 100);             // 每帧保留的角点数上限
        nh.param<double>("min_side_len", std_config.min_side_len, 2.0);              // 三角形最短边长(米)
        // NOTE: 这里把参数 max_side_len 读进了 std_config.proj_max_dis (疑似复制粘贴错误, 本意应是
        //       std_config.max_side_len)。后果: 上一行刚读到的 proj_max_dis 被本行覆盖(config.yaml 里的
        //       proj_max_dis: 0.0 因此不生效), 而 max_side_len 字段始终是结构体默认值 30.0
        nh.param<double>("max_side_len", std_config.proj_max_dis, 30.0);             // 三角形最长边长(米)

        nh.param<int>("desc_search_range", std_config.desc_search_range, 15);        // 构描述子时每个角点搜索的近邻角点数

        nh.param<double>("side_resolution", std_config.side_resolution, 0.2);        // 三角形边长的哈希分辨率(米)
        nh.param<double>("rough_dis_threshold", std_config.rough_dis_threshold, 0.03); // 候选检索时边长相对差阈值
        nh.param<int>("skip_near_num", std_config.skip_near_num, 50);                // 检索时跳过的近邻关键帧数(按关键帧 ID)
        nh.param<int>("candidate_num", std_config.candidate_num, 50);                // 候选匹配帧数量上限
        nh.param<double>("vertex_diff_threshold", std_config.vertex_diff_threshold, 0.7); // 顶点附加距离的相对差阈值
        nh.param<double>("verify_dis_thresh", std_config.verify_dis_thresh, 3.0);    // 验证时三角形顶点距离一致性阈值(米)
        nh.param<double>("geo_verify_dis_thresh", std_config.geo_verify_dis_thresh, 0.3); // 几何验证的点到面距离阈值(米)
        nh.param<double>("icp_thresh", std_config.icp_thresh, 0.5);                  // 匹配置信度阈值, 超过才认为回环有效

        nh.param<double>("iter_eps", std_config.iter_eps, 0.001);                    // ICP 迭代增量收敛阈值
        // NOTE: std_config.max_iter(ICP 最大迭代次数)没有被本节点读取, 沿用结构体默认值 10,
        //       config.yaml 里的 max_iter 对本节点不生效
    }

    /**
     * @brief 初始化 ISAM2 与两类噪声模型(位姿图的固定部分)
     *
     * 位姿图的节点键就是帧号 cloud_idx, 节点值是该帧优化后的位姿 T_world_body(前端世界系下的位姿)。
     *  - odometryNoise: 前 3 维是旋转、后 3 维是平移(切空间上的对角方差, 单位 rad^2 / m^2),
     *                   1e-6 / 1e-4 对应转角标准差约 1e-3 rad、平移标准差约 1e-2 m
     *  - robustLoopNoise: 6 维方差都取 0.1, 外层再套 Cauchy 鲁棒核(k = 1), 用于压低错误回环的影响
     *  - ISAM2Params: relinearizeThreshold 0.01, relinearizeSkip 1(每次 update 都检查是否重线性化)
     */
    void initSAM()
    {
        data_group.odometryNoise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished()); // 前 3 维旋转方差, 后 3 维平移方差
        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1; // 回环因子 6 维对角方差(标准差的平方)
        // NOTE: 下面这个 robustLoopNoise 是局部变量, 遮蔽了同名成员 DataGroup::robustLoopNoise;
        //       成员从头到尾没有被赋值(一直是空的 shared_ptr), 而主循环加回环因子时用的恰恰是成员。
        //       本意应是把结果赋给 data_group.robustLoopNoise, 否则回环因子会拿到空噪声模型
        gtsam::noiseModel::Base::shared_ptr robustLoopNoise =
            gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1), gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01; // 线性化点变化超过该幅度就重线性化
        parameters.relinearizeSkip = 1;         // 每次 update 都做一次重线性化检查
        data_group.isam = std::make_shared<gtsam::ISAM2>(parameters);
    }

    /**
     * @brief 注册订阅者: 前端"点云 + 位姿"话题
     *
     * 队列长度 100, 回调绑定到 this
     */
    void initSubScribers()
    {
        odom_cloud_sub = nh.subscribe(node_config.odom_cloud_topic, 100, &PGONode::odomCloudCB, this);
    }

    /**
     * @brief 把一整条轨迹发成 nav_msgs/Path
     *
     * @param pub      目标发布器
     * @param pose_list 位姿序列(T_world_body), 下标即帧号
     *
     * 无订阅者时直接返回以省去消息构造; 每个元素取位姿的平移与旋转(四元数)填 PoseStamped,
     * 整条路径在 path.header 上打一次时间戳和坐标系。
     * NOTE: 每个 msg_pose.header.stamp 没有赋值(保持 0), 只有 path.header.stamp 有效
     */
    void publishPath(ros::Publisher &pub, std::vector<Eigen::Affine3d> &pose_list)
    {
        if (pub.getNumSubscribers() < 1)
            return;
        nav_msgs::Path path;
        for (int i = 0; i < pose_list.size(); i++)
        {
            geometry_msgs::PoseStamped msg_pose;
            msg_pose.pose.position.x = pose_list[i].translation()[0];
            msg_pose.pose.position.y = pose_list[i].translation()[1];
            msg_pose.pose.position.z = pose_list[i].translation()[2];
            Eigen::Quaterniond pose_q(pose_list[i].rotation()); // 旋转矩阵 -> 四元数
            msg_pose.header.frame_id = node_config.map_frame;
            msg_pose.pose.orientation.x = pose_q.x();
            msg_pose.pose.orientation.y = pose_q.y();
            msg_pose.pose.orientation.z = pose_q.z();
            msg_pose.pose.orientation.w = pose_q.w();
            path.poses.push_back(msg_pose);
        }

        path.header.stamp = ros::Time().fromSec(data_group.current_time);
        path.header.frame_id = node_config.map_frame;
        pub.publish(path);
    }

    /**
     * @brief 发布回环可视化: 回环两端的位姿点(SPHERE_LIST) + 连线(LINE_LIST)
     *
     * 位置直接取 pose_vec 里对应的优化后位姿; loop_container 里的每个元素是
     * (历史关键帧 ID, 当前关键帧 ID), 关键帧 ID 与帧号的映射约定见 mainLoopCB()。
     */
    void publishLoopConstraints()
    {
        if (loop_contraints_pub.getNumSubscribers() < 1)
            return;
        if (data_group.loop_container.size() == 0)
            return;
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker_node;
        marker_node.header.frame_id = node_config.map_frame;
        marker_node.action = visualization_msgs::Marker::ADD;
        marker_node.type = visualization_msgs::Marker::SPHERE_LIST; // 回环两端的位置点
        marker_node.ns = "loop_nodes";
        marker_node.id = 0;
        marker_node.pose.orientation.w = 1;
        marker_node.scale.x = 0.3;
        marker_node.scale.y = 0.3;
        marker_node.scale.z = 0.3;
        marker_node.color.r = 0;
        marker_node.color.g = 0.8;
        marker_node.color.b = 1;
        marker_node.color.a = 1;

        visualization_msgs::Marker marker_edge;
        marker_edge.header.frame_id = node_config.map_frame;
        marker_edge.action = visualization_msgs::Marker::ADD;
        marker_edge.type = visualization_msgs::Marker::LINE_LIST; // 回环两端的连线
        marker_edge.ns = "loop_edges";
        marker_edge.id = 1;
        marker_edge.pose.orientation.w = 1;
        marker_edge.scale.x = 0.1; // 线宽
        marker_edge.color.r = 0.9;
        marker_edge.color.g = 0.9;
        marker_edge.color.b = 0;
        marker_edge.color.a = 1;

        for (auto it = data_group.loop_container.begin(); it != data_group.loop_container.end(); ++it)
        {
            // NOTE: 命名与存入顺序相反: loop_container 里 emplace_back(result.match_id, feature.id),
            //       所以 first 其实是历史(被匹配到)的关键帧 ID, second 才是当前关键帧 ID
            int key_cur = it->first;
            int key_pre = it->second;
            // NOTE: 关键帧 k 实际覆盖的帧号区间是 (k * sub_frame_num, (k+1) * sub_frame_num],
            //       这里用 k * sub_frame_num 作为代表帧, 比区间起点还早一帧; 仅用于画示意图
            geometry_msgs::Point p;
            p.x = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().x();
            p.y = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().y();
            p.z = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().z();
            marker_node.points.push_back(p);
            marker_edge.points.push_back(p);
            p.x = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().x();
            p.y = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().y();
            p.z = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().z();
            marker_node.points.push_back(p);
            marker_edge.points.push_back(p);
        }

        marker_array.markers.push_back(marker_node);
        marker_array.markers.push_back(marker_edge);
        loop_contraints_pub.publish(marker_array);
    }

    /**
     * @brief 注册三个发布器(都在私有命名空间下)
     */
    void initPublishers()
    {
        path_pub = nh.advertise<nav_msgs::Path>("origin_path", 10000);                    // 前端原始轨迹
        correct_path_pub = nh.advertise<nav_msgs::Path>("correct_path", 10000);           // 优化后的轨迹
        loop_contraints_pub = nh.advertise<visualization_msgs::MarkerArray>("loop_contriants", 10); // 回环可视化(话题名沿用源码拼写)
    }

    /**
     * @brief 前端里程计回调: 把一帧"点云 + 位姿"压进缓冲队列(生产者)
     *
     * @param msg 前端 lio_node 发布的 interface::PointCloudWithOdom
     *
     * 步骤: 1) ROS 点云消息 -> pcl 点云; 2) 记录点云所在坐标系(作为最终 TF 的子坐标系);
     *       3) 位姿(四元数 + 平移)转成 Eigen::Affine3d 入队; 4) 时间戳入队。
     * 整个函数由 buffer_mutex 保护, 三个队列的入队顺序与 mainLoopCB 的取出顺序一致。
     */
    void odomCloudCB(const interface::PointCloudWithOdom::ConstPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_group.buffer_mutex);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(msg->cloud, *cloud);
        // NOTE: msg->header.frame_id 是外层消息头, 前端 lio_node 往这里填的是"位姿所在的世界系名"
        //       (voxel_plus 的配置里恰好叫 "lidar"), 并不是点云自身的机体系(机体系写在内层 cloud.header 里)。
        //       该值在 mainLoopCB 末尾被当作 TF 的子坐标系名使用, 阅读时不要与机体系混淆
        node_config.local_frame = msg->header.frame_id; // 覆盖默认值, 作为 map -> local_frame 的子坐标系名
        data_group.cloud_buffer.push(cloud);
        // 四元数构造顺序是 (w, x, y, z)
        Eigen::Quaterniond rotation(msg->pose.pose.orientation.w,
                                    msg->pose.pose.orientation.x,
                                    msg->pose.pose.orientation.y,
                                    msg->pose.pose.orientation.z);
        Eigen::Vector3d translation(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.linear() = rotation.toRotationMatrix();
        pose.translation() = translation;
        data_group.pose_buffer.push(pose);
        data_group.time_buffer.push(msg->header.stamp.toSec());
    }

    /**
     * @brief 主循环(20 Hz 定时器回调): 处理一帧 -> 回环检测 -> 位姿图优化 -> 发布
     *
     * 步骤:
     *   1) 加锁从队列取一帧(保留队首, 其余丢弃), 得到 current_cloud / current_pose / current_time
     *   2) 点云变换到世界系 + 体素降采样, 累积进当前 STD 关键帧(key_cloud)
     *   3) 向位姿图加入初值以及先验因子(第 0 帧)或里程计因子(相邻帧相对位姿)
     *   4) 每满 sub_frame_num 帧抽一次 STD 特征并检索回环, 命中则做 ICP 精配准并加入回环因子
     *   5) ISAM2 增量更新, 用优化结果回填 pose_vec
     *   6) 广播校正后的 map -> lidar TF, 发布原始/优化轨迹与回环可视化
     */
    void mainLoopCB(const ros::TimerEvent &e)
    {
        {
            std::lock_guard<std::mutex> lock(data_group.buffer_mutex);
            if (data_group.cloud_buffer.size() < 1)
                return; // 还没有数据, 等下个周期
            data_group.current_cloud = data_group.cloud_buffer.front(); // 取队首(最早到达的一帧)
            data_group.current_pose = data_group.pose_buffer.front();
            data_group.current_time = data_group.time_buffer.front();
            // NOTE: 这里把队列一次清空, 队首之外的帧全部被丢弃。前端频率高于 20 Hz 时会丢帧,
            //       cloud_idx 只统计"实际处理的帧数", 与前端帧号不再一一对应
            while (!data_group.cloud_buffer.empty())
            {
                data_group.cloud_buffer.pop();
                data_group.pose_buffer.pop();
                data_group.time_buffer.pop();
            }
        }
        // 点云变换到前端世界系(T_world_body), 就地修改
        pcl::transformPointCloud(*data_group.current_cloud, *data_group.current_cloud, data_group.current_pose);

        std_desc::voxelFilter(data_group.current_cloud, node_config.ds_size); // 单帧体素降采样, 控制累积点数

        if (data_group.key_cloud == nullptr)
            data_group.key_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        // 累积到当前 STD 关键帧; 因为每帧点云已变换到世界系, 关键帧点云以及后续描述子
        // (三角形顶点/中心)都直接表达在世界系下, 这也是回环能估计"两个世界系之间偏差"的前提
        *data_group.key_cloud += *data_group.current_cloud;

        Eigen::Affine3d pose = data_group.current_pose;
        // (下面两行是位姿还是 rot/trans 分离类型时的遗留写法, 保留不动)
        // pose.linear() = data_group.current_pose.first;
        // pose.translation() = data_group.current_pose.second;
        data_group.initial.insert(data_group.cloud_idx, gtsam::Pose3(pose.matrix())); // 初值 = 前端位姿, 键 = 帧号
        if (!data_group.cloud_idx)
        {
            // 第 0 帧: 加先验因子固定全局基准, 否则位姿图的整体位姿不可观
            data_group.graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(pose.matrix()), data_group.odometryNoise));
        }
        else
        {
            // 里程计因子: 测量取前端相邻两帧的相对位姿, prev_pose.between(curr_pose) = T_{i-1}^{-1} * T_i,
            // 语义是 T_i = T_{i-1} * measurement; 因此优化前解与前端轨迹一致, 只有回环会把它拉弯
            auto prev_pose = gtsam::Pose3(data_group.origin_pose_vec[data_group.cloud_idx - 1].matrix());
            auto curr_pose = gtsam::Pose3(pose.matrix());
            data_group.graph.add(gtsam::BetweenFactor<gtsam::Pose3>(data_group.cloud_idx - 1, data_group.cloud_idx,
                                                                    prev_pose.between(curr_pose), data_group.odometryNoise));
        }

        data_group.origin_pose_vec.push_back(pose); // 原始轨迹(前端)
        data_group.pose_vec.push_back(pose);        // 优化轨迹, 先塞入前端位姿, 稍后被优化结果覆盖

        if (data_group.cloud_idx % node_config.sub_frame_num == 0 && data_group.cloud_idx != 0)
        {
            // (以下是被注释掉的调试代码: 把第 660 帧的累积点云存盘, 保留原样)
            // if (data_group.cloud_idx == 660)
            // {
            //     pcl::PCDWriter writer;
            //     writer.write("/home/zhouzhou/temp/660_new.pcd", *data_group.key_cloud);
            // }
            std_desc::STDFeature feature = std_manager->extract(data_group.key_cloud); // 由累积点云提取 STD 描述子
            ROS_INFO("ID: %lu  FEATRUE SIZE: %lu CLOUD SIZE: %lu", data_group.cloud_idx, feature.descs.size(), data_group.key_cloud->size());
            std_desc::LoopResult result;

            int64_t duration;
            // NOTE: duration 未初始化, 只有真的调用 searchLoop 时才会赋值;
            //       目前只在 result.valid 的分支被使用, 恰好不会读到未初始化值
            // NOTE: 下面这个判断的量纲不一致: cloud_idx 是"已处理的帧计数", 而 skip_near_num 在 STD 内部
            //       是按关键帧 ID 之差比较的(见 selectCanidates), 两者相差 sub_frame_num 倍
            if (data_group.cloud_idx > std_config.skip_near_num)
            {
                auto start = std::chrono::high_resolution_clock::now();
                result = std_manager->searchLoop(feature); // 在已入库的历史关键帧里检索回环
                auto end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            }

            std_manager->insert(feature); // 入库: 检索在入库之前, 保证不会匹配到自己
            if (result.valid)
            {

                // 这里得到是新旧世界坐标系下的差值 T_old_new;
                ROS_WARN("FIND MATCHED LOOP! CURRENT_ID: %lu, LOOP_ID: %lu MATCH SCORE: %.4f, TIME COST: %lu ms", feature.id, result.match_id, result.match_score, duration);
                // ICP 精配准: 以描述子给出的初值为起点继续优化, 结果通过引用原地写回
                // result.rotation / result.translation, 因此下面用的已经是精配准后的值
                // NOTE: 返回值 score(平均点面残差)没有被使用, 无法据此再拒绝一次质量差的回环
                double score = std_manager->verifyGeoPlaneICP(feature.cloud, std_manager->cloud_vec[result.match_id], result.rotation, result.translation);
                data_group.has_loop_flag = true;
                data_group.loop_container.emplace_back(result.match_id, feature.id); // 记录(历史关键帧 ID, 当前关键帧 ID)
                // 10 20
                // 回环要覆盖整个关键帧(共 sub_frame_num 帧), 而不只是当前这一帧, 所以逐帧加约束。
                // 关键帧 ID 与帧号的映射约定: 第 k 个 STD 关键帧覆盖帧号 (k * sub_frame_num, (k+1) * sub_frame_num]
                for (size_t j = 1; j <= node_config.sub_frame_num; j++)
                {
                    // 当前帧
                    int src_frame = data_group.cloud_idx + j - node_config.sub_frame_num;
                    // 历史帧
                    int tar_frame = result.match_id * node_config.sub_frame_num + j;

                    // delta_pose = T_old_new: 把"当前世界系"下的点变换到"历史世界系"的位姿
                    // (ICP 内部对当前关键帧点云做 rot * p + trans 后到历史关键帧点云里找最近点)
                    Eigen::Affine3d delta_pose = Eigen::Affine3d::Identity();
                    delta_pose.linear() = result.rotation;
                    delta_pose.translation() = result.translation;

                    // refined_src = 当前帧位姿换算到历史世界系下的位置: T_old_new * T_new_src
                    Eigen::Affine3d refined_src = delta_pose * data_group.origin_pose_vec[src_frame];
                    Eigen::Affine3d tar_pose = data_group.origin_pose_vec[tar_frame]; // 历史帧位姿(本就在历史世界系下)

                    // 因子测量 = tar_pose^{-1} * refined_src = T_tar^{-1} * T_src, 正好是
                    // BetweenFactor(tar_frame, src_frame, m) 要求的相对位姿 T_src = T_tar * m;
                    // 两个位姿都被表达在同一个(历史)世界系里, 所以这里只引入相对约束, 与全局基准无关
                    data_group.graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        tar_frame, src_frame,
                        gtsam::Pose3(tar_pose.matrix()).between(gtsam::Pose3(refined_src.matrix())),
                        data_group.robustLoopNoise));
                }
            }
            data_group.key_cloud->clear(); // 关键帧落地, 清空累积点云, 下一个关键帧重新累积
        }

        // 把本帧新增的因子与初值交给 ISAM2; 无参 update() 只继续迭代/重线性化, 不新增因子
        data_group.isam->update(data_group.graph, data_group.initial);
        data_group.isam->update();

        if (data_group.has_loop_flag)
        {
            // 检出回环时再多跑几次 update, 让新增的闭环约束在增量消元中充分传播
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
        }

        data_group.graph.resize(0); // 因子已交给 ISAM2, 清空以便下一帧继续攒
        data_group.initial.clear();

        gtsam::Values curr_estimates = data_group.isam->calculateEstimate(); // 取出全部节点的当前最优估计

        assert(curr_estimates.size() == data_group.pose_vec.size()); // 断言键与帧号一一对应(NEDEBUG 下不生效)

        for (int i = 0; i < curr_estimates.size(); i++)
        {
            gtsam::Pose3 est = curr_estimates.at<gtsam::Pose3>(i); // 键就是帧号, 可直接按下标取
            Eigen::Affine3d est_affine3d(est.matrix());
            data_group.pose_vec[i] = est_affine3d; // 回填优化后的位姿
        }

        Eigen::Affine3d last_pose = data_group.pose_vec.back(); // 最新一帧的优化后位姿

        // 校正变换 = T_map_last(优化后) * T_odom_last^{-1}: 把前端世界系里的位姿整体搬到优化后的地图系
        Eigen::Affine3d frame_delta_pose = last_pose * data_group.origin_pose_vec.back().inverse();

        // 广播 map -> local_frame 的 TF: frame_delta_pose 正是"前端世界系 -> 优化后地图系"的修正量,
        // 所以子坐标系取的是输入消息外层 frame_id(前端世界系名), 而不是机体系名;
        // 另外只广播当前时刻, 历史时刻的 TF 不会回填, 因此回环发生时位姿会瞬间跳变
        br.sendTransform(eigen2Transform(frame_delta_pose.linear(), frame_delta_pose.translation(), node_config.map_frame, node_config.local_frame, data_group.current_time));

        publishPath(path_pub, data_group.origin_pose_vec); // 前端原始轨迹

        publishPath(correct_path_pub, data_group.pose_vec); // 优化后的轨迹

        publishLoopConstraints(); // 回环可视化

        data_group.cloud_idx++; // 帧计数自增, 同时也是下一个 GTSAM 节点的键

        data_group.has_loop_flag = false; // 复位, 等待下一次回环
    }

public:
    ros::NodeHandle nh;                                // 私有句柄("~"), 读取本节点的私有 ROS 参数
    Config node_config;                                // 本节点配置
    tf2_ros::TransformBroadcaster br;                  // 广播校正后的 map -> lidar TF
    ros::Timer main_loop;                              // 20 Hz 主循环定时器
    ros::Subscriber odom_cloud_sub;                    // 前端"点云 + 位姿"订阅者
    DataGroup data_group;                              // 共享数据(缓冲区、位姿图、轨迹)
    std_desc::Config std_config;                       // STD 内部参数, 由 loadParams() 填充后拷贝给 STDManager
    std::shared_ptr<std_desc::STDManager> std_manager; // STD 描述子提取与回环检索管理器

    ros::Publisher correct_path_pub;   // 优化后的轨迹
    ros::Publisher path_pub;           // 前端原始轨迹
    ros::Publisher loop_contraints_pub; // 回环可视化
};

/**
 * @brief 节点入口: 初始化 ROS, 构造 PGONode 后进入单线程 spin
 *
 * ros::spin() 是单线程派发, 订阅回调与定时器回调不会并发执行(参见 DataGroup 的锁说明)
 */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "pgo_node");
    PGONode pgo_node;
    ros::spin();
    return 0;
}