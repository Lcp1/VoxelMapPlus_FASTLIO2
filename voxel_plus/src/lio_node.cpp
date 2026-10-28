#include <ros/ros.h>
#include <queue>    // 本文件用到 std::deque 与 std::mutex, 靠它以及其它头文件间接引入(严格说应直接 include <deque> / <mutex>)
#include "utils.h"
#include "map_builder/commons.h"
#include "map_builder/lio_builder.h"
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_broadcaster.h>
#include "interface/PointCloudWithOdom.h"

/**
 * 节点参数集合, 全部由 LIONode::loadConfig() 从参数服务器(默认私有命名空间)读取
 * 话题名与坐标系名见各成员后面的注释
 */
struct NodeConfig
{
    std::string lidar_topic;        // 雷达话题名(参数 lidar_topic, 默认 /livox/lidar), 消息类型必须是 livox_ros_driver2::CustomMsg
    std::string imu_topic;          // IMU 话题名(参数 imu_topic, 默认 /livox/imu)
    std::string map_frame;          // 世界(地图)坐标系名(参数 map_frame, 默认 "map")
    std::string body_frame;         // 机体坐标系名(参数 body_frame, 默认 "body")
    double range_min = 0.5;         // 有效点最近距离(米), 传给 livox2pcl
    double range_max = 20.0;        // 有效点最远距离(米), 传给 livox2pcl
    int filter_num = 3;             // 点云抽帧间隔, 传给 livox2pcl
    bool publish_voxel_map = false; // 是否启用体素平面可视化定时器
    int publish_voxel_num = 1000;   // 每次可视化最多发布多少个平面体素
};

/**
 * 节点的运行时数据: 雷达/IMU 缓冲与其时间戳, 供 syncPackage() 做帧同步
 */
struct NodeGroupData
{
    double last_imu_time = 0.0;     // 最近一帧 IMU 的时间(秒), 也用作"IMU 是否够用"的判断依据
    double last_lidar_time = 0.0;   // 最近一帧雷达的时间(秒), 用于检测时间倒流
    std::mutex imu_mutex;           // IMU 缓冲的互斥锁
    std::mutex lidar_mutex;         // 雷达缓冲的互斥锁
    std::deque<lio::IMUData> imu_buffer;                                                    // IMU 队列, 按时间递增
    std::deque<std::pair<double, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>> lidar_buffer;  // 点云队列, 每项是(帧时间, 去程的点云)
    bool lidar_pushed = false;      // sync_pack_ 里是否已经放好了一帧点云(避免重复取同一帧)
};

/**
 * 前端节点主体: 参数加载 -> 订阅雷达/IMU -> 定时把两路数据同步成帧交给 LIOBuilder -> 发布结果
 *
 * 数据流:
 *   imuCB/lidarCB 只负责缓冲原始数据
 *   mainCB(50Hz) 调 syncPackage() 攒出一帧 lio::SyncPackage, 交给 map_builder_.process() 做
 *   去畸变 + 迭代 EKF 更新 + 地图维护, 然后把 TF、机体系点云、世界系点云与位姿发布出去
 *   voxelTimerCB(5Hz) 则周期性把体素平面地图发成 MarkerArray 供 rviz 显示
 */
class LIONode
{
public:
    /**
     * 构造函数: 完成节点初始化
     *
     * 顺序很重要:
     *   1. loadConfig() 先把参数读进来
     *   2. initSubScribers() / initPublishers() 建立话题通道
     *   3. map_builder_.loadConfig() 用参数初始化算法主体(内部会创建体素地图)
     *   4. 最后才创建定时器, 保证定时器第一次触发时一切都已就绪
     * 句柄用 "~" 构造, 所以参数与话题都落在私有命名空间 /lio_node 下
     */
    LIONode() : nh_("~")
    {
        loadConfig();
        initSubScribers();
        initPublishers();
        map_builder_.loadConfig(lio_config_);                                                     // 把参数交给 LIOBuilder(内部创建 VoxelMap 与滤波器)
        main_loop_ = nh_.createTimer(ros::Duration(0.02), &LIONode::mainCB, this);                 // 主定时器: 50Hz
        voxel_map_loop_ = nh_.createTimer(ros::Duration(5.0), &LIONode::voxelTimerCB, this, false, false);  // 可视化定时器: 5s 一次, oneshot=false, autostart=false(默认不启动)
        if (config_.publish_voxel_map)
            voxel_map_loop_.start();                                                             // 只有需要可视化时才真正启动它
    }

    /**
     * 从参数服务器读取全部参数
     *
     * 前 9 个是节点自身要用的(话题名、坐标系名、预处理与可视化参数), 写进 config_;
     * 后面一批是算法参数, 写进 lio_config_, 最终由 map_builder_.loadConfig() 消费
     * 所有参数都有默认值, 读不到时使用默认值(参数文件见 voxel_plus/config/mapping.yaml)
     */
    void loadConfig()
    {
        nh_.param<std::string>("lidar_topic", config_.lidar_topic, "/livox/lidar");   // 雷达话题(绝对名, 不受私有命名空间影响)
        nh_.param<std::string>("imu_topic", config_.imu_topic, "/livox/imu");         // IMU 话题

        nh_.param<std::string>("body_frame", config_.body_frame, "body");             // 机体系名, 用于 TF 子坐标系与点云 header
        nh_.param<std::string>("map_frame", config_.map_frame, "map");                // 世界系名, 用于 TF 父坐标系与点云 header

        nh_.param<int>("filter_num", config_.filter_num, 3);                          // 点云抽帧间隔
        nh_.param<double>("range_min", config_.range_min, 0.5);                       // 有效点最近距离(米)
        nh_.param<double>("range_max", config_.range_max, 20.0);                      // 有效点最远距离(米)
        nh_.param<bool>("publish_voxel_map", config_.publish_voxel_map, false);       // 是否可视化体素平面
        nh_.param<int>("publish_voxel_num", config_.publish_voxel_num, 1000);         // 可视化的体素数量上限

        nh_.param<double>("scan_resolution", lio_config_.scan_resolution, 0.1);       // 输入点云的体素降采样边长(0 表示不降采样)
        nh_.param<double>("voxel_size", lio_config_.voxel_size, 0.4);                 // 地图体素边长(米)

        nh_.param<double>("merge_thresh_for_angle", lio_config_.merge_thresh_for_angle, 0.1);       // 平面合并的法向夹角阈值(1-cos)
        nh_.param<double>("merge_thresh_for_distance", lio_config_.merge_thresh_for_distance, 0.04); // 平面合并的原点到平面距离差阈值

        nh_.param<int>("max_point_thresh", lio_config_.max_point_thresh, 100);        // 单个体素保留的点数上限, 达到后冻结该平面
        nh_.param<int>("update_size_thresh", lio_config_.update_size_thresh, 10);     // 每新增多少个点重算一次平面
        // NOTE: config/mapping.yaml 里的键名是 update_size_threshes(多一个 es), 与此处不符, 该 yaml 项实际不会被读到, 生效的是默认值 10

        nh_.param<int>("map_capacity", lio_config_.map_capacity, 100000);             // 地图体素数量上限, 超出后按 LRU 淘汰
        nh_.param<bool>("gravity_align", lio_config_.gravity_align, true);            // 初始化时是否用重力方向对齐姿态
        nh_.param<int>("imu_init_num", lio_config_.imu_init_num, 20);                 // IMU 初始化所需的帧数
        nh_.param<double>("na", lio_config_.na, 0.01);                                // 加速度计白噪声
        nh_.param<double>("ng", lio_config_.ng, 0.01);                                // 陀螺仪白噪声
        nh_.param<double>("nbg", lio_config_.nbg, 0.0001);                            // 陀螺仪零偏随机游走
        nh_.param<double>("nba", lio_config_.nba, 0.0001);                            // 加速度计零偏随机游走
        nh_.param<int>("opti_max_iter", lio_config_.opti_max_iter, 5);                // 迭代 EKF 每帧最大迭代次数
        std::vector<double> r_il, p_il;
        nh_.param<std::vector<double>>("r_il", r_il, std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});  // 雷达->IMU 旋转外参(行优先 3x3)
        assert(r_il.size() == 9);                                                   // 长度校验(定义了 NDEBUG 时会被跳过)
        lio_config_.r_il << r_il[0], r_il[1], r_il[2], r_il[3], r_il[4], r_il[5], r_il[6], r_il[7], r_il[8];

        nh_.param<std::vector<double>>("p_il", p_il, std::vector<double>{0, 0, 0});  // 雷达->IMU 平移外参
        assert(p_il.size() == 3);                                                   // 长度校验
        lio_config_.p_il << p_il[0], p_il[1], p_il[2];

        // NOTE: lio_config_ 里还有几个字段没有对应的 nh_.param, 只能用 LIOConfig 结构体里的默认值, 无法用 yaml 调:
        //       plane_thresh(平面判定阈值, 默认 0.01)、ranging_cov(0.04)、angle_cov(0.1)、estimate_ext(false)
    }

    /**
     * 订阅雷达与 IMU
     *
     * 队列长度都取 10000: 回调只做缓冲, 真正处理在 mainCB 里, 队列大一些可以容忍短暂的处理延迟;
     * 话题名取自参数, 默认是绝对话题 /livox/lidar 与 /livox/imu
     */
    void initSubScribers()
    {
        lidar_sub_ = nh_.subscribe(config_.lidar_topic, 10000, &LIONode::lidarCB, this);
        imu_sub_ = nh_.subscribe(config_.imu_topic, 10000, &LIONode::imuCB, this);
    }

    /**
     * 创建发布者
     *
     * 话题都发布在节点私有命名空间下(句柄是 nh_("~")), 例如 slam_odom 实际是 /lio_node/slam_odom;
     * 下游 std_matcher 按 /lio_node/cloud_with_odom 订阅
     */
    void initPublishers()
    {
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("slam_odom", 1000);                                  // NOTE: 只 advertise 没有 publish, 该话题实际不会输出任何消息
        body_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("body_cloud", 1000);                     // 机体系点云
        world_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("world_cloud", 1000);                   // 世界系点云
        voxel_map_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("voxel_map", 1000);                // 体素平面标记数组
        pointcloud_with_odom_pub_ = nh_.advertise<interface::PointCloudWithOdom>("cloud_with_odom", 1000); // 机体系点云 + 世界系位姿, 给后端做回环
    }

    /**
     * IMU 回调: 把一帧 IMU 塞进缓冲
     *
     * 步骤: 加锁 -> 取时间戳 -> 若时间倒流则清空缓冲(防止乱序数据污染后续积分) -> 更新时间戳 -> 入队
     * 加速度与角速度按 x/y/z 顺序转成 Eigen::Vector3d 存入 lio::IMUData
     * 注意: 这里只做缓存与时间检查, 坐标变换、单位换算都由算法侧负责
     * 调用关系: 由订阅回调线程调用, 数据最终被 syncPackage() 取出交给 LIOBuilder
     */
    void imuCB(const sensor_msgs::Imu::ConstPtr msg)
    {
        std::lock_guard<std::mutex> lock(group_data_.imu_mutex);
        double timestamp = msg->header.stamp.toSec();   // ros::Time -> 秒
        if (timestamp < group_data_.last_imu_time)
        {
            ROS_WARN("IMU TIME SYNC ERROR");
            group_data_.imu_buffer.clear();              // 时间倒流: 丢弃已经缓冲的旧数据
        }
        group_data_.last_imu_time = timestamp;
        group_data_.imu_buffer.emplace_back(Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
                                           Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                           timestamp);  // (加速度, 角速度, 时间戳)
    }

    /**
     * 雷达回调: 把一帧 Livox 消息转成 PCL 点云后入队
     *
     * 步骤:
     *   1. 先建好输出点云并调用 livox2pcl() 做线号/回波筛选、抽帧与距离筛选(不做锁内耗时操作)
     *   2. 加锁, 检查时间戳是否倒流, 倒流则清空缓冲
     *   3. 以 (帧时间, 点云) 的形式入队
     * 注意: 时间戳用的是消息 header 的时间(帧起始时刻); 点云内部每个点的相对偏移存在 curvature 字段里
     */
    void lidarCB(const livox_ros_driver2::CustomMsg::ConstPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
        livox2pcl(msg, cloud, config_.filter_num, config_.range_min, config_.range_max);  // 预处理: 筛选 + 抽帧, 并把偏移时间写进 curvature
        std::lock_guard<std::mutex> lock(group_data_.lidar_mutex);
        double timestamp = msg->header.stamp.toSec();
        if (timestamp < group_data_.last_lidar_time)
        {
            ROS_WARN("LIDAR TIME SYNC ERROR");
            group_data_.lidar_buffer.clear();            // 时间倒流: 丢弃已经缓冲的旧数据
        }
        group_data_.last_lidar_time = timestamp;
        group_data_.lidar_buffer.emplace_back(timestamp, cloud);
    }

    /**
     * 把 IMU 缓冲与雷达缓冲拼成一帧同步数据(lio::SyncPackage)
     *
     * 步骤:
     *   1. 任一缓冲为空则直接返回 false, 等下一次定时器再试
     *   2. 若当前还没有取出点云(lidar_pushed 为假), 取 lidar_buffer 队首作为本帧点云,
     *      并用"帧起始时间 + 最后一个点的相对偏移(毫秒换算成秒)"估算帧结束时间;
     *      置 lidar_pushed 为真, 保证在等到足够 IMU 之前不会被重复取帧
     *   3. 若最近一帧 IMU 的时间还没到点云结束时间, 说明 IMU 不够用, 返回 false 继续等
     *   4. 把时间戳早于点云结束时间的 IMU 全部弹出, 放进 sync_pack_.imus(供积分/去畸变使用)
     *   5. 弹出这一帧点云并把 lidar_pushed 复位, 返回 true
     *
     * 返回值: true 表示 sync_pack_ 已装满一帧可用数据
     * 调用关系: mainCB() 每次触发都会调用它; 成功后由 mainCB() 交给 map_builder_.process()
     */
    bool syncPackage()
    {
        if (group_data_.imu_buffer.empty() || group_data_.lidar_buffer.empty())
            return false;
        // 同步点云数据
        if (!group_data_.lidar_pushed)
        {
            sync_pack_.cloud = group_data_.lidar_buffer.front().second;               // 本帧点云(雷达系, 未去畸变)
            sync_pack_.cloud_start_time = group_data_.lidar_buffer.front().first;     // 帧起始时间(秒)
            // 帧结束时间 = 帧起始时间 + 最后一个点的偏移时间(curvature 是毫秒, 除以 1000 换成秒)
            sync_pack_.cloud_end_time = sync_pack_.cloud_start_time + sync_pack_.cloud->points.back().curvature / double(1000.0);
            // NOTE: livox2pcl 可能把整帧点全部滤掉(例如 range_max 配得过小), 此时点云为空,
            //       上面的 points.back() 属于未定义行为, 这里没有做空点云判断
            group_data_.lidar_pushed = true;
        }
        // 等待IMU的数据
        if (group_data_.last_imu_time < sync_pack_.cloud_end_time)
            return false;   // IMU 还没覆盖到点云结束时刻, 下一轮再试
        //  说明inu数据还不够雷达点云时间一帧。
        sync_pack_.imus.clear();

        // 同步IMU的数据
        // IMU的最后一帧数据的时间小于点云最后一个点的时间
        while (!group_data_.imu_buffer.empty() && (group_data_.imu_buffer.front().timestamp < sync_pack_.cloud_end_time))
        {
            sync_pack_.imus.push_back(group_data_.imu_buffer.front());   // 取出的 IMU 全部交给后续处理
            group_data_.imu_buffer.pop_front();
        }
        group_data_.lidar_buffer.pop_front();   // 本帧点云已用完
        group_data_.lidar_pushed = false;       // 复位, 下次可以取新的一帧
        return true;
    }

    /**
     * 主定时器回调(50Hz): 一帧数据的处理与发布
     *
     * 步骤:
     *   1. syncPackage() 攒帧, 没攒好直接返回
     *   2. 交给 map_builder_.process(): 内部按状态机做 IMU 初始化 -> 首帧建图 -> 迭代 EKF 更新与地图增量更新
     *   3. 状态还没进入 LIO_MAPPING(即还在 IMU 初始化/首帧建图中)就不发布, 没有可信位姿
     *      注意状态是在 process() 内部更新的, 因此"完成首帧建图"的那一帧也会正常发布
     *   4. 缓存一份当前状态到 state_, 广播 world -> body 的 TF
     *   5. 把本帧点云分别变换到机体系与世界系发布, 并额外发一份"点云 + 位姿"给后端
     *
     * 注意: 发布用的是 sync_pack_.cloud(全分辨率), 而算法侧更新地图用的是 LIOBuilder 内部
     *       按 scan_resolution 降采样后的点云, 两者点数并不相同
     * 时间戳统一用 sync_pack_.cloud_end_time, 与点云时刻对齐
     */
    void mainCB(const ros::TimerEvent &e)
    {
        if (!syncPackage())
            return;
        map_builder_.process(sync_pack_);
        if (map_builder_.status_ != lio::LIOStatus::LIO_MAPPING)
            return;
        state_ = map_builder_.kf_.x();     // 复制一份当前状态(位置/姿态/速度/零偏/重力), 发布时用
        br_.sendTransform(eigen2Transform(state_.rot, state_.pos, config_.map_frame, config_.body_frame, sync_pack_.cloud_end_time));  // 广播 map -> body 的 TF
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr body_cloud = map_builder_.lidarToBody(sync_pack_.cloud);    // 雷达系 -> 机体系(只用外参)
        publishCloud(body_cloud_pub_, body_cloud, config_.body_frame, sync_pack_.cloud_end_time);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr world_cloud = map_builder_.lidarToWorld(sync_pack_.cloud);  // 雷达系 -> 世界系(用当前位姿)
        publishCloud(world_cloud_pub_, world_cloud, config_.map_frame, sync_pack_.cloud_end_time);

        publishCloudWithOdom(body_cloud, config_.map_frame, config_.body_frame, sync_pack_.cloud_end_time);
    }
    /**
     * 体素平面可视化定时器回调(5s 一次, 只在 publish_voxel_map 为真时启动)
     *
     * 步骤: 取一次地图指针 -> 地图太小时不发(还没建起来) -> 没有订阅者时不发(省掉构造 MarkerArray 的开销)
     *       -> 把体素平面转成 MarkerArray 发布
     * 注意: 帧号用 config_.map_frame; 时间戳用的是当前时间 ros::Time::now()(不是点云时间);
     *       marker 边长取 voxel_size * 0.8, 让相邻体素之间留一点缝, 视觉上不糊成一片
     */
    void voxelTimerCB(const ros::TimerEvent &event)
    {
        std::shared_ptr<lio::VoxelMap> voxel_map = map_builder_.map_;  // 直接引用 LIOBuilder 内部维护的地图
        if (voxel_map->cache_.size() < 10)
            return;                                          // 地图还太小, 没东西可显示
        if (voxel_map_pub_.getNumSubscribers() < 1)
            return;                                          // 没有订阅者, 不做无用的可视化计算
        voxel_map_pub_.publish(voxel2MarkerArray(voxel_map, config_.map_frame, ros::Time::now().toSec(), config_.publish_voxel_num, voxel_map->voxel_size_ * 0.8));
    }

    /**
     * 发布点云的通用小工具
     *
     * pub      : 目标发布者
     * cloud    : 待发布的点云
     * frame_id : 点云所在坐标系名(会写进消息 header)
     * time     : 时间戳(秒)
     * 没有订阅者时直接返回, 连消息构造都省掉
     */
    void publishCloud(ros::Publisher &pub, pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::string &frame_id, const double &time)
    {
        if (pub.getNumSubscribers() < 1)
            return;
        pub.publish(pcl2msg(cloud, frame_id, time));
    }

    /**
     * 发布"点云 + 位姿"组合消息(话题 cloud_with_odom), 供下游 std_matcher 做回环检测
     *
     * _cloud        : 机体系点云(本节点传的是 body_cloud)
     * _frame_id     : 写进外层消息 header 的坐标系名(即位姿所在的世界系)
     * _child_frame  : 写进内层点云 header 的坐标系名(即点云自身的机体系)
     * _timestamp    : 时间戳(秒)
     *
     * 步骤:
     *   1. 把 XYZINormal 点云转成 XYZI(丢掉法向量/协方差字段, 减小消息体积)
     *   2. 检查订阅者, 然后手工拼 sensor_msgs::PointCloud2 并填时间戳与机体系名
     *      (这里没有复用 pcl2msg, 因为内层点云的 frame_id 要用 _child_frame 而不是外层的 _frame_id)
     *   3. 外层 header 用世界系与同一时间戳, 位姿取成员 state_(即本帧位姿)
     *   4. 协方差取 kf_.P() 左上 6x6 块, 按行优先写进 pose.covariance
     *
     * NOTE: 外层 header.frame_id 是 _frame_id(map), 内层点云 header 是 _child_frame(body), 两者不同;
     *       下游 std_matcher/src/pgo_node.cpp 会把 msg->header.frame_id 当成本地坐标系名使用, 阅读时留意
     * 说明: state_ 里的位置误差在前 3 维、旋转误差在后 3 维(见 IESKF 的 P_ID/R_ID), 与 ROS
     *       PoseWithCovariance 的 x,y,z,rx,ry,rz 排列一致; 但旋转部分实际是 SO3 切空间上的误差协方差
     * NOTE: 订阅者检查放在点云拷贝之后, 没有订阅者时那次拷贝是白做的(只是浪费一点开销)
     */
    void publishCloudWithOdom(pcl::PointCloud<pcl::PointXYZINormal>::Ptr _cloud, std::string &_frame_id, std::string &_child_frame, double _timestamp)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr pub_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*_cloud, *pub_cloud);   // XYZINormal -> XYZI

        if (pointcloud_with_odom_pub_.getNumSubscribers() < 1)
            return;
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*pub_cloud, cloud_msg);
        if (_timestamp < 0)
            cloud_msg.header.stamp = ros::Time().now();             // 负时间表示用当前时刻
        else
            cloud_msg.header.stamp = ros::Time().fromSec(_timestamp);
        cloud_msg.header.frame_id = _child_frame;                   // 点云自身的坐标系(机体系)
        
        interface::PointCloudWithOdom msg;
        msg.header.stamp = ros::Time().fromSec(_timestamp);         // 外层时间戳: 与点云、位姿同源
        msg.header.frame_id = _frame_id;                            // 外层坐标系: 位姿所在的世界系
        msg.cloud = cloud_msg;
        msg.pose.pose.position.x = state_.pos.x();                   // 世界系下的位置
        msg.pose.pose.position.y = state_.pos.y();
        msg.pose.pose.position.z = state_.pos.z();
        Eigen::Quaterniond q(state_.rot);                            // 旋转矩阵 -> 四元数
        msg.pose.pose.orientation.w = q.w();
        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(msg.pose.covariance.data()) = map_builder_.kf_.P().block<6, 6>(0, 0);  // 按行优先展开成 36 个元素
        pointcloud_with_odom_pub_.publish(msg);
    }

public:
    ros::NodeHandle nh_;                     // 私有句柄(nh_("~")): 参数与话题都在 /lio_node 命名空间下
    tf2_ros::TransformBroadcaster br_;       // TF 广播器, 发 map -> body 变换
    NodeConfig config_;                      // 节点参数(见 loadConfig)
    NodeGroupData group_data_;               // 雷达/IMU 缓冲与最新时间戳
    lio::SyncPackage sync_pack_;             // 当前正在组装/处理的同步帧
    ros::Timer main_loop_;                   // 主定时器对象, 50Hz
    ros::Timer voxel_map_loop_;              // 可视化定时器对象, 5s 一次

    std::string map_frame_;                  // NOTE: 这两个成员没有被赋值也没有被读取, 实际用的是 config_.map_frame / config_.body_frame
    std::string body_frame_;

    ros::Subscriber lidar_sub_;              // 雷达订阅者
    ros::Subscriber imu_sub_;                // IMU 订阅者

    ros::Publisher odom_pub_;                // 里程计发布者(NOTE: 见 initPublishers 里的说明, 从未发布)
    ros::Publisher body_cloud_pub_;          // 机体系点云发布者
    ros::Publisher world_cloud_pub_;         // 世界系点云发布者
    ros::Publisher voxel_map_pub_;           // 体素平面 MarkerArray 发布者
    ros::Publisher pointcloud_with_odom_pub_; // 点云+位姿发布者(给后端 std_matcher)

    lio::LIOBuilder map_builder_;            // 算法主体: IMU 初始化、去畸变、迭代 EKF、体素地图维护

    lio::LIOConfig lio_config_;              // 从参数服务器读到的算法配置, 构造时交给 map_builder_

    kf::State state_;                        // 最近一次可用的状态副本, 供发布 TF 与位姿使用
};

/**
 * 程序入口
 *
 * ros::init 用 "lio_node" 作为节点名, 因此私有参数与私有话题都在 /lio_node 命名空间下;
 * 节点对象构造时完成全部初始化(参数、订阅、发布、定时器), 之后交给单线程 ros::spin()
 * 注意: 单线程 spin 意味着订阅回调与定时器回调是串行执行的
 */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "lio_node");
    LIONode lio_node;
    ros::spin();
    return 0;
}