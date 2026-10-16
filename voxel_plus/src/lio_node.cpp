#include <ros/ros.h>
#include <queue>
#include <algorithm>
#include "utils.h"
#include "map_builder/commons.h"
#include "map_builder/lio_builder.h"
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TwistStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include "interface/PointCloudWithOdom.h"

struct NodeConfig
{
    std::string lidar_type = "livox"; // livox: livox_ros_driver2::CustomMsg, pointcloud2: sensor_msgs::PointCloud2
    std::string lidar_topic;
    std::string imu_topic;
    std::string map_frame;
    std::string body_frame;
    std::string velocity_topic = ""; // 可选: 机体系速度提示(如 pose_to_imu 发布的 xxx/velocity)
    double range_min = 0.5;
    double range_max = 20.0;
    int filter_num = 3;
    double cloud_time_offset = 0.0; // 无逐点时间时, 帧参考时刻相对 header.stamp 的偏移(秒)
    bool time_from_azimuth = false; // 旋转式雷达: 用方位角推算逐点时间做去畸变
    double scan_period = 0.1;
    double azimuth_dir = -1.0;
    bool publish_voxel_map = false;
    int publish_voxel_num = 1000;
};

struct LidarFrame
{
    double start_time = 0.0;
    double end_time = 0.0;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud;
};

struct NodeGroupData
{
    double last_imu_time = 0.0;
    double last_lidar_time = 0.0;
    std::mutex imu_mutex;
    std::mutex lidar_mutex;
    std::deque<lio::IMUData> imu_buffer;
    std::deque<LidarFrame> lidar_buffer;
    bool lidar_pushed = false;
    bool drop_warned = false;
};

class LIONode
{
public:
    LIONode() : nh("~")
    {
        loadConfig();
        initSubScribers();
        initPublishers();
        map_builder.loadConfig(lio_config);
        main_loop = nh.createTimer(ros::Duration(0.02), &LIONode::mainCB, this);
        voxel_map_loop = nh.createTimer(ros::Duration(5.0), &LIONode::voxelTimerCB, this, false, false);
        if (config.publish_voxel_map)
            voxel_map_loop.start();
    }

    void loadConfig()
    {
        nh.param<std::string>("lidar_type", config.lidar_type, "livox");
        nh.param<std::string>("lidar_topic", config.lidar_topic, "/livox/lidar");
        nh.param<std::string>("imu_topic", config.imu_topic, "/livox/imu");
        nh.param<std::string>("velocity_topic", config.velocity_topic, "");

        nh.param<std::string>("body_frame", config.body_frame, "body");
        nh.param<std::string>("map_frame", config.map_frame, "map");

        nh.param<int>("filter_num", config.filter_num, 3);
        nh.param<double>("range_min", config.range_min, 0.5);
        nh.param<double>("range_max", config.range_max, 20.0);
        nh.param<double>("cloud_time_offset", config.cloud_time_offset, 0.0);
        nh.param<bool>("point_time_from_azimuth", config.time_from_azimuth, false);
        nh.param<double>("scan_period", config.scan_period, 0.1);
        nh.param<double>("azimuth_dir", config.azimuth_dir, -1.0);
        nh.param<bool>("publish_voxel_map", config.publish_voxel_map, false);
        nh.param<int>("publish_voxel_num", config.publish_voxel_num, 1000);

        nh.param<double>("scan_resolution", lio_config.scan_resolution, 0.1);
        nh.param<double>("voxel_size", lio_config.voxel_size, 0.4);

        nh.param<double>("max_residual", lio_config.max_residual, 0.5);

        nh.param<double>("merge_thresh_for_angle", lio_config.merge_thresh_for_angle, 0.1);
        nh.param<double>("merge_thresh_for_distance", lio_config.merge_thresh_for_distance, 0.04);

        nh.param<int>("max_point_thresh", lio_config.max_point_thresh, 100);
        nh.param<int>("update_size_thresh", lio_config.update_size_thresh, 10);
        if (!nh.hasParam("update_size_thresh") && nh.hasParam("update_size_threshes"))
        {
            ROS_WARN("update_size_threshes is deprecated, use update_size_thresh instead");
            nh.getParam("update_size_threshes", lio_config.update_size_thresh);
        }

        nh.param<int>("map_capacity", lio_config.map_capacity, 100000);
        nh.param<bool>("gravity_align", lio_config.gravity_align, true);
        nh.param<int>("imu_init_num", lio_config.imu_init_num, 20);
        nh.param<double>("na", lio_config.na, 0.01);
        nh.param<double>("ng", lio_config.ng, 0.01);
        nh.param<double>("nbg", lio_config.nbg, 0.0001);
        nh.param<double>("nba", lio_config.nba, 0.0001);
        nh.param<int>("opti_max_iter", lio_config.opti_max_iter, 5);
        nh.param<int>("init_frames", lio_config.init_frames, 10);
        nh.param<double>("init_pos_cov", lio_config.init_pos_cov, 2.5e-3);
        nh.param<double>("init_rot_cov", lio_config.init_rot_cov, 1e-4);
        nh.param<double>("init_vel_cov", lio_config.init_vel_cov, 1e-4);
        nh.param<double>("min_pos_cov", lio_config.min_pos_cov, 1e-4);
        nh.param<double>("min_rot_cov", lio_config.min_rot_cov, 1e-6);
        nh.param<double>("min_vel_cov", lio_config.min_vel_cov, 2.5e-3);
        nh.param<double>("min_bias_cov", lio_config.min_bias_cov, 1e-8);
        std::vector<double> r_il, p_il;
        nh.param<std::vector<double>>("r_il", r_il, std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
        assert(r_il.size() == 9);
        lio_config.r_il << r_il[0], r_il[1], r_il[2], r_il[3], r_il[4], r_il[5], r_il[6], r_il[7], r_il[8];

        nh.param<std::vector<double>>("p_il", p_il, std::vector<double>{0, 0, 0});
        assert(p_il.size() == 3);
        lio_config.p_il << p_il[0], p_il[1], p_il[2];
    }

    void initSubScribers()
    {
        if (config.lidar_type == "pointcloud2")
            lidar_sub = nh.subscribe(config.lidar_topic, 10000, &LIONode::cloudCB, this);
        else if (config.lidar_type == "livox")
            lidar_sub = nh.subscribe(config.lidar_topic, 10000, &LIONode::livoxCB, this);
        else
        {
            ROS_ERROR("unknown lidar_type: %s, should be 'livox' or 'pointcloud2'", config.lidar_type.c_str());
            ros::shutdown();
            return;
        }
        imu_sub = nh.subscribe(config.imu_topic, 10000, &LIONode::imuCB, this);
        if (!config.velocity_topic.empty())
            velocity_sub = nh.subscribe(config.velocity_topic, 100, &LIONode::velocityCB, this);
        ROS_INFO("subscribe lidar: %s (%s), imu: %s", config.lidar_topic.c_str(), config.lidar_type.c_str(), config.imu_topic.c_str());
    }

    void initPublishers()
    {
        odom_pub = nh.advertise<nav_msgs::Odometry>("slam_odom", 1000);
        body_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("body_cloud", 1000);
        world_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("world_cloud", 1000);
        voxel_map_pub = nh.advertise<visualization_msgs::MarkerArray>("voxel_map", 1000);
        pointcloud_with_odom_pub = nh.advertise<interface::PointCloudWithOdom>("cloud_with_odom", 1000);
    }

    void velocityCB(const geometry_msgs::TwistStamped::ConstPtr msg)
    {
        map_builder.setVelocityHint(Eigen::Vector3d(msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z));
    }

    void imuCB(const sensor_msgs::Imu::ConstPtr msg)
    {
        std::lock_guard<std::mutex> lock(group_data.imu_mutex);
        double timestamp = msg->header.stamp.toSec();
        if (timestamp < group_data.last_imu_time)
        {
            ROS_WARN("IMU TIME SYNC ERROR");
            group_data.imu_buffer.clear();
        }
        group_data.last_imu_time = timestamp;
        group_data.imu_buffer.emplace_back(Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
                                           Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                           timestamp);
    }

    void livoxCB(const livox_ros_driver2::CustomMsg::ConstPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
        livox2pcl(msg, cloud, config.filter_num, config.range_min, config.range_max);
        double timestamp = msg->header.stamp.toSec();
        double min_offset = 0.0, max_offset = 0.0;
        for (const auto &point : cloud->points)
        {
            min_offset = std::min(min_offset, point.curvature / double(1000.0));
            max_offset = std::max(max_offset, point.curvature / double(1000.0));
        }
        pushCloud(timestamp + min_offset, timestamp + max_offset, cloud);
    }

    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
        CloudConvertConfig convert_config;
        convert_config.range_min = config.range_min;
        convert_config.range_max = config.range_max;
        convert_config.filter_num = config.filter_num;
        convert_config.snapshot_offset = config.cloud_time_offset;
        convert_config.time_from_azimuth = config.time_from_azimuth;
        convert_config.scan_period = config.scan_period;
        convert_config.azimuth_dir = config.azimuth_dir;
        std::pair<double, double> cloud_time = msg2pcl(msg, cloud, convert_config);
        pushCloud(cloud_time.first, cloud_time.second, cloud);
    }

    void pushCloud(const double &start_time, const double &end_time, pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud)
    {
        std::lock_guard<std::mutex> lock(group_data.lidar_mutex);
        if (start_time < group_data.last_lidar_time)
        {
            ROS_WARN("LIDAR TIME SYNC ERROR");
            group_data.lidar_buffer.clear();
        }
        group_data.last_lidar_time = start_time;
        LidarFrame frame;
        frame.start_time = start_time;
        frame.end_time = end_time;
        frame.cloud = cloud;
        group_data.lidar_buffer.push_back(frame);
    }

    bool syncPackage()
    {
        if (group_data.imu_buffer.empty() || group_data.lidar_buffer.empty())
            return false;
        // 同步点云数据
        if (!group_data.lidar_pushed)
        {
            sync_pack.cloud = group_data.lidar_buffer.front().cloud;
            sync_pack.cloud_start_time = group_data.lidar_buffer.front().start_time;
            sync_pack.cloud_end_time = group_data.lidar_buffer.front().end_time;
            group_data.lidar_pushed = true;
        }
        // 该帧还没有任何IMU数据覆盖(常见于bag开头), 直接丢弃, 否则下游会拿到空的IMU数据
        if (group_data.imu_buffer.front().timestamp > sync_pack.cloud_end_time)
        {
            if (!group_data.drop_warned)
            {
                ROS_WARN("drop cloud without imu coverage: cloud [%.3f, %.3f], oldest imu %.3f",
                         sync_pack.cloud_start_time, sync_pack.cloud_end_time, group_data.imu_buffer.front().timestamp);
                group_data.drop_warned = true;
            }
            group_data.lidar_buffer.pop_front();
            group_data.lidar_pushed = false;
            return false;
        }
        // 时间基准跳变(同一话题上还有别的 bag 在发/换了数据源/长时间断流): 清空缓冲并重建估计器,
        // 否则 cloud_time_end - imu_time_end 会变成巨大的 dt, 直接把状态推到发散
        const double last_end_time = map_builder.data_group.last_cloud_end_time;
        if (last_end_time > 0.0)
        {
            const double time_gap = sync_pack.cloud_end_time - last_end_time;
            if (time_gap < -0.1 || time_gap > 5.0)
            {
                ROS_ERROR("time base jump detected: %.3fs (last cloud end %.3f, now %.3f), reset estimator",
                          time_gap, last_end_time, sync_pack.cloud_end_time);
                group_data.imu_buffer.clear();
                group_data.last_imu_time = 0.0; // 等待新时间基准下的IMU数据
                group_data.lidar_buffer.clear();
                group_data.lidar_pushed = false;
                group_data.last_lidar_time = 0.0;
                map_builder.reset();
                return false;
            }
        }
        // 等待IMU的数据
        if (group_data.last_imu_time < sync_pack.cloud_end_time)
            return false;

        sync_pack.imus.clear();

        // 同步IMU的数据
        // IMU的最后一帧数据的时间小于点云最后一个点的时间
        while (!group_data.imu_buffer.empty() && (group_data.imu_buffer.front().timestamp < sync_pack.cloud_end_time))
        {
            sync_pack.imus.push_back(group_data.imu_buffer.front());
            group_data.imu_buffer.pop_front();
        }
        group_data.lidar_buffer.pop_front();
        group_data.lidar_pushed = false;
        return true;
    }

    void mainCB(const ros::TimerEvent &e)
    {
        if (!syncPackage())
            return;
        map_builder.process(sync_pack);
        if (map_builder.status != lio::LIOStatus::LIO_MAPPING)
            return;
        state = map_builder.kf.x();
        ROS_INFO_THROTTLE(1.0, "pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f) effective_points=%d",
                          state.pos.x(), state.pos.y(), state.pos.z(),
                          state.vel.x(), state.vel.y(), state.vel.z(), map_builder.data_group.effect_num);
        br.sendTransform(eigen2Transform(state.rot, state.pos, config.map_frame, config.body_frame, sync_pack.cloud_end_time));
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr body_cloud = map_builder.lidarToBody(sync_pack.cloud);
        publishCloud(body_cloud_pub, body_cloud, config.body_frame, sync_pack.cloud_end_time);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr world_cloud = map_builder.lidarToWorld(sync_pack.cloud);
        publishCloud(world_cloud_pub, world_cloud, config.map_frame, sync_pack.cloud_end_time);

        publishCloudWithOdom(body_cloud, config.map_frame, config.body_frame, sync_pack.cloud_end_time);
    }
    void voxelTimerCB(const ros::TimerEvent &event)
    {
        std::shared_ptr<lio::VoxelMap> voxel_map = map_builder.map;
        if (voxel_map->cache.size() < 10)
            return;
        if (voxel_map_pub.getNumSubscribers() < 1)
            return;
        voxel_map_pub.publish(voxel2MarkerArray(voxel_map, config.map_frame, ros::Time::now().toSec(), config.publish_voxel_num, voxel_map->voxel_size * 0.8));
    }

    void publishCloud(ros::Publisher &pub, pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::string &frame_id, const double &time)
    {
        if (pub.getNumSubscribers() < 1)
            return;
        pub.publish(pcl2msg(cloud, frame_id, time));
    }

    void publishCloudWithOdom(pcl::PointCloud<pcl::PointXYZINormal>::Ptr _cloud, std::string &_frame_id, std::string &_child_frame, double _timestamp)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr pub_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*_cloud, *pub_cloud);

        if (pointcloud_with_odom_pub.getNumSubscribers() < 1)
            return;
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*pub_cloud, cloud_msg);
        if (_timestamp < 0)
            cloud_msg.header.stamp = ros::Time().now();
        else
            cloud_msg.header.stamp = ros::Time().fromSec(_timestamp);
        cloud_msg.header.frame_id = _child_frame;
        
        interface::PointCloudWithOdom msg;
        msg.header.stamp = ros::Time().fromSec(_timestamp);
        msg.header.frame_id = _frame_id;
        msg.cloud = cloud_msg;
        msg.pose.pose.position.x = state.pos.x();
        msg.pose.pose.position.y = state.pos.y();
        msg.pose.pose.position.z = state.pos.z();
        Eigen::Quaterniond q(state.rot);
        msg.pose.pose.orientation.w = q.w();
        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(msg.pose.covariance.data()) = map_builder.kf.P().block<6, 6>(0, 0);
        pointcloud_with_odom_pub.publish(msg);
    }

public:
    ros::NodeHandle nh;
    tf2_ros::TransformBroadcaster br;
    NodeConfig config;
    NodeGroupData group_data;
    lio::SyncPackage sync_pack;
    ros::Timer main_loop;
    ros::Timer voxel_map_loop;

    std::string map_frame;
    std::string body_frame;
    std::string velocity_topic = ""; // 可选: 机体系速度提示(如 pose_to_imu 发布的 xxx/velocity)

    ros::Subscriber lidar_sub;
    ros::Subscriber imu_sub;
    ros::Subscriber velocity_sub;

    ros::Publisher odom_pub;
    ros::Publisher body_cloud_pub;
    ros::Publisher world_cloud_pub;
    ros::Publisher voxel_map_pub;
    ros::Publisher pointcloud_with_odom_pub;

    lio::LIOBuilder map_builder;

    lio::LIOConfig lio_config;

    kf::State state;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lio_node");
    LIONode lio_node;
    ros::spin();
    return 0;
}