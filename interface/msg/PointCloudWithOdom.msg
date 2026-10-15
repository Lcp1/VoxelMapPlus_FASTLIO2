# 前端 lio_node 发给后端 pgo_node 的数据包:
# 一帧点云 + 该帧对应的位姿(含协方差)
# 发布者: voxel_plus 的 lio_node(话题 cloud_with_odom)
# 订阅者: std_matcher 的 pgo_node(参数 odom_cloud_topic)

std_msgs/Header header              # 时间戳与坐标系
sensor_msgs/PointCloud2 cloud       # 该帧点云(机体系)
geometry_msgs/PoseWithCovariance pose  # 该帧里程计位姿及其协方差
