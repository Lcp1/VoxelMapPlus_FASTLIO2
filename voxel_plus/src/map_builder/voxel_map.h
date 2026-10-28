#pragma once
#include <list>
#include <vector>
#include <memory>
#include <cstdint>
#include <Eigen/Eigen>
#include <unordered_map>
#include <chrono>
#include <iostream>

// 哈希质数: VoxelKey::Hasher 多项式滚动哈希的乘子
#define HASH_P 116101
// 哈希取模上限: 防止 (z * HASH_P + y) * HASH_P 在 int64 上溢出
#define MAX_N 10000000000

namespace lio
{

    /**
     * @brief 体素的整数索引, 同时作为哈希表的 key
     *
     * 世界坐标点 p 对应的体素索引为 floor(p / voxel_size_), 见 VoxelMap::index()
     */
    class VoxelKey
    {
    public:
        int64_t x, y, z;
        VoxelKey(int64_t _x = 0, int64_t _y = 0, int64_t _z = 0) : x(_x), y(_y), z(_z) {}

        // 判等: 三个分量全部相同才算同一个体素
        bool operator==(const VoxelKey &other) const
        {
            return (x == other.x && y == other.y && z == other.z);
        }

        // 哈希函数: 多项式滚动 ((z * P) % MAX_N + y) * P % MAX_N + x
        struct Hasher
        {
            int64_t operator()(const VoxelKey &k) const
            {
                return ((((k.z) * HASH_P) % MAX_N + (k.y)) * HASH_P) % MAX_N + (k.x);
            }
        };
    };

    // 输入点: 坐标 + 3x3 协方差
    // 协方差由雷达测量噪声和当前状态协方差合成后传入, 见 LIOBuilder 中的 pv.cov = point_cov
    struct PointWithCov
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d point;
        Eigen::Matrix3d cov;
    };

    // 体素内拟合出的平面, 同时兼作在线累加器(边收点边更新统计量)
    struct Plane
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d mean = Eigen::Vector3d::Zero(); // 点云质心, 在线增量均值
        Eigen::Matrix3d ppt = Eigen::Matrix3d::Zero();  // 二阶矩 Σ(p * p^T), 用于求样本协方差
        Eigen::Vector3d norm = Eigen::Vector3d::Zero(); // 单位法向量, 朝向已统一
        // 6x6 参数不确定度: block<3,3>(0,0) 是法向量的协方差
        //                   block<3,3>(3,3) 是质心的协方差
        Eigen::Matrix<double, 6, 6> cov;
        int n = 0; // 已累加的点数
    };

    // 一次点-面残差的完整描述, 由 VoxelMap::buildResidual() 填充后交给 IESKF 构造观测方程
    struct ResidualData
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d point_lidar;              // 点在雷达系下的坐标
        Eigen::Vector3d point_world;              // 点在当前估计位姿下的世界坐标
        Eigen::Vector3d plane_mean;               // 命中平面的质心(回填)
        Eigen::Vector3d plane_norm;               // 命中平面的法向量(回填)
        Eigen::Matrix<double, 6, 6> plane_cov;    // 命中平面的参数协方差
        Eigen::Matrix3d cov_lidar;                // 点自身在雷达系下的协方差
        Eigen::Matrix3d cov_world;                // 点自身传播到世界系下的协方差
        bool is_valid = false;                    // 该残差是否通过卡方检验
        double residual = 0.0;                    // 点到平面的有符号距离
    };

    class VoxelMap;

    /**
     * @brief 单个体素: 缓存点 + 拟合平面 + 与邻域共面合并
     *
     * 生命周期由 pushPoint() 的状态机驱动:
     *   未初始化 -> 攒点直到能拟合出平面 -> 平面态(增量更新) -> 冻结 -> 与邻居 merge()
     */
    class VoxelGrid
    {
    public:
        VoxelGrid(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, VoxelKey _position, VoxelMap *_map);

        // 由当前累积的点云拟合平面, 并把点的不确定度传播成 6x6 平面协方差
        void updatePlane();

        // 只把点累加进平面统计量(在线均值/二阶矩/点数), 不入 temp_points_
        void addToPlane(const PointWithCov &pv);

        // 累加统计量 + 存入 temp_points_, 不做任何阈值判断(供 build() 批量建图)
        void addPoint(const PointWithCov &pv);

        // 在线增量插入(供 update() 使用), 内部维护整个体素的状态机
        void pushPoint(const PointWithCov &pv);

        // 与 6 邻域中已冻结的共面体素合并, 统一平面参数并打上相同的 group_id_
        void merge();

    public:
        static uint64_t count_;      // 全局自增 ID 计数器, 用于分配 group_id_
        int max_point_thresh_;       // temp_points_ 的点数上限, 达到后冻结本体素
        int update_point_thresh_;    // 每新增多少个点重算一次平面
        double plane_thresh_;        // 平面判定阈值: 最小特征值大于它则认为不是平面
        bool is_init_;               // 是否已经初始化过(至少完整跑通一次 updatePlane)
        bool is_plane_;              // 当前是否被判定为平面
        bool update_enable_;         // 是否还在接收新点, false 表示已冻结
        int newly_add_point_;        // 距上次重算平面以来新增的点数
        bool merged_;                // 本帧是否发生过合并
        uint64_t group_id_;          // 平面合并后的分组 ID, 用于避免重复合并
        std::vector<PointWithCov> temp_points_; // 保留的原始点(算协方差时需要)
        VoxelKey position_;          // 本体素的整数索引
        VoxelMap *map_;              // 所属地图, merge() 时用于查邻居
        std::shared_ptr<Plane> plane_; // 拟合出的平面
        Eigen::Vector3d center_;     // 平面中心(等价于 plane_->mean)
        std::list<VoxelKey>::iterator cache_it_; // 指向 VoxelMap::cache_ 中自己的节点, 用于 O(1) LRU 移动
        static double merge_thresh_for_angle_;    // 合并的夹角阈值(1 - cos)
        static double merge_thresh_for_distance_; // 合并的平面间距阈值
    };

    // 体素哈希表: 由体素索引映射到体素对象
    typedef std::unordered_map<VoxelKey, std::shared_ptr<VoxelGrid>, VoxelKey::Hasher> Featmap;

    /**
     * @brief 体素平面地图: 哈希表 + LRU 容量管理 + 对外接口
     */
    class VoxelMap
    {
    public:
        VoxelMap(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, double _voxel_size, int _capacity = 2000000);

        // 世界坐标 -> 体素索引: floor(point / voxel_size_)
        VoxelKey index(const Eigen::Vector3d &point);

        // 初始化建图(首帧): 只累积点, 最后对全表统一拟合一次平面
        void build(std::vector<PointWithCov> &pvs);

        // 在线增量更新(后续每帧): 逐点 pushPoint, 由体素自己决定是否重算/合并
        void update(std::vector<PointWithCov> &pvs);

        // 构建点-面残差: 返回残差是否有效(通过 3 sigma 卡方检验)
        bool buildResidual(ResidualData &data, std::shared_ptr<VoxelGrid> voxel_grid);

    public:
        int max_point_thresh_;      // 下发给每个 VoxelGrid 的点数上限
        int update_point_thresh_;   // 下发给每个 VoxelGrid 的重算间隔
        double plane_thresh_;       // 下发给每个 VoxelGrid 的平面判定阈值
        double voxel_size_;         // 体素边长
        Featmap featmap_;           // 体素哈希表
        std::list<VoxelKey> cache_; // LRU 链表: 头部最近使用, 尾部最先淘汰
        int capacity_;              // 体素数量上限, 超出后按 LRU 淘汰
    };

} // namespace lio
