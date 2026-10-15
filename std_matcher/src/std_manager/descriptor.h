#pragma once
#include <vector>
#include <cstdint>
#include <Eigen/Eigen>
#include <unordered_set>
#include <unordered_map>
#include <Eigen/Eigen> // NOTE: Eigen/Eigen 被重复包含(无功能影响)
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sophus/so3.hpp>

// 哈希质数: VoxelKey::Hasher 多项式滚动哈希的乘子
#define HASH_P 116101
// 哈希取模上限: 防止 (z * HASH_P + y) * HASH_P 在 int64 上溢出
#define MAX_N 10000000000
namespace std_desc
{
    // 体素降采样(voxelFilter)的中间累加器: 按体素累加各分量, 最后除以点数得均值
    struct MPoint
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector4f xyzi; // 4 个分量的累加和: (Σx, Σy, Σz, Σintensity)
        int count;            // 落入该体素的点数
    };
    /**
     * @brief 体素降采样: 输入点云按 voxel_size 分体素后取组内均值(实现见 descriptor.cpp)
     * @param in         输入/输出点云(原地降采样)
     * @param voxel_size 体素边长(米)
     * @note 由 pgo_node 用于单帧点云抽稀
     */
    void voxelFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr in, double voxel_size);

    /**
     * @brief 反对称矩阵(叉乘矩阵): skew(v) * w == v × w
     * @param vec 三维向量
     * @return 3x3 反对称矩阵(对角线为 0)
     * @note 本工程内未被调用(平面 ICP 中用的是 Sophus::SO3d::hat)
     */
    Eigen::Matrix3d skew(const Eigen::Vector3d &vec);

    /**
     * @brief 体素的整数索引, 作为哈希表的 key, 在本工程中复用为三种用途
     *
     *   1) 空间体素索引: index(x, y, z, voxel_size)          -- 当前帧体素表 temp_voxels
     *   2) 边长哈希桶键:  index(s1, s2, s3, side_resolution)  -- 描述子数据库 data_base 的插入/检索
     *   3) 边长去重键:    index(s1, s2, s3, 0.001)            -- buildDescriptor 内部去重
     */
    class VoxelKey
    {
    public:
        int64_t x, y, z; // 体素整数坐标(或三个边长量化后的桶坐标)

        /**
         * @brief 构造函数: 三分量初始化
         * @param _x, _y, _z 体素索引三个分量, 缺省全 0
         */
        VoxelKey(int64_t _x = 0, int64_t _y = 0, int64_t _z = 0) : x(_x), y(_y), z(_z) {}

        /**
         * @brief 判等算子: 三个分量全部相同才算同一个体素
         * @param other 待比较的体素键
         * @return 三分量均相等返回 true
         */
        bool operator==(const VoxelKey &other) const
        {
            return (x == other.x && y == other.y && z == other.z);
        }

        /**
         * @brief 浮点坐标 -> 体素索引: floor(坐标 / resolution + bias)
         * @param x, y, z    输入坐标(空间点坐标, 或三角形的三条边长)
         * @param resolution 体素边长(空间体素用 voxel_size, 边长桶用 side_resolution)
         * @param bias       floor 之前的分量偏置, 默认 0.0
         * @return 各分量 floor 后转 int64 的体素索引
         */
        static VoxelKey index(double x, double y, double z, double resolution, double bias = 0.0);

        /**
         * @brief 生成 center 的 (2*range+1)^3 邻域体素索引(立方体邻域)
         * @param center       中心体素
         * @param range        邻域半径(体素个数, 断言要求 > 0)
         * @param exclude_self 是否剔除中心自身
         * @return 邻域索引列表, 遍历顺序为 x 外层 / y 中层 / z 内层
         */
        static std::vector<VoxelKey> rangeNear(VoxelKey center, int range, bool exclude_self);

        // 哈希器: 与 voxel_plus::VoxelKey::Hasher 相同的多项式滚动哈希
        struct Hasher
        {
            /**
             * @brief 哈希算子: ((z * P) % MAX_N + y) * P % MAX_N + x
             * @param k 体素键
             * @return 64 位哈希值(取模上限 MAX_N 用于防止乘法溢出)
             */
            int64_t operator()(const VoxelKey &k) const
            {
                return ((((k.z) * HASH_P) % MAX_N + (k.y)) * HASH_P) % MAX_N + (k.x);
            }
        };
    };

    /**
     * @brief STD 描述子提取与匹配的全部参数, 由 pgo_node 从 ROS 参数服务器加载(config.yaml)
     *
     * 按使用阶段分组: 平面体素分割 / 角点投影与 2D-NMS / 角点 NMS 与三角形约束 / 候选检索与验证
     */
    struct Config
    {
        // ---- 平面体素分割(buildVoxels / buildConnections) ----
        int voxel_min_point = 10;          // 体素点数不超过该值时不再判定平面(is_valid 置 false)
        double voxel_size = 1.0;           // 体素边长(米), 用于把点云划分到规则体素
        double voxel_plane_thresh = 0.01;  // 平面判定阈值: 样本协方差最小特征值小于该值则认为是平面
        double norm_merge_thresh = 0.1;    // 法向一致性阈值: |n1 - n2| 或 |n1 + n2| 小于该值认为两法向平行(共面)

        // ---- 角点投影与 2D-NMS(nms2d) ----
        double proj_2d_resolution = 0.25;  // 平面投影的 2D 栅格分辨率(米)
        double proj_min_dis = 0.0;         // 投影保留区间下限: 点到平面距离 |proj_z| 小于该值则丢弃(默认 0 表示不设下限)
        double proj_max_dis = 5.0;         // 投影保留区间上限: |proj_z| 大于该值则丢弃

        // ---- 角点提取与 NMS ----
        int nms_2d_range = 5;              // 2D-NMS 滑动窗口边长(栅格数), 同时是窗口步长
        double nms_3d_range = 2.0;         // 角点 3D-NMS 的半径搜索范围(米)
        double corner_thresh = 10.0;       // 角点判定阈值: 窗口内峰值栅格的点数达到该值才输出角点
        int max_corner_num = 100;          // 每帧最多保留的角点数(按 intensity 降序截断)
        double min_side_len = 2.0;         // 三角形边长下限(米), 任一边小于它则不生成描述子
        double max_side_len = 30.0;        // 三角形边长上限(米), 任一边大于它则不生成描述子
        int desc_search_range = 15;        // 构建描述子时每个角点取的最近邻个数(KNN 的 K, 含查询点自身)

        // ---- 候选检索(selectCanidates) ----
        double side_resolution = 0.2;       // 边长哈希桶分辨率(米): 边长 / side_resolution 量化后入桶
        double rough_dis_threshold = 0.03;  // 边长初筛的相对阈值: 三边长向量(缩放后)距离 < |该向量| * 该值
        int skip_near_num = 50;             // 时间近邻跳过帧数: 帧号差 <= 该值的历史帧不作为候选
        double vertex_diff_threshold = 0.7; // 顶点强度(attached)相对差阈值: 超过则丢弃该候选
        int candidate_num = 50;             // 单次检索最多输出的候选帧数(按票数从高到低)

        // ---- 验证(triangleSolver / verifyCandidate / verifyGeoPlane) ----
        double verify_dis_thresh = 3.0;     // 三角形验证: 顶点变换后与其对应目标顶点的距离阈值(米)
        double geo_verify_dis_thresh = 0.3; // 平面几何验证: 点到平面距离阈值(米)
        double icp_thresh = 0.5;            // 回环判定阈值: 几何验证得分(匹配比例)大于该值才认为回环有效

        // ---- 平面 ICP(verifyGeoPlaneICP) ----
        double iter_eps = 1e-3;             // 迭代终止阈值: 增量最大分量小于该值则提前结束
        size_t max_iter = 10;               // 最大迭代次数
    };

    /**
     * @brief 体素节点: 原始点云 + 一阶/二阶统计量 + 平面参数 + 6 邻域连接关系
     *
     * 每帧重建, 生命周期:
     *   buildVoxels()      -> 填 sum/ppt 并拟合平面(mean/norms/lamdas/norm/is_plane)
     *   buildConnections() -> 填 connect / connect_check / connect_nodes
     *   extractCorners()   -> 读取以上信息, 把非平面体素的点投影到相邻平面提取角点
     */
    struct VoxelNode
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d sum;    // 体素内点的坐标和 Σp
        Eigen::Vector3d mean;   // 质心 Σp / n, 平面特征时代表平面上的一个点
        Eigen::Matrix3d ppt;    // 二阶矩 Σ(p * p^T), 与 sum 一起求样本协方差
        Eigen::Vector3d norm;   // 平面单位法向量(最小特征值对应的特征向量), 仅平面体素有意义
        Eigen::Matrix3d norms;  // 三个特征向量按列存放, 列序为 (最小特征值, 中间, 最大)
        Eigen::Vector3d lamdas; // 三个特征值, 分量序 (x = 最小, y = 中间, z = 最大)

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud; // 落在该体素内的原始点云
        bool is_plane = false;      // 是否判定为平面(最小特征值 < voxel_plane_thresh)
        // NOTE: is_valid 只在点数不足时被置 false, 全流程没有任何地方把它置 true, 也没有被读取
        bool is_valid = false;      // 点数是否足够(buildVoxels 中的标记)
        bool is_projected = false;  // 本帧该体素是否已被投影过(避免同一批点重复贡献角点)
        std::vector<Eigen::Vector3d> projected_norms; // 已投影过的平面法向(法向相近的平面不再重复投影)
        bool connect[6] = {false, false, false, false, false, false}; // 6 邻域连接标志, 方向序 +x,+y,+z,-x,-y,-z;
                                                                      // 平面体素: 与邻居共面为 true; 非平面体素: 该方向邻居是平面才为 true
        bool connect_check[6] = {false, false, false, false, false, false}; // 该方向是否已检查过(防止双向重复处理)
        VoxelNode *connect_nodes[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}; // 对应方向的邻居节点指针
    };

    /**
     * @brief STD(Stable Triangle Descriptor) 三角形描述子
     *
     * 由 buildDescriptor() 从三个角点构造。核心是旋转/平移不变的几何量:
     *   side_length 三边长(升序), 与坐标系无关, 用作数据库检索键
     *   angle       三个顶点法向之间的夹角特征, 同样与坐标系无关
     * 顶点按“对边长度升序”的规则重排: A 对最短边 s1, B 对中间边 s2, C 对最长边 s3,
     * 因此两帧中同一个三角形的顶点对应关系是确定的(镜像歧义在 triangleSolver 中处理)
     */
    struct STDDescriptor
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d side_length; // 三边长(米), 升序 s1 <= s2 <= s3; 也是 data_base 哈希桶的量化依据
        Eigen::Vector3d angle;       // 顶点法向夹角特征: (5*|n_a·n_b|, 5*|n_b·n_c|, 5*|n_c·n_a|), 见 buildDescriptor
        Eigen::Vector3d center;      // 三角形重心 (A + B + C) / 3, triangleSolver 中作为平移对齐基准
        Eigen::Vector3d vertex_a;    // 顶点 A(A 的对边 s1 最短)
        Eigen::Vector3d vertex_b;    // 顶点 B(B 的对边 s2 居中)
        Eigen::Vector3d vertex_c;    // 顶点 C(C 的对边 s3 最长)
        Eigen::Vector3d attached;    // 三个顶点的 intensity 三元组(角点强度), 候选检索时用于顶点属性比较
        Eigen::Matrix3d norms;       // 三列分别为 A/B/C 三个顶点的单位法向量(取自角点来源的平面)
        uint64_t id;                 // 描述子所属帧号, 由 insert() 赋值为 frame_count
    };
    /**
     * @brief 一帧的全部 STD 特征: 三角形描述子集合 + 该帧的平面点云
     *
     * extract() 生产, searchLoop() / insert() 消费
     */
    struct STDFeature
    {
        // NOTE: extract() 未对 stamp 赋值, 当前工程中也没有地方读取它
        double stamp; // 时间戳(预留字段)
        uint64_t id;  // 帧号, extract() 中置为 frame_count(不自增), insert() 中再次确认
        // NOTE: extract() 未对 valid 赋值, 当前工程中也没有地方读取它
        bool valid; // 特征是否有效(预留字段)
        std::vector<STDDescriptor> descs; // 该帧所有三角形描述子
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud; // 平面点云(每个平面体素一个点: 质心 + 法向), 用于几何验证
    };
    /**
     * @brief 一个候选匹配帧: 该历史帧上所有与当前帧匹配上的描述子对
     */
    struct STDMatch
    {
        uint64_t match_id; // 候选历史帧号(即 data_base 中描述子的 id)
        std::vector<std::pair<STDDescriptor, STDDescriptor>> match_pairs; // 描述子对: first 当前帧, second 历史帧
    };
    /**
     * @brief 回环检索结果, 由 searchLoop() 返回
     */
    struct LoopResult
    {
        bool valid = false;           // 是否判定为有效回环(得分 > config.icp_thresh)
        uint64_t match_id;            // 匹配上的历史帧号(searchLoop 中初始化为 -1, 即 uint64 最大值, 作“无匹配”标记)
        double match_score;           // 几何验证得分(平面点匹配比例, 0 ~ 1)
        Eigen::Matrix3d rotation;     // 当前帧 -> 历史帧 的旋转(由最佳三角形对解出)
        Eigen::Vector3d translation;  // 对应的平移, 满足 p_历史帧 ≈ R * p_当前帧 + t
        std::vector<std::pair<STDDescriptor, STDDescriptor>> match_pairs; // 通过一致性检查的描述子对
    };

    /**
     * @brief STD 描述子管理器: 单帧特征提取 -> 回环检索 -> 历史入库 的完整流水线
     *
     * 典型调用顺序(pgo_node 对每个关键帧):
     *   extract() -> searchLoop() -> insert()
     * searchLoop 必须在 insert 之前调用, 保证当前帧不会检索到自身;
     * 时间上过近的历史帧由 config.skip_near_num 过滤
     */
    class STDManager
    {
    public:
        /**
         * @brief 构造函数: 仅保存参数副本
         * @param _config 提取/匹配参数(按值拷贝到成员 config)
         */
        STDManager(Config &_config) : config(_config) {}

        /**
         * @brief 单帧特征提取主入口: 体素化 -> 平面连接 -> 角点提取 -> 三角形描述子 -> 平面点云
         * @param cloud 当前关键帧点云(雷达系)
         * @return STDFeature: descs 为三角形描述子, cloud 为平面点云, id 置为 frame_count(不自增)
         * @note 内部调用 buildVoxels / buildConnections / extractCorners / buildDescriptor / currentPlaneCloud
         */
        STDFeature extract(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

        /**
         * @brief 把点云按 voxel_size 划分体素, 累加 sum/ppt 统计量并逐体素作特征分解判定平面
         * @param cloud 输入点云; 结果写入成员 temp_voxels
         */
        void buildVoxels(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

        /**
         * @brief 提取平面角点: 非平面体素的点投影到相邻平面做 2D-NMS, 再 3D-NMS, 最后按强度截断
         * @return 角点云(intensity = 角点强度, normal 字段 = 该角点来源平面的法向)
         * @note 依赖 buildVoxels 的平面判定与 buildConnections 的连接关系; 输出供 buildDescriptor 使用
         */
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr extractCorners();

        /**
         * @brief 在单个平面 (mean, norm) 上对投影点做 2D 非极大值抑制, 得到该平面上的角点
         * @param mean        平面质心(投影基准点)
         * @param norm        平面单位法向量
         * @param proj_points 待投影的 3D 点(来自投影平面邻域的非平面体素)
         * @return 角点云, 每个角点的 intensity 为窗口峰值栅格的点数, curvature 置 1.0(有效标记)
         */
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr nms2d(const Eigen::Vector3d &mean, const Eigen::Vector3d &norm, std::vector<Eigen::Vector3d> &proj_points);

        /**
         * @brief 角点 3D 非极大值抑制: 半径 nms_3d_range 内只保留 intensity 最大者
         * @param prepare_key_cloud 输入/输出角点云; curvature 被复用为保留标记(1.0 保留 / 0.0 抑制)
         */
        void nms3d(pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud);

        /**
         * @brief 由角点云构建 STD 三角形描述子: KNN 组三角形 -> 边长筛选 -> 顶点规范化 -> 去重 -> 填充描述子
         * @param cloud 角点云(extractCorners 的输出)
         * @param desc  输出描述子数组(追加式写入)
         */
        void buildDescriptor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::vector<STDDescriptor> &desc);

        /**
         * @brief 建立体素 6 邻域连接: 平面-平面共面连接, 以及非平面体素到相邻平面的从属连接
         * @note 供 extractCorners 为角点候选选择投影平面
         */
        void buildConnections();

        /**
         * @brief 汇总当前帧所有平面体素为一个平面点云(位置 = 质心, 法向 = 平面法向)
         * @return 平面点云, 供几何验证使用并随 STDFeature 保存到 cloud_vec
         */
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr currentPlaneCloud();

        /**
         * @brief 把一帧特征写入历史数据库: 平面点云入 cloud_vec, 描述子按边长桶入 data_base, 帧号自增
         * @param feature 待入库特征(其 id 与内部描述子的 id 会被改写为当前 frame_count)
         */
        void insert(STDFeature &feature);

        /**
         * @brief 由一对匹配三角形(顶点已按对边升序一一对应)用 SVD 求二者之间的旋转平移
         * @param pair  匹配对: first 为源(当前帧)描述子, second 为目标(历史帧)描述子
         * @param rot   输出旋转矩阵 R
         * @param trans 输出平移向量 t, 满足 p_target ≈ R * p_source + t
         */
        void triangleSolver(std::pair<STDDescriptor, STDDescriptor> &pair, Eigen::Matrix3d &rot, Eigen::Vector3d &trans);

        /**
         * @brief 在 data_base 中检索当前帧每个描述子的候选匹配(边长桶查找 + 初筛 + 按帧投票排序)
         * @param feature    当前帧特征(使用其 descs 与 id)
         * @param match_list 输出候选匹配列表, 按票数从高到低, 最多 config.candidate_num 个
         */
        void selectCanidates(STDFeature &feature, std::vector<STDMatch> &match_list);

        /**
         * @brief 候选验证: 抽样生成位姿假设并投票, 取最佳假设后用平面几何验证打分
         * @param matches           候选匹配(含 match_id 与全部描述子对)
         * @param rot, trans        输出位姿(由最佳三角形对解出)
         * @param success_match_vec 输出在最终位姿下顶点一致的匹配对
         * @param plane_cloud       当前帧平面点云(源)
         * @return 几何验证得分(0 ~ 1); 投票不足(小于 4)时返回 -1.0
         */
        double verifyCandidate(STDMatch &matches,
                               Eigen::Matrix3d &rot,
                               Eigen::Vector3d &trans,
                               std::vector<std::pair<STDDescriptor, STDDescriptor>> &success_match_vec,
                               pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud);

        /**
         * @brief 平面点云几何验证: 统计源平面点变换后能找到“法向平行且点到平面距离达标”的对应点的比例
         * @param source_cloud 源平面点云(当前帧)
         * @param target_cloud 目标平面点云(历史帧)
         * @param rot, trans   待评估的变换
         * @return 匹配比例(0 ~ 1), 作为回环得分
         */
        double verifyGeoPlane(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                              pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                              const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans);

        /**
         * @brief 平面点云 ICP: 最近邻建对应 + 高斯-牛顿迭代优化位姿, 返回平均点到平面残差
         * @param source_cloud 源平面点云(当前帧)
         * @param target_cloud 目标平面点云(历史帧)
         * @param rot, trans   输入初值, 同时被就地更新为优化结果
         * @return 平均绝对点到平面残差; 无对应点时的返回值见函数内 NOTE
         * @note 本文件内未调用, 由 pgo_node 在检出回环后调用
         */
        double verifyGeoPlaneICP(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                                 Eigen::Matrix3d &rot, Eigen::Vector3d &trans);

        /**
         * @brief 回环检索入口: 选候选 -> 逐个验证打分 -> 保留最高分
         * @param feature 当前帧特征(需在 insert 之前调用)
         * @return LoopResult, 其中 match_score > config.icp_thresh 时 valid 置 true
         */
        LoopResult searchLoop(STDFeature &feature);

    public:
        Config config;                // 参数副本(构造时从入参拷贝)
        static uint64_t frame_count;  // 全局帧计数器: insert() 每收录一帧自增一次, 同时用作描述子 id / cloud_vec 下标
        std::unordered_map<VoxelKey, VoxelNode, VoxelKey::Hasher> temp_voxels; // 当前帧的体素表, extract() 开始时清空重建
        // 历史描述子数据库: 边长桶(VoxelKey) -> 该桶内所有描述子; 随帧数持续增长, 不做清理
        std::unordered_map<VoxelKey, std::vector<STDDescriptor>, VoxelKey::Hasher> data_base;
        std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> cloud_vec; // 每帧平面点云, 下标 == 帧号(由 insert() 顺序 push 保证)
    };
}