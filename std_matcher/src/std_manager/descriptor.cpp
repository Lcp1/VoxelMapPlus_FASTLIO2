#include "descriptor.h"

namespace std_desc
{
    /**
     * @brief 体素降采样: 按 voxel_size 把输入点云划分到体素, 每个体素用组内均值(含 intensity)代替
     *
     * 步骤:
     *   1. 遍历输入点, 按体素索引累加 (x, y, z, intensity) 与点数到 MPoint
     *   2. 清空输入点云
     *   3. 每个体素输出一个均值点
     *
     * @param in         输入/输出点云(原地降采样)
     * @param voxel_size 体素边长(米)
     * @note 由 pgo_node 用于单帧点云抽稀
     */
    void voxelFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr in, double voxel_size)
    {
        std::unordered_map<VoxelKey, MPoint, VoxelKey::Hasher> voxels; // 体素索引 -> 分量累加器
        for (auto p : in->points) // NOTE: 按值拷贝每个点(用 const auto & 更省), 仅性能问题
        {
            VoxelKey k = VoxelKey::index(p.x, p.y, p.z, voxel_size); // 该点所属体素
            if (voxels.find(k) == voxels.end()) // 首次遇到该体素: 初始化累加器
            {
                voxels[k].xyzi.setZero();
                voxels[k].count = 0;
            }
            voxels[k].xyzi(0) += p.x;         // 累加 x
            voxels[k].xyzi(1) += p.y;         // 累加 y
            voxels[k].xyzi(2) += p.z;         // 累加 z
            voxels[k].xyzi(3) += p.intensity; // 累加 intensity
            voxels[k].count += 1;             // 计数
        }
        in->clear(); // 清空原云, 准备写回均值点
        for (auto it = voxels.begin(); it != voxels.end(); it++)
        {
            pcl::PointXYZI p;
            p.x = it->second.xyzi(0) / static_cast<double>(it->second.count); // 均值代替体素内所有点
            p.y = it->second.xyzi(1) / static_cast<double>(it->second.count);
            p.z = it->second.xyzi(2) / static_cast<double>(it->second.count);
            p.intensity = it->second.xyzi(3) / static_cast<double>(it->second.count);
            in->push_back(p);
        }
    }

    /**
     * @brief 求向量 vec 的反对称矩阵(叉乘矩阵) [vec]x, 满足 [vec]x * w == vec × w
     * @param vec 三维向量
     * @return 3x3 反对称矩阵(对角线为 0)
     * @note 本工程内未被调用(平面 ICP 中用的是 Sophus::SO3d::hat)
     */
    Eigen::Matrix3d skew(const Eigen::Vector3d &vec)
    {
        Eigen::Matrix3d ret;
        ret << 0.0, -vec(2), vec(1), // 第 1 行
            vec(2), 0.0, -vec(0),    // 第 2 行
            -vec(1), vec(0), 0.0;    // 第 3 行
        return ret;
    }

    /**
     * @brief 浮点坐标 -> 体素索引: floor(坐标 / resolution + bias)
     * @param x, y, z    输入坐标(空间点坐标, 或三角形的三条边长)
     * @param resolution 体素边长(空间体素用 voxel_size, 边长桶用 side_resolution)
     * @param bias       floor 之前的分量偏置, 默认 0.0
     * @return 各分量 floor 后转 int64 的体素索引
     * @note 先加 bias 再 floor, 保证负坐标也落在正确的体素
     */
    VoxelKey VoxelKey::index(double x, double y, double z, double resolution, double bias)
    {
        Eigen::Vector3d point(x, y, z);
        Eigen::Vector3d idx = (point / resolution + Eigen::Vector3d(bias, bias, bias)).array().floor(); // 逐分量除后 floor
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2))); // 转 int64 索引
    }

    /**
     * @brief 生成 center 的 (2*range+1)^3 立方体邻域体素索引
     * @param center  中心体素
     * @param range   邻域半径(体素个数), 断言要求 > 0
     * @param exclude 是否剔除中心自身
     * @return 邻域索引列表; 遍历顺序为 i(x) 外层 -> j(y) -> k(z) 内层
     * @note 被 extractCorners() 用来搜集 3x3x3 邻域中的非平面体素点
     */
    std::vector<VoxelKey> VoxelKey::rangeNear(VoxelKey center, int range, bool exclude)
    {
        assert(range > 0); // 半径必须为正
        int len = range * 2 + 1; // 每个方向的体素个数
        int size = exclude ? len * len * len - 1 : len * len * len; // 预分配的元素个数
        std::vector<VoxelKey> keys;
        keys.reserve(size);
        for (int i = -range; i <= range; i++)
            for (int j = -range; j <= range; j++)
                for (int k = -range; k <= range; k++)
                {
                    if (i == 0 && j == 0 && k == 0 && exclude) // 剔除中心自身
                        continue;
                    keys.emplace_back(center.x + i, center.y + j, center.z + k);
                }
        return keys;
    }

    // 全局帧计数器: insert() 每收录一帧自增一次; 同时作为描述子 id 与 cloud_vec 的下标(见 insert 的说明)
    uint64_t STDManager::frame_count = 0;

    /**
     * @brief 把当前帧点云划分到 voxel_size 体素, 累加一阶/二阶统计量, 并逐体素作特征分解判定平面
     *
     * 步骤:
     *   1. 逐点求体素索引; 新体素先初始化(点云指针/标志/统计量清零)
     *   2. 点入体素: cloud 存原始点, sum 累加 Σp, ppt 累加 Σ(p * p^T)
     *   3. 遍历体素: 点数 <= voxel_min_point 的置 is_valid = false 并跳过
     *   4. 求质心 mean 与样本协方差 cov = Σ(p * p^T) / n - mean * mean^T
     *   5. 特征分解(cov 为对称阵, 特征值必为实数, 这里取实部)
     *      求最小/最大特征值下标, 中间下标由 0 + 1 + 2 = 3 的恒等式推得
     *   6. 最小特征值 < voxel_plane_thresh 判为平面:
     *        特征值与特征向量按 (小, 中, 大) 重排到 lamdas / norms,
     *        norm = 最小特征值对应的特征向量(平面法向)
     *      否则 is_plane = false
     *
     * @param cloud 当前帧点云
     * @note 结果写入成员 temp_voxels, 供 buildConnections / extractCorners 使用
     */
    void STDManager::buildVoxels(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
    {
        for (auto &p : cloud->points)
        {
            VoxelKey k = VoxelKey::index(p.x, p.y, p.z, config.voxel_size); // 点所属体素
            if (temp_voxels.find(k) == temp_voxels.end()) // 新体素: 初始化各项状态
            {
                temp_voxels[k].cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
                temp_voxels[k].is_plane = false;
                temp_voxels[k].is_valid = false;
                temp_voxels[k].lamdas.setZero();
                temp_voxels[k].norms.setZero();
                temp_voxels[k].sum.setZero();
                temp_voxels[k].ppt.setZero();
            }
            temp_voxels[k].cloud->push_back(p); // 保留原始点
            Eigen::Vector3d point_vec(p.x, p.y, p.z);
            temp_voxels[k].sum += point_vec; // 一阶矩: Σp
            temp_voxels[k].ppt += point_vec * point_vec.transpose(); // 二阶矩: Σ(p * p^T)
        }

        for (auto it = temp_voxels.begin(); it != temp_voxels.end(); it++)
        {
            if (it->second.cloud->size() <= config.voxel_min_point) // 点数不足: 体素无效, 不参与平面判定
            {
                it->second.is_valid = false;
                continue;
            }
            Eigen::Vector3d mean = it->second.sum / static_cast<double>(it->second.cloud->size()); // 质心
            Eigen::Matrix3d cov = it->second.ppt / static_cast<double>(it->second.cloud->size()) - mean * mean.transpose(); // 样本协方差
            // NOTE: cov 是对称阵, 这里用一般特征分解再取实部(改用 SelfAdjointEigenSolver 会更快且保证实数)
            Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
            Eigen::Matrix3d norms = es.eigenvectors().real(); // 实部即真实特征向量(对称阵无虚部)
            Eigen::Vector3d lamdas = es.eigenvalues().real(); // 特征值
            Eigen::Matrix3d::Index evalsMin, evalsMax;
            lamdas.minCoeff(&evalsMin); // 最小特征值下标(对应法向方向)
            lamdas.maxCoeff(&evalsMax); // 最大特征值下标(点云展布最大的方向)
            int evalsMid = 3 - evalsMin - evalsMax; // 剩下的一维就是中间特征值(0 + 1 + 2 = 3)
            it->second.mean = mean;
            if (lamdas(evalsMin) < config.voxel_plane_thresh) // 最小特征值足够小 => 点云近似分布在一个平面上
            {
                it->second.is_plane = true;
                it->second.lamdas.x() = lamdas(evalsMin); // 重排: x = 最小特征值
                it->second.lamdas.y() = lamdas(evalsMid); // y = 中间
                it->second.lamdas.z() = lamdas(evalsMax); // z = 最大
                it->second.norms.col(0) = norms.col(evalsMin); // 第 0 列: 法向
                it->second.norms.col(1) = norms.col(evalsMid); // 第 1 列: 中间方向
                it->second.norms.col(2) = norms.col(evalsMax); // 第 2 列: 最大展布方向
                it->second.norm = it->second.norms.col(0); // 平面法向量
                continue;
            }
            it->second.is_plane = false; // 三个方向都有明显展布, 不是平面
        }
    }

    /**
     * @brief 建立体素 6 邻域连接关系(平面-平面共面连接, 以及非平面体素到相邻平面的从属连接)
     *
     * 只遍历平面体素(循环开头跳过非平面), i = 0..5 对应 +x, +y, +z, -x, -y, -z 六个方向:
     *   1. 邻域体素不存在: 标记已检查, 连接置 false(边界)
     *   2. 该方向已检查过(被对面的节点处理过): 跳过, 避免重复处理
     *   3. 反向索引 j = i ± 3 与 i 配对(0<->3, 1<->4, 2<->5), 双方都标记已检查
     *   4. 邻居是平面:
     *        |n1 - n2| < norm_merge_thresh(法向同向) 或 |n1 + n2| < norm_merge_thresh(法向反向)
     *        则双向 connect = true 并互存节点指针; 否则双向 connect = false
     *      邻居不是平面:
     *        本平面节点 connect[i] = false, 而邻居(角点候选)节点 connect[j] = true
     *        并记录指向本平面节点的指针, 即非平面体素用 connect 数组“记住”哪个方向的邻居是平面
     *
     * @note 连接关系由 extractCorners() 消费: 非平面体素把点投影到有连接的相邻平面上提取角点
     */
    void STDManager::buildConnections()
    {
        for (auto it = temp_voxels.begin(); it != temp_voxels.end(); it++)
        {
            // 如果不是平面，则跳过
            if (!it->second.is_plane)
                continue;

            VoxelNode &c_node = it->second; // 当前(平面)节点
            for (int i = 0; i < 6; i++)
            {
                VoxelKey nk = it->first; // 复制中心键, 按方向 i 偏移出邻居键
                if (i == 0)
                    nk.x = nk.x + 1; // +x
                else if (i == 1)
                    nk.y = nk.y + 1; // +y
                else if (i == 2)
                    nk.z = nk.z + 1; // +z
                else if (i == 3)
                    nk.x = nk.x - 1; // -x
                else if (i == 4)
                    nk.y = nk.y - 1; // -y
                else if (i == 5)
                    nk.z = nk.z - 1; // -z
                auto near = temp_voxels.find(nk);
                if (near == temp_voxels.end())
                {
                    c_node.connect_check[i] = true; // 邻域体素不存在(边界): 已检查且不连接
                    c_node.connect[i] = false;
                }
                else
                {
                    // 如果在其他节点被访问过，则跳过
                    if (c_node.connect_check[i])
                        continue;

                    VoxelNode &n_node = near->second; // 邻居节点
                    c_node.connect_check[i] = true;
                    int j = i >= 3 ? i - 3 : i + 3; // 反方向下标
                    n_node.connect_check[j] = true;
                    // 如果邻接节点是平面
                    if (n_node.is_plane)
                    {

                        Eigen::Vector3d normal_dif = c_node.norm - n_node.norm; // 法向差: 同向时范数接近 0
                        Eigen::Vector3d normal_add = c_node.norm + n_node.norm; // 法向和: 反向时范数接近 0
                        if (normal_add.norm() < config.norm_merge_thresh || normal_dif.norm() < config.norm_merge_thresh)
                        {
                            c_node.connect[i] = true; // 双向共面连接
                            n_node.connect[j] = true;
                            c_node.connect_nodes[i] = &n_node;
                            n_node.connect_nodes[j] = &c_node;
                        }
                        else
                        {
                            c_node.connect[i] = false; // 法向不一致: 双方都不连接
                            n_node.connect[j] = false;
                        }
                    }
                    else
                    {
                        c_node.connect[i] = false; // 邻居是角点候选体素: 平面侧不连接
                        n_node.connect[j] = true; // 但邻居侧记录“该方向邻居是平面”, 供投影提角点用
                        n_node.connect_nodes[j] = &c_node;
                    }
                }
            }
        }
    }

    /**
     * @brief 在平面 (mean, norm) 上对投影点做 2D 非极大值抑制, 提取该平面的角点
     *
     * 步骤:
     *   1. 构造平面内正交基: x_axis 先取 (1, 1, 0), 再修正为与 norm 垂直并归一化;
     *      y_axis = norm × x_axis, (x_axis, y_axis, norm) 构成右手标准正交坐标系
     *   2. 逐点投影: p2c = p - mean,
     *      proj_x = p2c · y_axis, proj_y = p2c · x_axis, proj_z = p2c · norm(点到平面的有符号距离);
     *      只保留 |proj_z| 落在 [proj_min_dis, proj_max_dis] 区间内的点
     *   3. 投影点太少(<= 5)直接返回空
     *   4. 求投影点的 2D 包围盒, 按 proj_2d_resolution 划栅格, 每格累计点数与坐标和
     *   5. 以 nms_2d_range x nms_2d_range 个栅格为滑动窗口逐窗扫描, 取窗内点数最大的栅格;
     *      峰值 >= corner_thresh 才记录为候选
     *   6. 候选栅格输出为角点: 坐标取栅格内点的均值反投影回 3D
     *      coord = mean + py * x_axis + px * y_axis(丢弃 proj_z 分量, 即投影到平面上);
     *      intensity 存峰值点数, curvature 置 1.0(有效标记, 见 nms3d),
     *      normal 字段存平面法向 norm(记录该角点来自哪个平面)
     *
     * @param mean        平面质心(投影基准点)
     * @param norm        平面单位法向量
     * @param proj_points 待投影的 3D 点(来自该平面邻域的非平面体素)
     * @return 角点云(intensity = 峰值点数, curvature = 1.0, normal = 平面法向)
     */
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr STDManager::nms2d(const Eigen::Vector3d &mean, const Eigen::Vector3d &norm, std::vector<Eigen::Vector3d> &proj_points)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr ret(new pcl::PointCloud<pcl::PointXYZINormal>);
        Eigen::Vector3d x_axis(1, 1, 0); // 平面内基向量的初始猜测

        double A = norm(0), B = norm(1), C = norm(2); // 平面方程系数: A*x + B*y + C*z = 0(过 mean)
        if (C != 0)
            x_axis[2] = -(A + B) / C; // 取 (1, 1, z) 形式, 令 z 分量满足垂直条件
        else if (B != 0)
            x_axis[1] = -A / B; // C = 0 时改调 y 分量使 (1, y, 0) 垂直
        else
        {
            x_axis[0] = 0; // B, C 均为 0(法向只剩 x 分量), (0, 1, 0) 必然垂直
            x_axis[1] = 1;
        }
        x_axis.normalize(); // 单位化后 x_axis 与 norm 正交
        Eigen::Vector3d y_axis = norm.cross(x_axis); // 平面内第二基向量(norm 与 x_axis 正交且单位, 故 y_axis 也是单位向量)
        std::vector<Eigen::Vector2d> point_list_2d; // 通过距离筛选的平面内 2D 点 (proj_x, proj_y)
        for (Eigen::Vector3d point : proj_points)
        {
            Eigen::Vector3d p2c = point - mean; // 相对质心的向量
            double proj_x = p2c.dot(y_axis), proj_y = p2c.dot(x_axis), proj_z = p2c.dot(norm); // 2D 坐标 + 到平面的有符号距离
            if (std::abs(proj_z) < config.proj_min_dis || std::abs(proj_z) > config.proj_max_dis)
                continue; // 点到平面距离落在保留区间之外: 丢弃
            point_list_2d.push_back(Eigen::Vector2d(proj_x, proj_y));
        }
        if (point_list_2d.size() <= 5)
            return ret; // 点数太少, 无法形成统计峰值
        // NOTE: 包围盒初值硬编码为 ±10 米, 投影点坐标超出该范围时统计会出错(按当前 3x3x3 邻域的点源一般不会超出)
        double min_x = 10, max_x = -10, min_y = 10, max_y = -10;
        for (auto pi : point_list_2d)
        {
            if (pi[0] < min_x)
                min_x = pi[0];
            if (pi[0] > max_x)
                max_x = pi[0];
            if (pi[1] < min_y)
                min_y = pi[1];
            if (pi[1] > max_y)
                max_y = pi[1];
        }

        double segmen_len = config.nms_2d_range * config.proj_2d_resolution; // 滑动窗口的物理边长 = 窗宽栅格数 * 栅格分辨率
        int x_segment_num = (max_x - min_x) / segmen_len + 1; // 窗口个数(除法为 double, 隐式截断成 int)
        int y_segment_num = (max_y - min_y) / segmen_len + 1;
        int x_axis_len = (int)((max_x - min_x) / config.proj_2d_resolution + config.nms_2d_range); // 栅格数组长度(含窗口余量)
        int y_axis_len = (int)((max_y - min_y) / config.proj_2d_resolution + config.nms_2d_range);

        // NOTE: 以下三个变长数组(VLA)是 GCC 扩展而非标准 C++, 长度由运行时数据决定且分配在栈上
        double img_count_array[x_axis_len][y_axis_len]; // 每格点数
        double mean_x_array[x_axis_len][y_axis_len]; // 每格 proj_x 坐标和
        double mean_y_array[x_axis_len][y_axis_len]; // 每格 proj_y 坐标和

        for (int x = 0; x < x_axis_len; x++)
        {
            for (int y = 0; y < y_axis_len; y++)
            {
                img_count_array[x][y] = 0; // 初始化: 清零点数与坐标和
                mean_x_array[x][y] = 0;
                mean_y_array[x][y] = 0;
            }
        }

        for (size_t i = 0; i < point_list_2d.size(); i++)
        {

            int x_index = (int)((point_list_2d[i][0] - min_x) / config.proj_2d_resolution); // 落入的栅格(以 min 为原点, 截断取整)
            int y_index = (int)((point_list_2d[i][1] - min_y) / config.proj_2d_resolution);
            mean_x_array[x_index][y_index] += point_list_2d[i][0]; // 累加坐标和, 输出角点时再取均值
            mean_y_array[x_index][y_index] += point_list_2d[i][1];
            img_count_array[x_index][y_index]++; // 该格点数加一
        }
        std::vector<int> max_gradient_vec; // 每个窗口的峰值点数
        std::vector<int> max_gradient_x_index_vec; // 峰值所在栅格的 x 下标
        std::vector<int> max_gradient_y_index_vec; // 峰值所在栅格的 y 下标

        for (int x_segment_index = 0; x_segment_index < x_segment_num; x_segment_index++)
        {
            for (int y_segment_index = 0; y_segment_index < y_segment_num; y_segment_index++)
            {
                // 在 nms_2d_range x nms_2d_range 的窗口内找点数最多的栅格
                double max_gradient = 0;
                int max_gradient_x_index = -10;
                int max_gradient_y_index = -10;
                for (int x_index = x_segment_index * config.nms_2d_range; x_index < (x_segment_index + 1) * config.nms_2d_range; x_index++)
                {
                    for (int y_index = y_segment_index * config.nms_2d_range; y_index < (y_segment_index + 1) * config.nms_2d_range; y_index++)
                    {
                        if (img_count_array[x_index][y_index] > max_gradient)
                        {
                            max_gradient = img_count_array[x_index][y_index];
                            max_gradient_x_index = x_index;
                            max_gradient_y_index = y_index;
                        }
                    }
                }
                if (max_gradient >= config.corner_thresh) // 峰值点数达到角点阈值: 记录该窗口的候选
                {
                    max_gradient_vec.push_back(max_gradient);
                    max_gradient_x_index_vec.push_back(max_gradient_x_index);
                    max_gradient_y_index_vec.push_back(max_gradient_y_index);
                }
            }
        }

        for (int i = 0; i < max_gradient_vec.size(); i++)
        {
            // 峰值栅格内点的坐标均值(亚栅格精度), 再反投影成 3D 坐标
            double px = mean_x_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]] /
                        img_count_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]];
            double py = mean_y_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]] /
                        img_count_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]];
            Eigen::Vector3d coord = py * x_axis + px * y_axis + mean; // x_axis 方向系数为 py, y_axis 方向系数为 px, 基准点为 mean
            pcl::PointXYZINormal pi;
            pi.x = coord[0];
            pi.y = coord[1];
            pi.z = coord[2];
            pi.intensity = max_gradient_vec[i]; // intensity 存峰值点数(角点强度)
            pi.curvature = 1.0; // curvature 复用为有效标记: 1.0 保留 / 0.0 被 nms3d 压制
            pi.normal_x = norm[0]; // normal 字段记录该角点来源平面的法向
            pi.normal_y = norm[1];
            pi.normal_z = norm[2];
            ret->points.push_back(pi);
        }
        return ret;
    }

    /**
     * @brief 提取平面角点: 非平面体素的点云投影到相邻平面做 2D-NMS 得候选角点,
     *        全体再做 3D-NMS 去重, 最后按强度截断到 max_corner_num
     *
     * 步骤:
     *   1. 遍历非平面体素(角点候选), 从 connect/connect_nodes 找到相邻平面作为投影平面
     *      (要求该平面节点自身有有效连接, 且本体素点数足够)
     *   2. 取本体素 3x3x3 邻域内所有“非平面”体素的点作为待投影点集;
     *      邻域体素若已为相近法向的平面投影过(is_projected / projected_norms),
     *      则跳过该体素, 避免同一批点重复贡献角点
     *   3. 对点集调用 nms2d() 得该平面上的角点, 头插进总角点集合
     *   4. 全体角点 nms3d(): 半径 nms_3d_range 内按 intensity 互相抑制
     *   5. 若角点数超过 max_corner_num, 按 intensity 降序取前 max_corner_num 个
     *
     * @return 角点云(intensity = 角点强度, 法向量 = 该角点来源平面)
     * @note 输入依赖 buildVoxels() 的平面判定与 buildConnections() 的连接关系;
     *       输出供 buildDescriptor() 构建三角形描述子
     */
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr STDManager::extractCorners()
    {
        // 以下为被注释掉的旧方案: 先把体素 key 排序再按序遍历, 用于保证遍历顺序与哈希表实现无关
        // std::vector<VoxelKey> keys;
        // for (auto iter = temp_voxels.begin(); iter != temp_voxels.end(); iter++)
        // {
        //     keys.push_back(iter->first);
        // }
        // std::sort(keys.begin(), keys.end(), [](VoxelKey &k1, VoxelKey &k2)
        //           {
        //     std::string ks1 =  std::to_string(k1.x) + std::to_string(k1.y)+std::to_string(k1.z);
        //     std::string ks2 =  std::to_string(k2.x) + std::to_string(k2.y)+std::to_string(k2.z);
        //     return ks1 > ks2; });
        // TODO: 遍历的一致性约束

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_corner_points(new pcl::PointCloud<pcl::PointXYZINormal>); // 角点总集合

        for (auto iter = temp_voxels.begin(); iter != temp_voxels.end(); iter++)
        // for (VoxelKey &ck_ : keys)
        {
            // auto iter = temp_voxels.find(ck_);
            if (iter->second.is_plane)
                continue; // 只处理非平面体素(角点候选)

            VoxelKey ck = iter->first; // 当前体素索引
            VoxelNode *c_node = &iter->second; // 当前体素节点
            int connect_index = -1; // 找到的“指向平面邻居”的方向下标
            for (int i = 0; i < 6; i++)
            {
                if (!c_node->connect[i])
                    continue; // 非平面体素中 connect 为 true 表示该方向邻居是平面
                connect_index = i;
                VoxelNode *n_node = c_node->connect_nodes[connect_index]; // 相邻平面节点
                bool valid_plane = false; // 该平面节点自身是否有有效连接(否则认为平面不可靠)
                for (int j = 0; j < 6; j++)
                    if (n_node->connect_check[j])
                        if (n_node->connect[j])
                            valid_plane = true;

                if (!valid_plane)
                    continue;
                // NOTE: 这里的 10 是硬编码(数值上与 config.voxel_min_point 默认值相同, 但两者并无关联)
                if (c_node->cloud->size() <= 10)
                    continue;
                Eigen::Vector3d projection_normal = n_node->norm; // 投影平面法向
                Eigen::Vector3d projection_center = n_node->mean; // 投影平面质心

                std::vector<Eigen::Vector3d> proj_points; // 待投影到该平面的 3D 点集

                for (VoxelKey nk : VoxelKey::rangeNear(ck, 1, false)) // 3x3x3 邻域(exclude_self = false, 含自身)
                {
                    auto iter_near = temp_voxels.find(nk);
                    if (iter_near == temp_voxels.end())
                        continue; // 邻域体素不存在
                    if (iter_near->second.is_plane)
                        continue; // 平面体素的点不参与角点投影

                    bool skip = false; // 该体素是否已为相近法向的平面投影过
                    if (iter_near->second.is_projected)
                    {
                        for (auto pnorm : iter_near->second.projected_norms)
                        {
                            Eigen::Vector3d normal_dif = projection_normal - pnorm; // 法向差
                            Eigen::Vector3d normal_add = projection_normal + pnorm; // 法向和
                            if (normal_dif.norm() < 0.5 || normal_add.norm() < 0.5) // 同向或反向相近(阈值硬编码 0.5)
                                skip = true; // 已为相似平面投影过: 跳过, 避免重复贡献
                        }
                    }
                    if (skip)
                        continue;
                    iter_near->second.is_projected = true; // 标记已投影
                    iter_near->second.projected_norms.push_back(projection_normal); // 记录投影过的法向
                    for (auto point : iter_near->second.cloud->points)
                    {
                        proj_points.emplace_back(point.x, point.y, point.z); // 收集该邻域体素的点
                    }
                }

                if (proj_points.size() == 0)
                    continue;
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr corners = nms2d(projection_center, projection_normal, proj_points); // 2D-NMS 提角点

                prepare_corner_points->points.insert(prepare_corner_points->points.begin(), corners->points.begin(), corners->points.end()); // 头插进总角点集合
            }
        }

        if (prepare_corner_points->size() > 0)
            nms3d(prepare_corner_points);
        // NOTE: 下面这个 if 后面只有一句注释, 没有语句体, 因此再下面的 if 整个成为它的子语句
        // (等价于两个条件串联判断, 行为与单个 if 相同); 疑似删掉调试输出后遗留的空 if, 未改动代码
        if (prepare_corner_points->size() > 0)
            // std::cout << "after nms3d: " << prepare_corner_points->size() << std::endl;

        if (prepare_corner_points->size() > config.max_corner_num)
        {
            /**
             * lambda 比较器: 按 intensity(角点强度)降序排序,
             * 使强度高的角点排在前面, 便于截断保留前 max_corner_num 个
             */
            std::sort(prepare_corner_points->points.begin(), prepare_corner_points->points.end(), [](pcl::PointXYZINormal &p1, pcl::PointXYZINormal &p2) -> bool
                      { return p1.intensity > p2.intensity; });
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
            prepare_key_cloud->points.assign(prepare_corner_points->points.begin(), prepare_corner_points->points.begin() + config.max_corner_num); // 截断
            prepare_corner_points = prepare_key_cloud;
        }
        return prepare_corner_points;
    }

    /**
     * @brief 角点 3D 非极大值抑制: 半径 nms_3d_range 内只保留 intensity 最大者
     *
     * 步骤:
     *   1. 用输入角点云构建 KdTree(FLANN)
     *   2. 逐点半径搜索(结果包含查询点自身),
     *      若邻域内存在 intensity 不小于自身的其它点, 则把本点 curvature 置 0(被压制)
     *   3. 用 curvature == 1.0 的点重建点云并原地覆盖入参
     *
     * @param prepare_key_cloud 输入/输出角点云; curvature 复用为保留标记(1.0 保留 / 0.0 抑制, 初值由 nms2d 置 1.0)
     * @note 在 extractCorners() 中调用, 用于合并不同平面上提取出的重复角点
     */
    void STDManager::nms3d(pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud)
    {
        pcl::KdTreeFLANN<pcl::PointXYZINormal> kd_tree; // FLANN KdTree
        kd_tree.setInputCloud(prepare_key_cloud);
        std::vector<int> pointIdxRadiusSearch; // 半径搜索结果: 邻域点下标
        std::vector<float> pointRadiusSquaredDistance; // 对应平方距离(此处未使用)

        for (size_t i = 0; i < prepare_key_cloud->size(); i++)
        {
            pcl::PointXYZINormal searchPoint = prepare_key_cloud->points[i]; // 查询点
            if (kd_tree.radiusSearch(searchPoint, config.nms_3d_range, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
            {
                for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j)
                {
                    if (pointIdxRadiusSearch[j] == i)
                        continue; // 跳过查询点自身(半径搜索包含它)
                    // NOTE: 使用 <= 比较, 强度完全相同的两个点会互相压制而同时被删除
                    if (prepare_key_cloud->points[i].intensity <= prepare_key_cloud->points[pointIdxRadiusSearch[j]].intensity)
                    {
                        prepare_key_cloud->points[i].curvature = 0.0; // 邻域内已有不比自己弱的点: 压制本点
                        break;
                    }
                }
            }
        }
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr temp(new pcl::PointCloud<pcl::PointXYZINormal>);
        temp->reserve(prepare_key_cloud->size());
        for (auto &p : prepare_key_cloud->points)
            if (p.curvature == 1.0)
                temp->push_back(p); // 只保留未被压制的点
        prepare_key_cloud->clear();
        pcl::copyPointCloud(*temp, *prepare_key_cloud); // 用过滤结果原地覆盖入参
    }

    /**
     * @brief 由角点云构建 STD 三角形描述子
     *
     * 以每个角点作为三角形第一个顶点, 用 KdTree 取前 desc_search_range 个最近邻(含自身),
     * 从中任选两个不同邻居组成三角形:
     *   1. 三边长为点间 3D 欧氏距离: s1 = |p1 - p2|, s2 = |p2 - p3|, s3 = |p3 - p1|
     *   2. 任一边超出 [min_side_len, max_side_len] 则丢弃(过短边噪声大, 过长边随距离退化)
     *   3. 顶点规范化: 因为边长与旋转/平移无关, 可按“对边长度升序”给顶点重排,
     *      使顶点 A 对最短边 s1, B 对 s2, C 对 s3(见下面 arr_pair / temp_ps 的对应关系),
     *      从而同一个三角形在任意坐标系下生成的描述子同名顶点一致
     *   4. 用 (s1, s2, s3) 以 0.001 米分辨率量化成去重键, 同一组边长只生成一个描述子
     *   5. 填充描述子: 重心、三顶点、attached(顶点 intensity 三元组)、
     *      side_length(升序边长)、norms(顶点法向)、angle(法向夹角特征)
     *
     * 描述子的比较量(side_length, angle)是旋转/平移不变量, 因此可用于跨帧匹配
     *
     * @param cloud 角点云(extractCorners 输出), 点的 normal 字段是来源平面的法向
     * @param desc  输出描述子数组(追加式写入)
     * @note 匹配阶段 selectCanidates() 用 side_length 检索, triangleSolver() 用 vertex_a/b/c 与 center 求位姿
     */
    void STDManager::buildDescriptor(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud, std::vector<STDDescriptor> &desc)
    {
        std::unordered_set<VoxelKey, VoxelKey::Hasher> flag; // 边长去重集合: 键为量化后的 (s1, s2, s3)
        pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(
            new pcl::KdTreeFLANN<pcl::PointXYZINormal>); // 角点云的 KNN 索引
        kd_tree->setInputCloud(cloud);
        std::vector<int> pointIdxNKNSearch(config.desc_search_range); // KNN 结果下标(长度会被 PCL 按实际找到的点数调整)
        std::vector<float> pointNKNSquaredDistance(config.desc_search_range); // 对应平方距离(此处未使用)
        for (size_t i = 0; i < cloud->size(); i++)
        {
            pcl::PointXYZINormal searchPoint = cloud->points[i]; // 三角形第一个顶点
            // NOTE: 返回值为实际找到的邻居数, 不保证等于 desc_search_range; 而下面的循环按固定的
            // [1, desc_search_range) 下标访问结果, 当角点数少于 desc_search_range 时存在越界读风险
            if (kd_tree->nearestKSearch(searchPoint, config.desc_search_range, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                for (int m = 1; m < config.desc_search_range - 1; m++) // 邻居下标 1..13(下标 0 是查询点自身, 距离 0)
                {
                    for (int n = m + 1; n < config.desc_search_range; n++) // 邻居下标 m + 1..14, 保证两个邻居不同
                    {
                        pcl::PointXYZINormal p1 = searchPoint; // 顶点 1(查询点)
                        pcl::PointXYZINormal p2 = cloud->points[pointIdxNKNSearch[m]]; // 顶点 2(第 m 近邻)
                        pcl::PointXYZINormal p3 = cloud->points[pointIdxNKNSearch[n]]; // 顶点 3(第 n 近邻)

                        double s1 = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2)); // 边 p1-p2
                        double s2 = sqrt(pow(p2.x - p3.x, 2) + pow(p2.y - p3.y, 2) + pow(p2.z - p3.z, 2)); // 边 p2-p3
                        double s3 = sqrt(pow(p3.x - p1.x, 2) + pow(p3.y - p1.y, 2) + pow(p3.z - p1.z, 2)); // 边 p3-p1

                        if (s1 < config.min_side_len || s1 > config.max_side_len ||
                            s2 < config.min_side_len || s2 > config.max_side_len ||
                            s3 < config.min_side_len || s3 > config.max_side_len)
                        {
                            continue; // 任一边超出长度范围: 丢弃该三角形
                        }
                        std::vector<std::pair<double, int>> arr_pair{{s1, 0}, {s2, 1}, {s3, 2}}; // (边长值, 原始下标): 0/1/2 对应 s1/s2/s3
                        /**
                         * lambda 比较器: 按边长值(pair.first)升序排序,
                         * 排序后 arr_pair[0].second 即三条边中最短边在 s1/s2/s3 中的下标
                         */
                        std::sort(arr_pair.begin(), arr_pair.end(), [](std::pair<double, int> &i1, std::pair<double, int> &i2)
                                  { return i1.first < i2.first; });
                        // (边长值, 该边所对的顶点): s1 是 p1-p2 的边, 对顶点 p3; s2 对 p1; s3 对 p2
                        std::vector<std::pair<double, pcl::PointXYZINormal>> temp_ps{{s1, p3}, {s2, p1}, {s3, p2}};
                        p1 = temp_ps[arr_pair[0].second].second; // 顶点 A: 对最短边
                        p2 = temp_ps[arr_pair[1].second].second; // 顶点 B: 对中间边
                        p3 = temp_ps[arr_pair[2].second].second; // 顶点 C: 对最长边

                        s1 = temp_ps[arr_pair[0].second].first; // 最短边
                        s2 = temp_ps[arr_pair[1].second].first; // 中间边
                        s3 = temp_ps[arr_pair[2].second].first; // 最长边
                        VoxelKey k = VoxelKey::index(s1, s2, s3, 0.001); // 三边长按 0.001 米分辨率量化为去重键
                        if (flag.find(k) != flag.end())
                        {
                            continue; // NOTE: 同一帧内该 (s1, s2, s3) 组合已生成过描述子;
                                      // 去重是全局的, 空间中位置不同但三边长相同的三角形也会被丢弃
                        }
                        flag.insert(k);
                        STDDescriptor one_des;
                        one_des.center = Eigen::Vector3d(p1.x + p2.x + p3.x, p1.y + p2.y + p3.y, p1.z + p2.z + p3.z) / 3.0; // 重心
                        one_des.vertex_a = Eigen::Vector3d(p1.x, p1.y, p1.z); // 顶点 A
                        one_des.vertex_b = Eigen::Vector3d(p2.x, p2.y, p2.z); // 顶点 B
                        one_des.vertex_c = Eigen::Vector3d(p3.x, p3.y, p3.z); // 顶点 C
                        one_des.attached = Eigen::Vector3d(p1.intensity, p2.intensity, p3.intensity); // 三个顶点的角点强度
                        one_des.side_length = Eigen::Vector3d(s1, s2, s3); // 升序边长
                        one_des.norms.col(0) = Eigen::Vector3d(p1.normal_x, p1.normal_y, p1.normal_z); // 顶点 A 的法向
                        one_des.norms.col(1) = Eigen::Vector3d(p2.normal_x, p2.normal_y, p2.normal_z); // 顶点 B 的法向
                        one_des.norms.col(2) = Eigen::Vector3d(p3.normal_x, p3.normal_y, p3.normal_z); // 顶点 C 的法向
                        // angle 特征: 三个顶点法向两两之间点积(即夹角余弦)取绝对值再乘 5;
                        // 法向是旋转不变量, 所以该特征只与三个平面的相对朝向有关(注意不是三角形的内角);
                        // 取绝对值是因为平面法向本身有正负二义性
                        one_des.angle = Eigen::Vector3d(
                            std::abs(5 * one_des.norms.col(0).dot(one_des.norms.col(1))),
                            std::abs(5 * one_des.norms.col(1).dot(one_des.norms.col(2))),
                            std::abs(5 * one_des.norms.col(2).dot(one_des.norms.col(0))));
                        desc.push_back(one_des); // 收集描述子
                    }
                }
            }
        }
    }

    /**
     * @brief 汇总当前帧所有平面体素为一个平面点云: 每个平面体素出一个点
     *        (位置 = 平面质心, normal 字段 = 平面法向)
     * @return 平面点云; 供 verifyGeoPlane / verifyGeoPlaneICP 几何验证使用, 并随 STDFeature 存入 cloud_vec
     */
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr STDManager::currentPlaneCloud()
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr ret(new pcl::PointCloud<pcl::PointXYZINormal>);
        for (auto it = temp_voxels.begin(); it != temp_voxels.end(); it++)
        {
            if (!it->second.is_plane)
                continue; // 非平面体素跳过
            pcl::PointXYZINormal p;
            p.x = it->second.mean.x(); // 位置取平面质心
            p.y = it->second.mean.y();
            p.z = it->second.mean.z();
            p.normal_x = it->second.norms.col(0).x(); // 法向取平面法向(norms 第 0 列)
            p.normal_y = it->second.norms.col(0).y();
            p.normal_z = it->second.norms.col(0).z();
            ret->push_back(p);
        }
        return ret;
    }

    /**
     * @brief 单帧特征提取入口: 从原始点云到 STDFeature 的完整流水线
     *
     * 步骤:
     *   1. 清空上一帧的 temp_voxels(与空表 swap, 同时释放内存)
     *   2. buildVoxels(): 体素化 + 统计量 + 平面判定
     *   3. buildConnections(): 建立 6 邻域连接
     *   4. extractCorners(): 提取角点(2D-NMS + 3D-NMS)
     *   5. 角点存在时 buildDescriptor() 构建三角形描述子
     *   6. currentPlaneCloud(): 汇总平面点云
     *   7. feature.id = frame_count(此处不自增, 由 insert() 自增)
     *
     * @param cloud 当前关键帧点云
     * @return 该帧特征
     * @note 调用方(pgo_node)随后调用 searchLoop() 检索回环, 再调用 insert() 入库
     * @note 结构体中的 stamp / valid 字段在此处未赋值
     */
    STDFeature STDManager::extract(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
    {
        std::unordered_map<VoxelKey, VoxelNode, VoxelKey::Hasher>().swap(temp_voxels); // 清空并释放上一帧体素表

        buildVoxels(cloud);

        buildConnections();

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr corners = extractCorners();

        STDFeature feature;

        if (corners->size() > 0)
            buildDescriptor(corners, feature.descs); // 有角点才构建描述子

        feature.cloud = currentPlaneCloud(); // 平面点云(几何验证用)

        feature.id = frame_count; // 当前帧号(自增在 insert 中)
        return feature;
    }

    /**
     * @brief 把一帧特征写入历史数据库
     *
     * 步骤:
     *   1. 平面点云按帧序压入 cloud_vec,
     *      因此 cloud_vec 下标 == 帧号(verifyCandidate / pgo_node 依赖该约定)
     *   2. 描述子 id 全部置为当前 frame_count
     *   3. 计算边长哈希桶 position = (int)(边长 / side_resolution + 0.5),
     *      对正数而言等价于四舍五入取整
     *   4. 描述子按桶加入 data_base(按值拷贝存储, feature 本身不保留)
     *   5. frame_count 自增, 为下一帧分配新帧号
     *
     * @param feature 待入库特征(其 id 与内部描述子的 id 会被改写为当前帧号)
     * @note cloud_vec 只增不减(内存随帧数持续增长)
     * NOTE: 若某帧只 extract 不 insert, 后续帧号与 cloud_vec 下标会错位(当前 pgo_node 每帧都 insert, 满足约定)
     */
    void STDManager::insert(STDFeature &feature)
    {
        cloud_vec.push_back(feature.cloud); // 平面点云按帧序保存(下标即帧号)
        feature.id = frame_count; // 以全局帧号为准覆盖
        for (STDDescriptor &des : feature.descs)
        {
            des.id = frame_count; // 描述子记录所属帧号(时间过滤时使用)
            VoxelKey position;
            position.x = (int)(des.side_length[0] / config.side_resolution + 0.5); // 最短边量化后的桶下标
            position.y = (int)(des.side_length[1] / config.side_resolution + 0.5); // 中间边
            position.z = (int)(des.side_length[2] / config.side_resolution + 0.5); // 最长边
            auto iter = data_base.find(position);
            if (iter == data_base.end())
            {
                data_base[position] = std::vector<STDDescriptor>(); // 新桶: 先建立空列表
            }
            data_base[position].push_back(des); // 按桶存储
        }
        frame_count++; // 帧号自增, 供下一帧使用
    }

    /**
     * @brief 由一对匹配的三角形描述子求解二者之间的刚体变换 (R, t)
     *
     * 与 SVD 点云配准(Kabsch 方法)同思路, 区别是三个顶点已相对各自重心中心化:
     *   1. 构造 3x3 矩阵: src 的三列为源三角形顶点相对其重心的坐标,
     *      ref 的三列为目标三角形顶点相对其重心的坐标
     *   2. 相关矩阵 H = src * ref^T, 做 SVD: H = U * S * V^T
     *   3. 旋转 R = V * U^T; 若 det(R) < 0(出现镜像解), 用 K = diag(1, 1, -1) 修正: R = V * K * U^T
     *   4. 平移 t = -R * center_src + center_ref, 即把源三角形重心映射到目标三角形重心
     *
     * @param pair  匹配对: first 为源(当前帧)描述子, second 为目标(历史帧)描述子
     * @param rot   输出旋转矩阵 R
     * @param trans 输出平移向量 t, 满足 p_target ≈ R * p_source + t
     * @note 只用 3 对点求解, 对顶点对应错误非常敏感;
     *       verifyCandidate() 用多组三角形做假设-投票来抵抗个别的错误匹配
     */
    void STDManager::triangleSolver(std::pair<STDDescriptor, STDDescriptor> &pair, Eigen::Matrix3d &rot, Eigen::Vector3d &trans)
    {
        Eigen::Matrix3d src = Eigen::Matrix3d::Zero(); // 源三角形: 列为顶点相对重心的坐标
        Eigen::Matrix3d ref = Eigen::Matrix3d::Zero(); // 目标三角形: 列为顶点相对重心的坐标
        src.col(0) = pair.first.vertex_a - pair.first.center; // 去中心化
        src.col(1) = pair.first.vertex_b - pair.first.center;
        src.col(2) = pair.first.vertex_c - pair.first.center;
        ref.col(0) = pair.second.vertex_a - pair.second.center;
        ref.col(1) = pair.second.vertex_b - pair.second.center;
        ref.col(2) = pair.second.vertex_c - pair.second.center;
        Eigen::Matrix3d covariance = src * ref.transpose(); // 相关矩阵 H = Σ src_i * ref_i^T
        // NOTE: 这里用动态尺寸的 MatrixXd(3x3 用 Matrix3d 即可), 每次调用有一次堆分配开销
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV); // H = U * S * V^T
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d U = svd.matrixU();
        rot = V * U.transpose(); // 旋转 = V * U^T
        if (rot.determinant() < 0) // 行列式为负: 得到的是镜像(反射)解, 需要修正
        {
            Eigen::Matrix3d K; // 镜像修正矩阵
            K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
            rot = V * K * U.transpose(); // 修正后的旋转(行列式为正)
        }
        trans = -rot * pair.first.center + pair.second.center; // 平移: 令源重心映射到目标重心
    }

    /**
     * @brief 用当前帧的描述子检索历史数据库, 生成候选匹配帧列表
     *
     * 步骤:
     *   1. 预生成 27 个整数偏移 {-1, 0, 1}^3, 用于在边长桶空间做 3x3x3 邻域查找
     *   2. 对当前帧的每个描述子:
     *      a. 计算边长初筛阈值 dis_threshold = |side_length / side_resolution| * rough_dis_threshold
     *         (与自身尺度成比例的相对阈值)
     *      b. 遍历 27 个邻域桶: 先算桶中心(桶下标 + 0.5), 若描述子的缩放边长到桶中心的欧氏距离
     *         大于等于 1.5 则跳过该桶(限制跨桶查找范围); 桶不存在也跳过
     *      c. 对桶内每个历史描述子: 帧号差 <= skip_near_num 的跳过(避免时间近邻);
     *         缩放边长之间的欧氏距离 >= dis_threshold 的跳过;
     *         attached(intensity 三元组)的相对差 2 * |a - b| / |a + b| >= vertex_diff_threshold 的跳过;
     *         全部通过则记录 (描述子下标 i, 所在桶, 桶内下标 j)
     *   3. 统计每个历史帧号获得的票数(match_arr), index_recorder 记录每张票的来源
     *   4. 最多输出 candidate_num 个候选: 每轮取票数最大的帧号(票数 < 5 或找不到时结束),
     *      收集属于该帧的全部描述子对组成一个 STDMatch, 并把该帧票数清零, 避免重复输出
     *
     * @param feature    当前帧特征(使用其 descs 与 id)
     * @param match_list 输出候选匹配列表, 按票数从高到低
     * @note 数据来源是 insert() 写入的 data_base; 候选的验证见 searchLoop -> verifyCandidate
     */
    void STDManager::selectCanidates(STDFeature &feature, std::vector<STDMatch> &match_list)
    {
        std::vector<Eigen::Vector3i> voxel_round; // 27 个整数偏移

        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    Eigen::Vector3i voxel_inc(x, y, z);
                    voxel_round.push_back(voxel_inc);
                }
            }
        }

        std::vector<bool> useful_match(feature.descs.size(), false); // 每个描述子是否有命中
        std::vector<std::vector<size_t>> useful_match_index(feature.descs.size()); // 命中候选在该桶内的下标
        std::vector<std::vector<VoxelKey>> useful_match_position(feature.descs.size()); // 命中候选所在的桶
        // 根据三角边长选择候选
        for (size_t i = 0; i < feature.descs.size(); i++)
        {
            STDDescriptor &src_std = feature.descs[i]; // 当前描述子(源)
            VoxelKey position;
            double dis_threshold = (src_std.side_length / config.side_resolution).norm() * config.rough_dis_threshold; // 边长初筛的相对阈值

            for (Eigen::Vector3i &voxel_inc : voxel_round) // 遍历 3x3x3 个相邻边长桶
            {
                position.x = (int)(src_std.side_length[0] / config.side_resolution + voxel_inc[0]); // 桶下标 = 量化边长 + 整数偏移
                position.y = (int)(src_std.side_length[1] / config.side_resolution + voxel_inc[1]);
                position.z = (int)(src_std.side_length[2] / config.side_resolution + voxel_inc[2]);
                Eigen::Vector3d voxel_center((double)position.x + 0.5, (double)position.y + 0.5, (double)position.z + 0.5); // 桶中心(桶坐标)

                if ((src_std.side_length / config.side_resolution - voxel_center).norm() >= 1.5)
                    continue; // 桶中心太远(阈值 1.5 个桶单位): 跳过
                auto iter = data_base.find(position);
                if (iter == data_base.end())
                    continue; // 桶不存在

                for (size_t j = 0; j < data_base[position].size(); j++)
                {
                    // NOTE: 两个 id 都是 uint64_t, 该差值为无符号; 若出现候选 id 大于当前 id 的异常顺序会下溢成极大值,
                    // 导致候选被全部误过滤
                    if ((feature.id - data_base[position][j].id) <= config.skip_near_num)
                        continue; // 跳过时间上过近的历史帧
                    double dis = (src_std.side_length / config.side_resolution - data_base[position][j].side_length / config.side_resolution).norm(); // 缩放边长之间的欧氏距离
                    if (dis >= dis_threshold)
                        continue; // 边长差异过大
                    double vertex_attach_diff = 2.0 * (src_std.attached - data_base[position][j].attached).norm() /
                                                (src_std.attached + data_base[position][j].attached).norm(); // 顶点强度的相对差 2*|a-b|/|a+b|
                    if (vertex_attach_diff >= config.vertex_diff_threshold)
                        continue; // 顶点强度分布差异过大(attached 为角点强度, 恒为正, 分母不为 0)
                    useful_match[i] = true; // 记录一个有效命中
                    useful_match_position[i].push_back(position);
                    useful_match_index[i].push_back(j);
                }
            }
        }

        std::unordered_map<uint64_t, int> match_arr; // 历史帧号 -> 得票数(每命中一个描述子记一票)
        std::vector<int> match_id_vec; // 与 index_recorder 一一对应的帧号
        std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> index_recorder; // (描述子下标 i, 命中在 useful_match_* 中的下标 j)

        // 记录所有匹配上的frame_id
        for (size_t i = 0; i < useful_match.size(); i++)
        {
            if (!useful_match[i])
                continue;
            for (size_t j = 0; j < useful_match_index[i].size(); j++)
            {
                uint64_t match_id = data_base[useful_match_position[i][j]][useful_match_index[i][j]].id; // 命中描述子所属历史帧号
                if (match_arr.find(match_id) == match_arr.end())
                    match_arr[match_id] = 0;
                match_arr[match_id]++; // 该帧得票加一
                index_recorder.emplace_back(i, j); // 记录票的来源
                match_id_vec.push_back(match_id);
            }
        }

        for (int cnt = 0; cnt < config.candidate_num; cnt++) // 最多输出 candidate_num 个候选
        {

            int max_vote = 1; // 当前发现的最高票数
            uint64_t max_vote_index = -1; // 最高票对应的帧号(-1 即 uint64 最大值, 表示尚未找到)

            for (auto it = match_arr.begin(); it != match_arr.end(); it++) // 线性扫描找票数最多的帧
            {
                if (it->second > max_vote)
                {
                    max_vote = it->second;
                    max_vote_index = it->first;
                }
            }
            // NOTE: max_vote_index 为 uint64_t, 与 0 比较恒为 false(无符号类型不可能小于 0);
            // 真正生效的结束条件是 max_vote < 5(投票不足 5 票的候选不输出)
            if (max_vote_index < 0 || max_vote < 5)
                break;
            STDMatch match_triangle;
            match_arr[max_vote_index] = 0; // 清零该帧票数, 防止同一帧被重复输出
            match_triangle.match_id = max_vote_index;
            for (size_t i = 0; i < index_recorder.size(); i++)
            {
                if (match_id_vec[i] != max_vote_index)
                    continue; // 只收集属于当前选出帧号的描述子对
                std::pair<STDDescriptor, STDDescriptor> single_match_pair;
                single_match_pair.first = feature.descs[index_recorder[i][0]]; // 当前帧描述子
                single_match_pair.second =
                    data_base[useful_match_position[index_recorder[i][0]][index_recorder[i][1]]]
                             [useful_match_index[index_recorder[i][0]][index_recorder[i][1]]]; // 历史帧描述子(按记录的桶与下标取回)
                match_triangle.match_pairs.push_back(single_match_pair);
            }
            match_list.push_back(match_triangle);
        }
    }

    /**
     * @brief 对一个候选匹配帧做三角形一致性验证, 估计位姿并给出匹配得分
     *
     * 步骤:
     *   1. 从匹配对中按固定间隔抽样(最多约 50 组)作为位姿假设:
     *      每组用 triangleSolver() 解出一个 (R, t)
     *   2. 对每组假设做“投票”: 遍历全部匹配对, 把源三角形的三个顶点按 (R, t) 变换后,
     *      与目标三角形对应顶点的距离都 < verify_dis_thresh(米)才计一票
     *   3. 取票数最多的假设; 票数 < 4 认为候选不可信, 返回 -1.0
     *   4. 用最佳假设对应的匹配对重新解一次 (R, t) 作为最终位姿,
     *      并收集所有在该位姿下顶点一致的匹配对到 success_match_vec
     *   5. 用平面点云做几何验证 verifyGeoPlane()(源 = 当前帧平面云, 目标 = 候选帧平面云),
     *      返回匹配比例作为得分
     *
     * @param matches           候选匹配(含 match_id 与全部描述子对)
     * @param rot, trans        输出位姿(由最佳三角形对解出)
     * @param success_match_vec 输出在最终位姿下顶点一致的匹配对
     * @param plane_cloud       当前帧平面点云(源)
     * @return 几何验证得分(0 ~ 1 的匹配比例); 投票不足时返回 -1.0
     */
    double STDManager::verifyCandidate(STDMatch &matches,
                                       Eigen::Matrix3d &rot,
                                       Eigen::Vector3d &trans,
                                       std::vector<std::pair<STDDescriptor, STDDescriptor>> &success_match_vec,
                                       pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud)
    {
        success_match_vec.clear();
        int skip_len = (int)(matches.match_pairs.size() / 50) + 1; // 抽样间隔: 使抽样组数不超过约 50
        int use_size = matches.match_pairs.size() / skip_len; // 实际抽样组数
        std::vector<int> vote_list(use_size); // 每组假设的得票数
        for (size_t i = 0; i < use_size; i++)
        {
            auto single_pair = matches.match_pairs[i * skip_len]; // 按间隔取一个匹配对作为位姿假设
            Eigen::Matrix3d test_rot;
            Eigen::Vector3d test_t;
            int vote = 0;
            triangleSolver(single_pair, test_rot, test_t); // 由该三角形对解出 (R, t) 假设
            for (size_t j = 0; j < matches.match_pairs.size(); j++)
            {
                auto verify_pair = matches.match_pairs[j];
                Eigen::Vector3d A = verify_pair.first.vertex_a; // 源三角形顶点 A
                Eigen::Vector3d A_transform = test_rot * A + test_t; // 按假设变换到目标坐标系
                Eigen::Vector3d B = verify_pair.first.vertex_b;
                Eigen::Vector3d B_transform = test_rot * B + test_t;
                Eigen::Vector3d C = verify_pair.first.vertex_c;
                Eigen::Vector3d C_transform = test_rot * C + test_t;
                double dis_A = (A_transform - verify_pair.second.vertex_a).norm(); // 与目标顶点 A 的残差
                double dis_B = (B_transform - verify_pair.second.vertex_b).norm(); // 与目标顶点 B 的残差
                double dis_C = (C_transform - verify_pair.second.vertex_c).norm(); // 与目标顶点 C 的残差
                if (dis_A < config.verify_dis_thresh && dis_B < config.verify_dis_thresh && dis_C < config.verify_dis_thresh)
                    vote++; // 三顶点残差同时达标: 该匹配对支持此假设
            }
            vote_list[i] = vote;
        }

        int max_vote_index = 0; // 最佳假设在抽样列表中的下标
        int max_vote = 0; // 最佳假设的票数
        for (size_t i = 0; i < vote_list.size(); i++)
        {
            if (max_vote < vote_list[i])
            {
                max_vote_index = i;
                max_vote = vote_list[i];
            }
        }

        // std::cout << "max vot" << max_vote << std::endl;

        if (max_vote < 4) // 4
            return -1.0; // 票数不足: 该候选不可信
        auto best_pair = matches.match_pairs[max_vote_index * skip_len]; // 最佳假设对应的匹配对
        triangleSolver(best_pair, rot, trans); // 用它重新求解, 输出最终位姿
        for (size_t j = 0; j < matches.match_pairs.size(); j++)
        {
            auto verify_pair = matches.match_pairs[j];
            Eigen::Vector3d A = verify_pair.first.vertex_a; // 源三角形顶点 A
            Eigen::Vector3d A_transform = rot * A + trans; // 按最终位姿变换
            Eigen::Vector3d B = verify_pair.first.vertex_b;
            Eigen::Vector3d B_transform = rot * B + trans;
            Eigen::Vector3d C = verify_pair.first.vertex_c;
            Eigen::Vector3d C_transform = rot * C + trans;
            double dis_A = (A_transform - verify_pair.second.vertex_a).norm(); // 与目标顶点 A 的残差
            double dis_B = (B_transform - verify_pair.second.vertex_b).norm(); // 与目标顶点 B 的残差
            double dis_C = (C_transform - verify_pair.second.vertex_c).norm(); // 与目标顶点 C 的残差
            if (dis_A < config.verify_dis_thresh && dis_B < config.verify_dis_thresh && dis_C < config.verify_dis_thresh)
                success_match_vec.push_back(verify_pair); // 在该位姿下一致的匹配对
        }
        // std::cout << "do verify" << std::endl;

        // 平面几何验证; 注意 cloud_vec 下标 == 帧号(见 insert), 因此用 match_id 直接索引
        return verifyGeoPlane(plane_cloud, cloud_vec[matches.match_id], rot, trans);
    }

    /**
     * @brief 平面点云几何验证: 统计源平面点变换后能找到对应平面点的比例
     *
     * 步骤(逐源点):
     *   1. 点与法向都施加变换: p' = R * p + t, n' = R * n
     *   2. 在目标平面云上取 3 个最近邻
     *   3. 只要有 1 个邻居同时满足:
     *        法向平行: |n' - n_t| < norm_merge_thresh 或 |n' + n_t| < norm_merge_thresh
     *        点到平面距离: |n_t · (p' - p_t)| < geo_verify_dis_thresh
     *      该源点即算匹配成功(break)
     *
     * @param source_cloud 源平面点云(当前帧)
     * @param target_cloud 目标平面点云(候选历史帧)
     * @param rot, trans   待评估的变换
     * @return 匹配成功的源点比例(0 ~ 1), 作为 verifyCandidate 的得分
     * NOTE: source_cloud 为空时除数为 0, 返回 NaN(当前调用链中 source 为当前帧平面云, 一般非空)
     */
    double STDManager::verifyGeoPlane(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                                      const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans)
    {
        pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
        kd_tree->setInputCloud(target_cloud); // 在目标点云上建 KdTree(注意参数顺序: 源在前, 目标在后)
        std::vector<int> pointIdxNKNSearch; // 最近邻下标
        std::vector<float> pointNKNSquaredDistance; // 对应平方距离(此处未使用)
        double useful_match = 0; // 匹配成功的源点数
        for (size_t i = 0; i < source_cloud->size(); i++)
        {
            pcl::PointXYZINormal searchPoint = source_cloud->points[i]; // 源平面点
            Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
            pi = rot * pi + trans; // 位置按变换映射到目标坐标系
            Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
            ni = rot * ni; // 法向只旋转不平移
            pcl::PointXYZINormal use_search_point;
            use_search_point.x = pi.x();
            use_search_point.y = pi.y();
            use_search_point.z = pi.z();
            if (kd_tree->nearestKSearch(use_search_point, 3, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                for (size_t j = 0; j < 3; j++) // 检查 3 个最近邻
                {
                    pcl::PointXYZINormal nearstPoint = target_cloud->points[pointIdxNKNSearch[j]];
                    Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z); // 目标点
                    Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z); // 目标法向
                    Eigen::Vector3d normal_inc = ni - tni; // 法向差: 同向时接近 0
                    Eigen::Vector3d normal_add = ni + tni; // 法向和: 反向时接近 0
                    double point_to_plane = std::abs(tni.transpose() * (pi - tpi)); // 源点到目标平面(过 tpi, 法向 tni)的距离

                    // std::cout << "point_to_plane: " << point_to_plane << ":" << trans.transpose() << std::endl;

                    if ((normal_inc.norm() < config.norm_merge_thresh || normal_add.norm() < config.norm_merge_thresh) && point_to_plane < config.geo_verify_dis_thresh)
                    {
                        useful_match++; // 法向平行且距离达标: 该源点匹配成功
                        break;
                    }
                }
            }
        }
        return useful_match / source_cloud->size(); // 匹配比例(0 ~ 1)
    }

    /**
     * @brief 平面点云 ICP: 最近邻建对应 + 高斯-牛顿迭代优化位姿, 返回平均点到平面残差
     *
     * 第一阶段(建立对应): 对每个源点施加当前 (rot, trans) 后取目标云最近邻,
     *   要求法向平行(norm_merge_thresh)、点到平面距离 < geo_verify_dis_thresh、点到点距离 < 3(硬编码);
     *   对应关系保存为 (未变换的源点, 目标点, 目标法向)
     * 第二阶段(SE(3) 迭代优化, 高斯-牛顿):
     *   残差 res = n_t · (R * p_i + t - p_t)                点到目标平面的有符号距离
     *   雅可比 J = [ -n_t^T * R * [p_i]x , n_t^T ]          左 3 列对旋转(李代数)求导, 右 3 列对平移求导,
     *                                                        [p_i]x 为 p_i 的反对称矩阵
     *   H = Σ J^T * J, b = -Σ J^T * res
     *   增量 delta = H^{-1} * b; 旋转用 SO(3) 指数映射右乘更新, 平移直接相加
     *   若 delta 的最大分量 < iter_eps 则提前结束
     *
     * @param source_cloud 源平面点云(当前帧)
     * @param target_cloud 目标平面点云(历史帧)
     * @param rot, trans   输入初值, 同时被就地更新为优化后的位姿
     * @return 平均绝对点到平面残差
     * @note 本文件内未调用, 由 pgo_node 在检出回环后调用做进一步验证
     */
    double STDManager::verifyGeoPlaneICP(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                                         pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                                         Eigen::Matrix3d &rot, Eigen::Vector3d &trans)
    {
        pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
        kd_tree->setInputCloud(target_cloud); // 目标点云的 KdTree
        std::vector<int> pointIdxNKNSearch(1); // 最近邻(1 个)下标
        std::vector<float> pointNKNSquaredDistance(1); // 对应平方距离(此处未使用)

        std::vector<Eigen::Vector3d> matched_source; // 匹配上的源点(未变换, 优化时现算变换)
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> matched_target; // 对应目标点与目标法向

        for (size_t i = 0; i < source_cloud->size(); i++)
        {
            pcl::PointXYZINormal &searchPoint = source_cloud->points[i]; // 源平面点
            Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);

            pi = rot * pi + trans; // 按当前位姿映射到目标坐标系

            Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
            ni = rot * ni; // 法向只旋转

            pcl::PointXYZINormal use_search_point;
            use_search_point.x = pi[0];
            use_search_point.y = pi[1];
            use_search_point.z = pi[2];
            if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                pcl::PointXYZINormal nearstPoint = target_cloud->points[pointIdxNKNSearch[0]];
                Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z); // 目标点
                Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z); // 目标法向
                Eigen::Vector3d normal_inc = ni - tni; // 法向差: 同向时接近 0
                Eigen::Vector3d normal_add = ni + tni; // 法向和: 反向时接近 0
                double point_to_point_dis = (pi - tpi).norm(); // 点到点距离
                double point_to_plane = std::abs(tni.transpose() * (pi - tpi)); // 点到平面距离
                if ((normal_inc.norm() < config.norm_merge_thresh ||
                     normal_add.norm() < config.norm_merge_thresh) &&
                    point_to_plane < config.geo_verify_dis_thresh && point_to_point_dis < 3) // NOTE: 点到点距离阈值 3 为硬编码
                {
                    matched_source.emplace_back(searchPoint.x, searchPoint.y, searchPoint.z); // 保存源点原始坐标
                    matched_target.emplace_back(tpi, tni); // 保存对应目标点与法向
                }
            }
        }

        Eigen::Matrix<double, 1, 6> J; // 残差对 (旋转增量, 平移增量) 的 1x6 雅可比
        Eigen::Matrix<double, 6, 6> H; // 正规方程系数矩阵 Σ J^T * J
        Eigen::Matrix<double, 6, 1> b; // 正规方程右端项 -Σ J^T * res
        Eigen::Matrix<double, 6, 1> delta; // 增量 (旋转增量 3 维, 平移增量 3 维)

        Eigen::Matrix3d opti_rot = rot; // 待优化的旋转
        Eigen::Vector3d opti_trans = trans; // 待优化的平移
        double sum_res = -1.0; // 残差绝对值之和(-1.0 表示初始无效值)
        for (size_t i = 0; i < config.max_iter; i++)
        {
            H.setZero();
            b.setZero();
            sum_res = 0.0;
            for (size_t j = 0; j < matched_source.size(); j++)
            {
                Eigen::Vector3d &pi = matched_source[j]; // 源点(未变换)
                Eigen::Vector3d &tpi = matched_target[j].first; // 目标点
                Eigen::Vector3d &tni = matched_target[j].second; // 目标法向
                J.block<1, 3>(0, 0) = -tni.transpose() * opti_rot * Sophus::SO3d::hat(pi); // 对旋转增量求导
                J.block<1, 3>(0, 3) = tni.transpose(); // 对平移增量求导
                double res = tni.dot(opti_rot * pi + opti_trans - tpi); // 点到平面残差(有符号)
                H += J.transpose() * J;
                b -= J.transpose() * res;
                sum_res += std::abs(res); // 统计残差绝对值之和
            }
            // NOTE: 直接对 H 求逆而非分解求解; 且无对应点时 H 为全零矩阵, H.inverse() 会产生非数值结果(NaN/Inf)
            delta = H.inverse() * b;
            opti_rot *= Sophus::SO3d::exp(delta.block<3, 1>(0, 0)).matrix(); // 旋转用 SO(3) 指数映射右乘更新
            opti_trans += delta.block<3, 1>(3, 0); // 平移直接相加
            // NOTE: maxCoeff() 取的是最大“带符号”分量而非最大绝对值, 当各增量都是较大负值时可能提前结束;
            // 疑似应为 delta.cwiseAbs().maxCoeff() 或 delta.norm()
            if (delta.maxCoeff() < config.iter_eps)
                break;
        }
        rot = opti_rot; // 输出优化后的位姿
        trans = opti_trans;
        if (matched_source.size() > 0)
            return sum_res / static_cast<double>(matched_source.size()); // 平均绝对残差
        // NOTE: 无对应点时 sum_res 已在循环内被重置为 0.0, 这里返回 0.0 而非初值 -1.0;
        // 调用方若用 -1.0 作为“失败”标记需注意
        return sum_res;
    }

    /**
     * @brief 回环检索入口: 选候选 -> 逐个验证打分 -> 保留最高分
     *
     * 步骤:
     *   1. 初始化失败结果(valid = false, match_id = -1, score = 0)
     *   2. 当前帧没有描述子则直接返回
     *   3. selectCanidates() 生成候选匹配列表
     *   4. 对每个候选调用 verifyCandidate() 打分, 得分更高则覆盖结果
     *      (verifyCandidate 内部输出位姿 rot/trans 与一致的描述子对)
     *   5. 最高得分 > config.icp_thresh 才判定为有效回环
     *
     * @param feature 当前帧特征(需在 insert 之前调用, 此时 feature.id 即当前帧号)
     * @return LoopResult: valid / match_id / match_score / rotation / translation / match_pairs
     * @note 调用方 pgo_node 在帧号超过 skip_near_num 后才调用本函数, 之后调用 insert() 入库
     */
    LoopResult STDManager::searchLoop(STDFeature &feature)
    {
        LoopResult result;
        result.valid = false;
        result.match_id = -1; // uint64 赋 -1 即 2^64 - 1, 作为“无匹配”标记
        result.match_score = 0.0;
        if (feature.descs.size() == 0)
            return result; // 当前帧没有描述子, 无法检索

        std::vector<STDMatch> candidate_matcher_vec;
        selectCanidates(feature, candidate_matcher_vec); // 选出候选历史帧

        for (size_t i = 0; i < candidate_matcher_vec.size(); i++)
        {
            Eigen::Vector3d trans;
            Eigen::Matrix3d rot;
            std::vector<std::pair<STDDescriptor, STDDescriptor>> sucess_match_vec;
            double verify_score = verifyCandidate(candidate_matcher_vec[i], rot, trans, sucess_match_vec, feature.cloud); // 验证并打分(失败为 -1.0)

            // std::cout << "verify_score: " << verify_score << std::endl;

            if (verify_score > result.match_score) // 严格大于: 得分并列时保留先出现的候选
            {
                result.match_score = verify_score;
                result.match_id = candidate_matcher_vec[i].match_id;
                result.translation = trans;
                result.rotation = rot;
                result.match_pairs = sucess_match_vec;
            }
        }
        if (result.match_score > config.icp_thresh)
            result.valid = true; // 得分超过阈值: 判定为有效回环
        return result;
    }
} // namespace name
