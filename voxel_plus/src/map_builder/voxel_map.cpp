#include "voxel_map.h"

namespace lio
{

    // 全局自增 ID 计数器, 每新建一个体素发一个唯一的 group_id
    uint64_t VoxelGrid::count = 0;

    // 平面合并阈值, 实际值由 LIOBuilder 用配置覆盖
    double VoxelGrid::merge_thresh_for_angle = 0.1;    // 夹角阈值: 1 - cos(theta)
    double VoxelGrid::merge_thresh_for_distance = 0.04; // 平面间距阈值(原点到平面的距离差)

    /**
     * 构造函数: 记录三个阈值和体素索引, 并设置初始状态
     */
    VoxelGrid::VoxelGrid(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, VoxelKey _position, VoxelMap *_map)
        : max_point_thresh(_max_point_thresh),
          update_point_thresh(_update_point_thresh),
          plane_thresh(_plane_thresh),
          position(_position.x, _position.y, _position.z)
    {
        merged = false;                                      // 尚未参与合并
        group_id = VoxelGrid::count++;                       // 领取全局唯一 ID
        is_init = false;                                     // 尚未初始化
        is_plane = false;                                    // 尚未判定为平面
        temp_points.reserve(max_point_thresh);               // 预留容量, 避免反复扩容
        newly_add_point = 0;
        plane = std::make_shared<Plane>();
        update_enable = true;                                // 允许接收新点
        map = _map;                                          // 所属地图, merge() 时查邻居
        // center = Eigen::Vector3d(position.x + 0.5, position.y + 0.5, position.z + 0.5) * map->voxel_size;
        // 注意: center 没有在这里初始化, 而是在 updatePlane() 成功拟合后设为 plane->mean
    }

    /**
     * 把一个点累加进平面的在线统计量(O(1), 无需保留全部点)
     *   mean = Σp / n            在线均值
     *   ppt  = Σp * p^T          二阶矩
     *   n    = 点数
     */
    void VoxelGrid::addToPlane(const PointWithCov &pv)
    {
        plane->mean += (pv.point - plane->mean) / (plane->n + 1.0);
        plane->ppt += pv.point * pv.point.transpose();
        plane->n += 1;
    }

    /**
     * 仅累积: 统计量 + 原始点入 temp_points, 不做任何阈值判断
     * 供 VoxelMap::build() 批量建图使用(最后统一 updatePlane 一次)
     */
    void VoxelGrid::addPoint(const PointWithCov &pv)
    {
        addToPlane(pv);
        temp_points.push_back(pv);
    }

    /**
     * 在线增量插入, 体素内部的状态机(供 VoxelMap::update() 使用)
     *
     * 三个分支:
     *   1) 未初始化          : 每来一个点都直接累加并尝试拟合
     *   2) 已初始化且是平面  : update_enable 时按 update_point_thresh 周期重算,
     *                          达到 max_point_thresh 后冻结并释放 temp_points;
     *                          冻结后转而尝试与邻居 merge()
     *   3) 已初始化但不是平面: 继续攒点并周期重算, 期望点数足够后能判定为平面
     *                          (注意此分支在 update_enable=false 时什么都不做)
     */
    void VoxelGrid::pushPoint(const PointWithCov &pv)
    {
        if (!is_init)
        {
            // 分支1: 尚未初始化, 攒一个点就尝试拟合一次
            // (点数不足时 updatePlane 会直接 return, 所以前期相当于纯累积)
            addToPlane(pv);
            temp_points.push_back(pv);
            updatePlane();
        }
        else
        {
            if (is_plane)
            {
                if (update_enable)
                {
                    // 分支2a: 平面态且可更新, 按新增点数触发重算
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_add_point++;
                    if (newly_add_point >= update_point_thresh)
                    {
                        updatePlane();
                        newly_add_point = 0;
                    }
                    if (temp_points.size() >= max_point_thresh)
                    {
                        // 点数到上限: 冻结该体素, 并释放 temp_points 以节省内存
                        // 平面参数已经固定, 后续不再需要原始点参与计算
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
                else
                {
                    // 分支2b: 平面已冻结, 转为尝试与 6 邻域共面体素合并
                    merge();
                }
            }
            else
            {
                if (update_enable)
                {
                    // 分支3: 还不是平面, 继续攒点并周期重算
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_add_point++;
                    if (newly_add_point >= update_point_thresh)
                    {
                        updatePlane();
                        newly_add_point = 0;
                    }
                    if (temp_points.size() >= max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
            }
        }
    }

    /**
     * 由累积的点云拟合平面, 并把点的不确定度传播为 6x6 的平面参数协方差
     *
     * 步骤:
     *   1. 求样本协方差 cov = Σp*p^T / n - mean * mean^T
     *   2. 特征分解, 最小特征值对应的特征向量即平面法向
     *   3. 若最小特征值 > plane_thresh, 认为三个方向都散开(不是平面), is_plane = false
     *   4. 协方差传播: 把 (法向量, 质心) 视作点的函数, 用雅可比链式传播
     *   5. 统一法向量朝向, 使其一致指向远离原点的一侧
     */
    void VoxelGrid::updatePlane()
    {
        assert(temp_points.size() == plane->n); // 注意: temp_points 被 swap 清空后此断言不再成立
        if (plane->n < update_point_thresh)
            return; // 点数不足, 不改动状态(此时 is_init 保持 false)
        is_init = true;

        // 1. 样本协方差矩阵
        Eigen::Matrix3d cov = plane->ppt / static_cast<double>(plane->n) - plane->mean * plane->mean.transpose();

        // 2. 对称特征分解; evals 升序, evals(0) 最小, 对应法向方向
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Matrix3d evecs = es.eigenvectors();
        Eigen::Vector3d evals = es.eigenvalues();

        // 3. 平面判定: 最小特征值太大说明点云沿法向也有明显展布, 不是平面
        if (evals(0) > plane_thresh)
        {
            is_plane = false;
            return;
        }
        is_plane = true;

        // 4. 协方差传播
        //    J_Q: 质心对每个点的偏导 d(mean)/d(p) = I / n
        Eigen::Matrix3d J_Q = Eigen::Matrix3d::Identity() / static_cast<double>(plane->n);
        Eigen::Vector3d plane_norm = evecs.col(0);
        for (PointWithCov &pv : temp_points)
        {
            // J 为 6x3: 上半 block 是法向量对点的偏导, 下半 block 是质心对点的偏导
            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
            // 特征向量对协方差阵的导数: 1/(λ0-λm) * (e_m * n^T + n * e_m^T)
            // 只对 m = 1,2 两维求和(λ0 方向即法向量自身, 无需参与)
            for (int m = 1; m < 3; m++)
            {
                Eigen::Matrix<double, 1, 3> F_m = (pv.point - plane->mean).transpose() / ((plane->n) * (evals(0) - evals(m))) *
                                                  (evecs.col(m) * plane_norm.transpose() +
                                                   plane_norm * evecs.col(m).transpose());
                F.row(m) = F_m;
            }
            J.block<3, 3>(0, 0) = evecs * F;
            J.block<3, 3>(3, 0) = J_Q;
            // 累加每个点对平面协方差的贡献
            plane->cov += J * pv.cov * J.transpose();
        }

        // 5. 统一法向量朝向: 使法向量指向原点所在的一侧, 避免同一平面法向反号
        double axis_distance = -plane->mean.dot(plane_norm);
        if (axis_distance < 0.0)
            plane_norm = -plane_norm;
        plane->norm = plane_norm;
        center = plane->mean; // 平面中心取点云质心
    }

    /**
     * 与 6 邻域中已冻结的共面体素合并
     *
     * 判定条件(任一不满足则跳过):
     *   - 不属于同一 group_id(已合并过)
     *   - 邻居仍在更新(update_enable == true)
     *   - 邻居不是平面
     *   - 法向量夹角过大: 1 - cos > merge_thresh_for_angle
     *   - 两平面到原点的距离差过大: > merge_thresh_for_distance
     *
     * 合并方式为"标记式": 只统一平面参数和 group_id, 并不搬运点集、也不删除对方体素
     */
    void VoxelGrid::merge()
    {
        // 6 邻域: ±X / ±Y / ±Z
        std::vector<VoxelKey> near;
        near.push_back(VoxelKey(position.x - 1, position.y, position.z));
        near.push_back(VoxelKey(position.x, position.y - 1, position.z));
        near.push_back(VoxelKey(position.x, position.y, position.z - 1));
        near.push_back(VoxelKey(position.x + 1, position.y, position.z));
        near.push_back(VoxelKey(position.x, position.y + 1, position.z));
        near.push_back(VoxelKey(position.x, position.y, position.z + 1));

        for (VoxelKey &k : near)
        {
            auto it = map->featmap.find(k);
            if (it != map->featmap.end())
            {
                std::shared_ptr<VoxelGrid> near_node = it->second;
                // 同一组(已合并过) / 邻居还在更新 / 邻居不是平面 -> 跳过
                if (near_node->group_id == group_id || near_node->update_enable || !near_node->is_plane)
                    continue;

                // 夹角判定: 1 - cos(theta)
                double norm_distance = 1.0 - near_node->plane->norm.dot(plane->norm);
                // 平面间距判定: 两个平面到原点距离之差的绝对值
                double axis_distance = std::abs(near_node->plane->norm.dot(near_node->plane->mean) - plane->norm.dot(plane->mean));

                if (norm_distance > merge_thresh_for_angle || axis_distance > merge_thresh_for_distance)
                    continue;

                // 以两平面协方差的迹作为权重
                // 注意 block(0,0) 是法向量协方差, block(3,3) 是质心协方差
                double tn0 = plane->cov.block<3, 3>(0, 0).trace(),
                       tm0 = plane->cov.block<3, 3>(3, 3).trace(),
                       tn1 = near_node->plane->cov.block<3, 3>(0, 0).trace(),
                       tm1 = near_node->plane->cov.block<3, 3>(3, 3).trace();
                double tc0 = tn0 + tm0, tc1 = tn1 + tm1;
                // NOTE: 下面两行疑似缺少括号, 按运算符优先级等价于
                //   tm0 * mean1 + (tm1 * mean2) / (tm0 + tm1)
                // 而加权平均的本意应为 (tm0 * mean1 + tm1 * mean2) / (tm0 + tm1), 建议核对
                Eigen::Vector3d new_mean = tm0 * near_node->plane->mean + tm1 * plane->mean / (tm0 + tm1);
                Eigen::Vector3d new_norm = tn0 * near_node->plane->norm + tn1 * plane->norm / (tn0 + tn1);
                // 按权重融合两平面的 6x6 协方差
                Eigen::Matrix<double, 6, 6> new_cov = (tc0 * tc0 * near_node->plane->cov + tc1 * tc1 * plane->cov) / ((tc0 + tc1) * (tc0 + tc1));

                // 打上相同的 group_id, 避免后续重复合并
                near_node->group_id = group_id;
                merged = true;
                near_node->merged = true;

                // 统一法向量朝向
                if (-new_mean.dot(new_norm) < 0.0)
                    new_norm = -new_norm;

                // 把融合结果同时写给自己和邻居, 使二者参数完全一致
                plane->mean = new_mean;
                plane->norm = new_norm;
                plane->cov = new_cov;

                near_node->plane->mean = new_mean;
                near_node->plane->norm = new_norm;
                near_node->plane->cov = new_cov;
            }
        }
    }

    /**
     * 构造函数: 保存体素化参数, 清空地图和 LRU 链表
     */
    VoxelMap::VoxelMap(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, double _voxel_size, int _capacity) : max_point_thresh(_max_point_thresh), update_point_thresh(_update_point_thresh), plane_thresh(_plane_thresh), voxel_size(_voxel_size), capacity(_capacity)
    {
        featmap.clear();
        cache.clear();
    }

    /**
     * 世界坐标 -> 体素索引: floor(point / voxel_size)
     * 先 floor 再转 int64, 保证负坐标也落在正确的体素里(直接截断会出错)
     */
    VoxelKey VoxelMap::index(const Eigen::Vector3d &point)
    {
        Eigen::Vector3d idx = (point / voxel_size).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    /**
     * 初始化建图(用于首帧点云)
     *
     * 对每个点:
     *   - 体素不存在则新建, 压入 LRU 链表头部并记录 cache_it;
     *     若链表长度超过 capacity, 从尾部淘汰最久未使用的体素
     *   - 体素已存在则用 splice 把自己提到链表头部(标记为最近使用, O(1))
     *   - addPoint 只累积, 不做阈值判断
     * 所有点处理完后, 遍历整个 featmap 统一调用一次 updatePlane()
     */
    void VoxelMap::build(std::vector<PointWithCov> &pvs)
    {
        for (PointWithCov &pv : pvs)
        {
            VoxelKey k = index(pv.point);
            auto it = featmap.find(k);
            if (it == featmap.end())
            {
                // 新体素: 建对象 + 记录 LRU 位置
                featmap[k] = std::make_shared<VoxelGrid>(max_point_thresh, update_point_thresh, plane_thresh, k, this);
                cache.push_front(k);
                featmap[k]->cache_it = cache.begin();

                // 超出容量: 淘汰链表尾部(最久未使用)的体素
                if (cache.size() > capacity)
                {
                    featmap.erase(cache.back());
                    cache.pop_back();
                }
            }
            else
            {
                // 已存在: 提到 LRU 链表头部, 标记为最近使用
                cache.splice(cache.begin(), cache, featmap[k]->cache_it);
            }

            featmap[k]->addPoint(pv);
        }

        // 建图结束后统一拟合平面
        for (auto it = featmap.begin(); it != featmap.end(); it++)
        {
            it->second->updatePlane();
        }
    }

    /**
     * 在线增量更新(用于后续每帧点云)
     *
     * 前半段(查表 / 新建 / LRU 维护 / 容量淘汰)与 build() 完全一致,
     * 唯一区别是这里调用 pushPoint() 而非 addPoint():
     * 每个点都可能由体素自己触发 updatePlane() 或 merge(), 无需全表重算
     */
    void VoxelMap::update(std::vector<PointWithCov> &pvs)
    {

        for (PointWithCov &pv : pvs)
        {
            VoxelKey k = index(pv.point);
            auto it = featmap.find(k);
            if (it == featmap.end())
            {
                featmap[k] = std::make_shared<VoxelGrid>(max_point_thresh, update_point_thresh, plane_thresh, k, this);
                cache.push_front(k);
                featmap[k]->cache_it = cache.begin();
                if (cache.size() > capacity)
                {
                    featmap.erase(cache.back());
                    cache.pop_back();
                }
            }
            else
            {
                cache.splice(cache.begin(), cache, featmap[k]->cache_it);
            }
            // 增量插入: 由体素内部状态机决定是否重算平面或与邻居合并
            featmap[k]->pushPoint(pv);
        }
    }

    /**
     * 构建点-面残差(观测模型)
     *
     * 仅当体素已被判定为平面时才计算, 返回 data.is_valid:
     *   residual = norm · (point_world - plane_mean)     点到平面的有符号距离
     *   J_nq     = [p2m^T, -norm^T]                      残差对(法向量, 质心)的雅可比
     *   sigma_l  = J_nq * plane_cov * J_nq^T             平面参数自身的误差传导
     *            + norm^T * cov_world * norm             当前点定位误差在法向上的投影
     * 若 |residual| < 3 * sqrt(sigma_l) 则残差有效, 否则视为外点
     */
    bool VoxelMap::buildResidual(ResidualData &data, std::shared_ptr<VoxelGrid> voxel_grid)
    {
        data.is_valid = false;
        if (voxel_grid->is_plane)
        {
            // 点到平面质心的向量
            Eigen::Vector3d p2m = (data.point_world - voxel_grid->plane->mean);
            // 回填平面参数, 供 IESKF 使用
            data.plane_norm = voxel_grid->plane->norm;
            data.plane_mean = voxel_grid->plane->mean;
            // 残差 = 点到平面的有符号距离
            data.residual = data.plane_norm.dot(p2m);

            // 残差对(法向量, 质心)的雅可比
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p2m;
            J_nq.block<1, 3>(0, 3) = -data.plane_norm;
            // 平面参数不确定度传导到残差
            double sigma_l = J_nq * data.plane_cov * J_nq.transpose();
            // 叠加当前点自身的不确定度(取法向投影)
            sigma_l += data.plane_norm.transpose() * data.cov_world * data.plane_norm;
            // 3 sigma 卡方检验, 通过才认为残差可用
            if (std::abs(data.residual) < 3.0 * sqrt(sigma_l))
                data.is_valid = true;
        }
        return data.is_valid;
    }

} // namespace lio
