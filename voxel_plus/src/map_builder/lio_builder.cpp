#include "lio_builder.h"

namespace lio
{
    /**
     * 加载配置并完成一次性初始化(由 lio_node.cpp 在构造时调用一次)
     *
     * 步骤:
     *   1. 保存配置副本, 状态机复位到 IMU_INIT
     *   2. 按 ng/na/nbg/nba 填过程噪声 Q 的四个 3x3 对角块
     *      (块序与 ieskf.cpp 中 G_ 噪声矩阵的列块一致: [陀螺测量噪声, 加速度计测量噪声, 陀螺零偏游走, 加速度计零偏游走])
     *   3. scan_resolution > 0 时设置降采样器叶节点尺寸(输入点云每帧在 process() 中绑定)
     *   4. 新建 VoxelMap, 参数顺序为 (体素点数上限, 周期重算阈值, 平面判定阈值, 体素边长, 容量)
     *   5. 用配置覆盖 VoxelGrid 的两个平面合并阈值(静态成员, 对所有体素生效)
     *   6. 复位当前帧点云, 注册 IESKF 的观测回调 lambda, 预分配残差缓存
     *
     * 调用关系: 由前端入口 lio_node.cpp 调用; 内部创建 VoxelMap 供 process()/sharedUpdateFunc() 使用
     */
    void LIOBuilder::loadConfig(LIOConfig &_config)
    {
        config_ = _config;                   // 整体拷贝, 之后各函数都从 config_ 读参数
        status_ = LIOStatus::IMU_INIT;       // 从头开始: 先做 IMU 初始化

        // 过程噪声 Q 的四个对角块, 顺序与 ieskf.cpp 中 G_ 的列块对应
        data_group_.Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * config_.ng;  // 陀螺测量噪声
        data_group_.Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * config_.na;  // 加速度计测量噪声
        data_group_.Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * config_.nbg; // 陀螺零偏随机游走
        data_group_.Q.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * config_.nba; // 加速度计零偏随机游走
        if (config_.scan_resolution > 0.0)
            scan_filter_.setLeafSize(config_.scan_resolution, config_.scan_resolution, config_.scan_resolution); // 只设叶节点尺寸

        // 建立体素平面地图: (点数上限, 周期重算点数阈值, 平面判定阈值, 体素边长, 体素容量)
        map_ = std::make_shared<VoxelMap>(config_.max_point_thresh, config_.update_size_thresh, config_.plane_thresh, config_.voxel_size, config_.map_capacity);

        VoxelGrid::merge_thresh_for_angle_ = config_.merge_thresh_for_angle;       // 平面合并夹角阈值
        VoxelGrid::merge_thresh_for_distance_ = config_.merge_thresh_for_distance; // 平面合并间距阈值

        lidar_cloud_.reset(new pcl::PointCloud<pcl::PointXYZINormal>); // 当前帧去畸变+降采样后的点云
        /** 观测回调 lambda: 捕获 this, 把 IESKF 传入的状态 s 与共享量 d 原样转发给 sharedUpdateFunc() */
        kf_.set_share_function(
            [this](kf::State &s, kf::SharedState &d)
            { sharedUpdateFunc(s, d); });
        data_group_.residual_info.resize(10000); // 逐点残差缓存固定容量
        // NOTE: 容量固定为 10000, 而 process()/sharedUpdateFunc() 都按 lidar_cloud_->size() 做下标访问,
        //       降采样后点数超过 10000 时会越界写入(这里没有做容量检查)
    }

    /**
     * IMU 初始化(状态机 IMU_INIT 阶段, 由 process() 每帧调用直到返回 true)
     *
     * 流程:
     *   1. 把本帧 IMU 追加进 imu_cache, 不足 config_.imu_init_num 帧直接返回 false
     *   2. 求静止段平均加速度与平均角速度
     *   3. 外参初值取配置的 r_il/p_il; 陀螺零偏初值取平均角速度; 位置/速度保持零(视作从静止开始)
     *   4. gravity_align 时: 用平均加速度构造初始姿态, 使体坐标系下的静止加速度方向对齐世界 +Z,
     *      并令重力向量 g = (0,0,-9.81); 否则直接把 g 取为 -acc_mean
     *   5. 初始化协方差 P: 先整体置单位阵, 再把外参/零偏/重力方向相关的块调小(初值置信度高)
     *   6. 记录本帧最后一个 IMU, 作为后续预积分的起点
     *
     * 调用关系: 只被 process() 在 IMU_INIT 状态下调用; 成功后 process() 会把状态切到 MAP_INIT
     */
    bool LIOBuilder::initializeImu(std::vector<IMUData> &imus)
    {
        data_group_.imu_cache.insert(data_group_.imu_cache.end(), imus.begin(), imus.end()); // 累积 IMU(初始化前不清空)
        if (data_group_.imu_cache.size() < config_.imu_init_num) // 帧数不够: 本次不做初始化
            return false;
        Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
        for (const auto &imu : data_group_.imu_cache)
        {
            acc_mean += imu.acc; // 求和, 循环结束后除以帧数
            gyro_mean += imu.gyro;
        }
        acc_mean /= static_cast<double>(data_group_.imu_cache.size());  // 平均加速度(静止段 ≈ 比力)
        gyro_mean /= static_cast<double>(data_group_.imu_cache.size()); // 平均角速度(静止段 ≈ 陀螺零偏)
        data_group_.gravity_norm = acc_mean.norm(); // 记录模长, 后续把加速度归一到 9.81
        kf_.x().rot_ext = config_.r_il; // 外参初值: R_IL
        kf_.x().pos_ext = config_.p_il; // 外参初值: P_IL
        kf_.x().bg = gyro_mean;        // 陀螺零偏初值取静止段平均角速度
        if (config_.gravity_align)
        {
            // 构造把 (-acc_mean) 方向转到世界 -Z 轴的姿态, 等价于把 acc_mean 转到 +Z;
            // 与速度传播式 v_dot = R*(a - ba) + g (g = (0,0,-9.81)) 配套: 静止时两项恰好抵消
            kf_.x().rot = (Eigen::Quaterniond::FromTwoVectors((-acc_mean).normalized(), Eigen::Vector3d(0.0, 0.0, -1.0)).matrix());
            kf_.x().initG(Eigen::Vector3d(0, 0, -1.0)); // 重力方向固定为世界 -Z
        }
        else
        {
            kf_.x().initG(-acc_mean); // 不做重力对齐: 直接用 -acc_mean 作为重力向量(模长被归一到 GRAVITY)
        }
        kf_.P().setIdentity(); // 位置/姿态/速度等块先置单位阵
        kf_.P().block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 0.00001;   // 外参旋转(ER_ID)
        kf_.P().block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 0.00001;   // 外参平移(EP_ID)
        kf_.P().block<3, 3>(15, 15) = Eigen::Matrix3d::Identity() * 0.0001;  // 陀螺零偏(BG_ID)
        kf_.P().block<3, 3>(18, 18) = Eigen::Matrix3d::Identity() * 0.0001;  // 加速度计零偏(BA_ID)
        kf_.P().block<2, 2>(21, 21) = Eigen::Matrix2d::Identity() * 0.00001; // 重力方向(2 维, G_ID)
        data_group_.last_imu = imus.back(); // 记录最后一个 IMU(下一帧预积分的起点)
        return true;
    }

    /**
     * 点云去畸变: 用 IMU 预积分把每个点补偿到扫描结束时刻的雷达系(由 process() 每帧调用)
     *
     * 输入: package.cloud(curvature 字段存放每点相对扫描起点的毫秒偏移)与 package.imus
     * 输出: 就地覆盖 package.cloud 的坐标(仍在雷达系, 对应扫描结束时刻的位姿)
     *
     * 流程:
     *   1. imu_cache = [上一帧最后一个 IMU] + 本帧 package.imus, 保证积分区间连续
     *   2. 点云按 curvature 升序排序(后面从尾部倒序处理)
     *   3. imu_poses_cache 首元素写当前状态(offset 记 0)
     *   4. 逐对 IMU 取中点加速度/角速度, 归一化到标准重力后调 kf_.predict() 推进状态,
     *      并把推进后的位姿按 offset = 该 IMU 时间 - 扫描起始时间 存入 imu_poses_cache
     *      (跳过 tail 早于上一帧积分终点的 IMU 对; 首对的 dt 从上一帧点云结束时刻起算)
     *   5. 再补积到扫描结束时刻(dt = cloud_end - 最后一个 IMU 时间)
     *   6. 从后往前两两取 imu_poses_cache 的位姿, 对落在该时间段内的点做补偿:
     *        p_world(t) = R_head(dt) * (R_IL * p + P_IL) + p_head(dt)
     *          (旋转按角速度指数映射外推, 平移按匀速 + 匀加速外推)
     *        p_out = R_IL^T * (R_wi^T * (p_world(t) - p_wi) - P_IL)
     *          (再转回扫描结束时刻的雷达系)
     *
     * 注意: 本函数同时承担 IESKF 的预测步骤, 每帧调用后 kf_ 的状态与协方差都被推进到扫描结束时刻
     * 调用关系: 只被 process() 调用; 内部调用 kf_.predict()(ieskf.cpp)与 Sophus 的 SO3 指数映射
     */
    void LIOBuilder::undistortCloud(SyncPackage &package)
    {
        data_group_.imu_cache.clear(); // 复用缓存: 只保留本帧需要的 IMU
        data_group_.imu_cache.push_back(data_group_.last_imu); // 补上上一帧最后一个 IMU, 保证区间连续
        data_group_.imu_cache.insert(data_group_.imu_cache.end(), package.imus.begin(), package.imus.end());

        const double imu_time_begin = data_group_.imu_cache.front().timestamp; // NOTE: 定义了但后续未使用
        const double imu_time_end = data_group_.imu_cache.back().timestamp;    // 本帧 IMU 数据的结束时刻
        const double cloud_time_begin = package.cloud_start_time;             // 扫描起始时刻(offset 的时间基准)
        const double cloud_time_end = package.cloud_end_time;                 // 扫描结束时刻(去畸变的目标时刻)

        /** 排序 lambda: 按 curvature 字段(每点相对扫描起点的毫秒数)升序排列点云 */
        std::sort(package.cloud->points.begin(), package.cloud->points.end(), [](pcl::PointXYZINormal &p1, pcl::PointXYZINormal &p2) -> bool
                  { return p1.curvature < p2.curvature; });

        // 位姿缓存第一个元素: 当前状态(offset 固定写 0, 表示扫描起始时刻的参考位姿)
        // NOTE: 此处当前状态实际还停在上一帧点云结束时刻; 且首次调用(首帧建图)时
        //       last_acc/last_gyro 仍是结构体默认的零值(它们只在本函数循环内被赋值)
        data_group_.imu_poses_cache.clear();
        data_group_.imu_poses_cache.emplace_back(0.0, data_group_.last_acc, data_group_.last_gyro,
                                                kf_.x().vel, kf_.x().pos, kf_.x().rot);

        Eigen::Vector3d acc_val, gyro_val; // 相邻两帧 IMU 的均值(中点积分)
        double dt = 0.0;                   // 积分步长(秒)
        kf::Input inp;                     // 传给 kf_.predict() 的输入(加速度/角速度)

        for (auto it_imu = data_group_.imu_cache.begin(); it_imu < (data_group_.imu_cache.end() - 1); it_imu++)
        {
            IMUData &head = *it_imu;
            IMUData &tail = *(it_imu + 1);

            if (tail.timestamp < data_group_.last_cloud_end_time) // 落在上一帧已积分过的区间内, 跳过避免重复积分
                continue;
            gyro_val = 0.5 * (head.gyro + tail.gyro); // 角速度取相邻两帧均值
            acc_val = 0.5 * (head.acc + tail.acc);    // 加速度取相邻两帧均值

            acc_val = acc_val * 9.81 / data_group_.gravity_norm; // 把加速度归一到标准重力 9.81(校正 IMU 刻度)

            if (head.timestamp < data_group_.last_cloud_end_time) // 首对的起始时刻被截断到上一帧结束时刻
                dt = tail.timestamp - data_group_.last_cloud_end_time;
            else
                dt = tail.timestamp - head.timestamp;

            inp.acc = acc_val;
            inp.gyro = gyro_val;

            kf_.predict(inp, dt, data_group_.Q); // 推进名义状态并传播协方差 P = F*P*F^T + G*Q*G^T

            // 记录本段末尾的速度量, 供去畸变的短时外推使用:
            //   last_gyro: 机体系角速度(扣陀螺零偏)
            //   last_acc : 世界系加速度(扣加计零偏再加已知重力)
            data_group_.last_gyro = gyro_val - kf_.x().bg;
            data_group_.last_acc = kf_.x().rot * (acc_val - kf_.x().ba) + kf_.x().g;

            double offset = tail.timestamp - cloud_time_begin; // 该 IMU 时刻相对扫描起点的偏移(秒)
            data_group_.imu_poses_cache.emplace_back(offset, data_group_.last_acc, data_group_.last_gyro, kf_.x().vel, kf_.x().pos, kf_.x().rot); // 缓存位姿
        }

        dt = cloud_time_end - imu_time_end; // 从最后一个 IMU 补积到扫描结束时刻
        // NOTE: inp 复用循环内最后一次赋值的输入; 若上面的循环一次都没执行
        //       (例如 imu_cache 只剩 last_imu), inp 是未初始化的, 会被直接使用
        kf_.predict(inp, dt, data_group_.Q);

        data_group_.last_imu = package.imus.back();       // 记录最后一个 IMU 样本
        // NOTE: 上面假设 package.imus 非空(由 lio_node.cpp 的同步逻辑保证), 为空时 back() 是未定义行为
        data_group_.last_cloud_end_time = cloud_time_end; // 下一帧从这里继续积分

        Eigen::Matrix3d cur_rot = kf_.x().rot;         // 扫描结束时刻的姿态 R_wi
        Eigen::Vector3d cur_pos = kf_.x().pos;         // 扫描结束时刻的位置 p_wi
        Eigen::Matrix3d cur_rot_ext = kf_.x().rot_ext; // 外参 R_IL
        Eigen::Vector3d cur_pos_ext = kf_.x().pos_ext; // 外参 P_IL

        auto it_pcl = package.cloud->points.end() - 1; // 从时间最晚的点开始倒序遍历(点云已按时间升序排好)
        for (auto it_kp = data_group_.imu_poses_cache.end() - 1; it_kp != data_group_.imu_poses_cache.begin(); it_kp--)
        {
            // head/tail 是相邻两个缓存位姿: head 作为区间起点的积分基准, tail 提供区间末端的加速度/角速度
            auto head = it_kp - 1;
            auto tail = it_kp;

            Eigen::Matrix3d imu_rot = head->rot;   // 区间起点姿态
            Eigen::Vector3d imu_pos = head->pos;   // 区间起点位置
            Eigen::Vector3d imu_vel = head->vel;   // 区间起点速度
            Eigen::Vector3d imu_acc = tail->acc;   // 区间末端的世界系加速度估计
            Eigen::Vector3d imu_gyro = tail->gyro; // 区间末端的机体系角速度估计

            for (; it_pcl->curvature / double(1000) > head->offset; it_pcl--) // 处理时间落在本区间内的点(毫秒转秒)
            {
                dt = it_pcl->curvature / double(1000) - head->offset; // dt 被复用为"点时刻相对区间起点"的偏移
                Eigen::Vector3d point(it_pcl->x, it_pcl->y, it_pcl->z); // 点在雷达系下的坐标
                Eigen::Matrix3d point_rot = imu_rot * Sophus::SO3d::exp(imu_gyro * dt).matrix();  // 点时刻姿态: 由 head 按角速度外推
                Eigen::Vector3d point_pos = imu_pos + imu_vel * dt + 0.5 * imu_acc * dt * dt;     // 点时刻位置: 匀速 + 匀加速外推
                // 先经外参转到 IMU 系并用点时刻位姿得到世界坐标, 再转到扫描末时刻的 IMU 系, 最后经外参转回雷达系
                Eigen::Vector3d p_compensate = cur_rot_ext.transpose() * (cur_rot.transpose() * (point_rot * (cur_rot_ext * point + cur_pos_ext) + point_pos - cur_pos) - cur_pos_ext);
                it_pcl->x = p_compensate(0);
                it_pcl->y = p_compensate(1);
                it_pcl->z = p_compensate(2);

                // NOTE: 只在 begin() 处 break, 若外层循环继续走到更早的 head 而该点时间仍大于
                //       head->offset, 时间戳最小的那个点会被再次补偿(此时读到的是上次已覆盖的坐标)
                if (it_pcl == package.cloud->points.begin())
                    break;
            }
        }
    }

    /**
     * 雷达系点云 -> 世界系(由 process() 与 lio_node.cpp 的 mainCB 调用)
     * 变换为 T_wl = [R_wi * R_IL | R_wi * P_IL + p_wi], 即 T_wl = T_wi * T_il
     * 用的是当前 kf_.x() 的状态估计, 未做任何时间插值
     */
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr LIOBuilder::lidarToWorld(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZINormal>);
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity(); // 单精度 4x4, 供 pcl::transformPointCloud 使用
        transform.block<3, 3>(0, 0) = (kf_.x().rot * kf_.x().rot_ext).cast<float>();              // R_wl = R_wi * R_IL
        transform.block<3, 1>(0, 3) = (kf_.x().rot * kf_.x().pos_ext + kf_.x().pos).cast<float>(); // p_wl = R_wi * P_IL + p_wi
        pcl::transformPointCloud(*cloud, *cloud_world, transform); // 只变换点坐标字段
        return cloud_world;
    }

    /**
     * 雷达系点云 -> 机体系(IMU 系)(由 lio_node.cpp 的 mainCB 调用, 用于发布 body_cloud)
     * 只施加外参 T_il: p_body = R_IL * p_l + P_IL, 与状态估计无关
     */
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr LIOBuilder::lidarToBody(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud)
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_body(new pcl::PointCloud<pcl::PointXYZINormal>);
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity(); // 单精度 4x4, 供 pcl::transformPointCloud 使用
        transform.block<3, 3>(0, 0) = kf_.x().rot_ext.cast<float>(); // R_IL
        transform.block<3, 1>(0, 3) = kf_.x().pos_ext.cast<float>(); // P_IL
        pcl::transformPointCloud(*cloud, *cloud_body, transform);   // 只变换点坐标字段
        return cloud_body;
    }

    /**
     * 每帧主入口(由 lio_node.cpp 的定时器回调 mainCB 调用)
     *
     * 状态机三个分支:
     *   IMU_INIT   : 调 initializeImu() 攒 IMU, 成功后切到 MAP_INIT, 并记下本帧点云结束时刻作为积分起点
     *                (本帧点云被丢弃, 不参与建图)
     *   MAP_INIT   : 首帧建图: 去畸变 -> 转世界系 -> 逐点算协方差 -> map_->build(); 之后切到 LIO_MAPPING
     *                注意本帧不做 IESKF 更新
     *   LIO_MAPPING: 正常帧: 去畸变 -> 体素降采样 -> 预存每点雷达系坐标与噪声协方差 -> kf_.update()
     *                -> 用更新后的位姿把点云转世界系并算协方差 -> map_->update() 增量插入地图
     */
    void LIOBuilder::process(SyncPackage &package)
    {
        if (status_ == LIOStatus::IMU_INIT)
        {
            // IMU 初始化阶段: 每来一帧就试着初始化, 成功前不处理点云
            if (initializeImu(package.imus))
            {
                status_ = LIOStatus::MAP_INIT;                            // 下一帧用来建初始地图
                data_group_.last_cloud_end_time = package.cloud_end_time; // 预积分起点: 本帧(未用)点云的结束时刻
            }
        }
        else if (status_ == LIOStatus::MAP_INIT)
        {
            // 首帧建图: 只建地图, 不做滤波更新
            undistortCloud(package); // 去畸变, 同时把 IESKF 状态预测到扫描结束时刻
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr point_world = lidarToWorld(package.cloud); // 雷达系 -> 世界系
            std::vector<PointWithCov> pv_list;
            Eigen::Matrix3d r_wl = kf_.x().rot * kf_.x().rot_ext;               // R_wl = R_wi * R_IL
            Eigen::Vector3d p_wl = kf_.x().rot * kf_.x().pos_ext + kf_.x().pos;  // NOTE: 本分支未使用该变量(冗余)
            for (size_t i = 0; i < point_world->size(); i++)
            {
                PointWithCov pv;
                pv.point = Eigen::Vector3d(point_world->points[i].x, point_world->points[i].y, point_world->points[i].z); // 世界系坐标
                // 注意: 这里取的是 package.cloud(已被去畸变)的雷达系坐标, 变量名 body 只是沿用习惯叫法
                Eigen::Vector3d point_body(package.cloud->points[i].x, package.cloud->points[i].y, package.cloud->points[i].z);
                Eigen::Matrix3d point_cov;
                calcBodyCov(point_body, config_.ranging_cov, config_.angle_cov, point_cov); // 点的量测噪声协方差(雷达系)
                Eigen::Matrix3d point_crossmat = Sophus::SO3d::hat(point_body);           // 旋转不确定度传到点位的杠杆臂
                // NOTE: 杠杆臂直接用雷达系点坐标; 严格推导应为 hat(R_IL * p_l + P_IL)(见 sharedUpdateFunc 中 J 的姿态块),
                //       外参接近单位阵/零向量时差异很小

                // 点在世界系下的协方差 = 量测噪声经 R_wl 旋转 + 姿态不确定度(经 hat(p)) + 位置不确定度
                point_cov = r_wl * point_cov * r_wl.transpose() +
                            point_crossmat * kf_.P().block<3, 3>(kf::IESKF::R_ID, kf::IESKF::R_ID) * point_crossmat.transpose() +
                            kf_.P().block<3, 3>(kf::IESKF::P_ID, kf::IESKF::P_ID);
                pv.cov = point_cov;
                pv_list.push_back(pv);
            }

            map_->build(pv_list); // 逐点累积进体素, 最后对全表统一拟合一次平面(VoxelMap::build)

            status_ = LIOStatus::LIO_MAPPING; // 首帧建图完成, 后续进入正常定位建图
        }
        else
        {
            // 正常帧: 去畸变 -> 降采样 -> IESKF 更新 -> 转世界系 -> 增量更新地图
            undistortCloud(package);
            if (config_.scan_resolution > 0.0)
            {
                scan_filter_.setInputCloud(package.cloud); // 每帧重新绑定输入点云
                scan_filter_.filter(*lidar_cloud_);         // 体素降采样, 结果作为本帧参与滤波的点集
            }
            else
            {
                pcl::copyPointCloud(*package.cloud, *lidar_cloud_); // 不降采样, 直接拷贝
            }
            int size = lidar_cloud_->size();
            for (int i = 0; i < size; i++)
            {
                // 预先算好每点的雷达系坐标与量测噪声协方差, 供 kf_.update() 内部的回调反复使用
                data_group_.residual_info[i].point_lidar = Eigen::Vector3d(lidar_cloud_->points[i].x, lidar_cloud_->points[i].y, lidar_cloud_->points[i].z);
                calcBodyCov(data_group_.residual_info[i].point_lidar, config_.ranging_cov, config_.angle_cov, data_group_.residual_info[i].cov_lidar);
            }
            kf_.update(); // IESKF 迭代更新: 每次迭代回调 sharedUpdateFunc() 重建残差与 H/b
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr point_world = lidarToWorld(lidar_cloud_); // 用更新后的位姿转世界系
            std::vector<PointWithCov> pv_list;
            Eigen::Matrix3d r_wl = kf_.x().rot * kf_.x().rot_ext;               // 更新后的 R_wl
            Eigen::Vector3d p_wl = kf_.x().rot * kf_.x().pos_ext + kf_.x().pos;  // NOTE: 本分支未使用该变量(冗余)
            for (int i = 0; i < size; i++)
            {
                PointWithCov pv;
                pv.point = Eigen::Vector3d(point_world->points[i].x, point_world->points[i].y, point_world->points[i].z); // 世界系坐标
                Eigen::Matrix3d cov = data_group_.residual_info[i].cov_lidar; // 复用前面算好的雷达系量测噪声
                Eigen::Matrix3d point_crossmat = Sophus::SO3d::hat(data_group_.residual_info[i].point_lidar);
                // 世界系下点的协方差, 注意这里用的是 kf_.update() 之后的 P(后验)
                pv.cov = r_wl * cov * r_wl.transpose() +
                         point_crossmat * kf_.P().block<3, 3>(kf::IESKF::R_ID, kf::IESKF::R_ID) * point_crossmat.transpose() +
                         kf_.P().block<3, 3>(kf::IESKF::P_ID, kf::IESKF::P_ID);
                pv_list.push_back(pv);
            }
            map_->update(pv_list); // 增量插入地图, 由 VoxelGrid 自行决定重算平面/合并
        }
    }

    /**
     * 点-面残差的观测方程回调(由 IESKF::update() 的每次迭代调用)
     * 注册方式: loadConfig() 中 kf_.set_share_function(lambda -> sharedUpdateFunc)
     *
     * 参数:
     *   state        当前迭代的状态估计(位姿/速度/零偏/重力)
     *   shared_state 输出: H(12x12)与 b(12x1)累加器, 每次调用先清零
     *
     * 第一部分(每点独立, 可用 OpenMP 并行):
     *   - 用当前迭代状态把点从雷达系转到世界系: p_w = R_wl * p_l + p_wl
     *   - 计算点在世界系下的协方差 cov_world(量测噪声 + 状态不确定度)
     *   - 找到点所在体素, 命中则调 VoxelMap::buildResidual() 填残差与有效性标记
     * 第二部分(串行累加):
     *   - 只对有效点组装观测雅可比 J(1x12)与信息权重 r_info
     *   - shared_state.H += J^T * r_info * J, shared_state.b += J^T * r_info * residual
     *     (IESKF::update() 再把它们加到先验项上解增量方程)
     * J 的四个 3 列块顺序与 IESKF 状态一致: 位置(0-2)/姿态(3-5)/外参旋转(6-8)/外参平移(9-11),
     * 对应 kf::IESKF::P_ID/R_ID/ER_ID/EP_ID
     *
     * 调用关系: 只被 kf_.update() 调用; 内部调用 VoxelMap::index() 与 VoxelMap::buildResidual()
     */
    void LIOBuilder::sharedUpdateFunc(kf::State &state, kf::SharedState &shared_state)
    {
        Eigen::Matrix3d r_wl = state.rot * state.rot_ext;               // 当前迭代下的 R_wl
        Eigen::Vector3d p_wl = state.rot * state.pos_ext + state.pos;   // 当前迭代下的 p_wl
        int size = lidar_cloud_->size();

#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM); // MP_PROC_NUM 在 CMakeLists.txt 里被定义为 1
#pragma omp parallel for
#endif
        for (int i = 0; i < size; i++)
        {
            data_group_.residual_info[i].point_world = r_wl * data_group_.residual_info[i].point_lidar + p_wl; // 世界系坐标

            // 点在世界系下的协方差: 量测噪声旋转到世界系 + 姿态不确定度 + 位置不确定度
            // (用预测协方差 kf_.P(), 与迭代中的状态估计配套)
            Eigen::Matrix3d point_crossmat = Sophus::SO3d::hat(data_group_.residual_info[i].point_lidar);
            data_group_.residual_info[i].cov_world = r_wl * data_group_.residual_info[i].cov_lidar * r_wl.transpose() +
                                                    point_crossmat * kf_.P().block<3, 3>(kf::IESKF::R_ID, kf::IESKF::R_ID) * point_crossmat.transpose() +
                                                    kf_.P().block<3, 3>(kf::IESKF::P_ID, kf::IESKF::P_ID);
            VoxelKey position = map_->index(data_group_.residual_info[i].point_world); // 查点所在的体素
            auto iter = map_->featmap_.find(position);
            if (iter != map_->featmap_.end())
            {
                map_->buildResidual(data_group_.residual_info[i], iter->second); // 命中: 计算点到面残差并回填平面参数
            }
            // NOTE: 未命中体素时不会调用 buildResidual, 该下标的 is_valid 保留上一次调用留下的取值;
            //       下面按 is_valid 筛选时可能用到过期的结果, 请核对是否为本意
        }

        shared_state.H.setZero();
        shared_state.b.setZero();
        Eigen::Matrix<double, 1, 12> J; // 残差对状态(前 12 维)的雅可比
        Eigen::Matrix<double, 1, 6> Jn;

        int effect_num = 0; // 本帧有效残差数量(统计用)

        for (int i = 0; i < size; i++)
        {
            if (!data_group_.residual_info[i].is_valid) // 只累加通过卡方检验的残差
                continue;
            effect_num++;
            J.setZero(); // 外参不参与估计时, J 的 6-11 块保持为 0

            Eigen::Vector3d plane_norm = data_group_.residual_info[i].plane_norm;

            // 与 VoxelMap::buildResidual() 内部同构: 残差对(法向量, 质心)的雅可比
            Jn.block<1, 3>(0, 0) = (data_group_.residual_info[i].point_world - data_group_.residual_info[i].plane_mean).transpose();
            Jn.block<1, 3>(0, 3) = -plane_norm.transpose();
            double r_cov = Jn * data_group_.residual_info[i].plane_cov * Jn.transpose(); // 平面参数的不确定度
            r_cov += plane_norm.transpose() * r_wl * data_group_.residual_info[i].cov_lidar * r_wl.transpose() * plane_norm;
            // 上面只叠加点自身的量测噪声(不用 cov_world), 避免和先验协方差重复计入状态不确定度

            double r_info = r_cov < 0.0002 ? 5000 : 1.0 / r_cov; // 权重取 1/r_cov, 并做上限截断(最小方差 0.0002)
            J.block<1, 3>(0, 0) = plane_norm.transpose(); // 对位置: d(residual)/d(p_wi) = n^T
            // 对姿态: d(R_wi * p_body)/d(delta_theta) = -R_wi * hat(R_IL * p_l + P_IL), 再投影到法向
            J.block<1, 3>(0, 3) = -plane_norm.transpose() * state.rot * Sophus::SO3d::hat(state.rot_ext * data_group_.residual_info[i].point_lidar + state.pos_ext);
            if (config_.estimate_ext)
            {
                // 外参也作为状态量时才填这两块(6-8: 外参旋转, 9-11: 外参平移)
                J.block<1, 3>(0, 6) = -plane_norm.transpose() * r_wl * Sophus::SO3d::hat(data_group_.residual_info[i].point_lidar);
                J.block<1, 3>(0, 9) = plane_norm.transpose() * state.rot;
            }
            shared_state.H += J.transpose() * r_info * J;                                    // 信息矩阵累加
            shared_state.b += J.transpose() * r_info * data_group_.residual_info[i].residual; // 信息向量累加
        }
        if (effect_num < 1)
            std::cout << "NO EFFECTIVE POINT"; // NOTE: 该输出没有换行符, 会与后续打印挤在同一行
        // std::cout << "==================: " << effect_num << std::endl;
    }

}