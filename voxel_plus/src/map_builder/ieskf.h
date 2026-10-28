#pragma once
#include <Eigen/Core>
#include <sophus/so3.hpp>

// 前端 voxel_plus 的滤波层: 误差状态迭代卡尔曼滤波(IESKF)
// 误差状态共 23 维, 分块顺序为
//   [pos(0) rot(3) rot_ext(6) pos_ext(9) vel(12) bg(15) ba(18) g(21, 2 维)]
// 其中 rot / rot_ext 是 SO3 流形量(只取 3 维最小扰动), 重力 g 是球面流形量(只有 2 个自由度),
// 于是 7 个 3 维块 + 1 个 2 维块 = 23
// 上游调用方是 lio_builder.cpp 的 LIOBuilder: 它持有本类的对象, 用 predict() 做 IMU 递推,
// 用 update() 做点-面残差的迭代更新, 观测函数则由 LIOBuilder::sharedUpdateFunc() 注册进来
// NOTE: 本文件使用了 std::function 却没有显式 include <functional>, 依赖其它头文件的间接包含
namespace kf
{
    // 重力加速度标量值(m/s^2); 重力向量 g 的模长始终被约束为它
    const double GRAVITY = 9.81;
    // 21 维 = 23 维误差状态去掉重力那 2 维
    // NOTE: Vector21d / Matrix21d / Matrix21x12d 这三个别名在本工程中没有任何地方使用
    using Vector21d = Eigen::Matrix<double, 21, 1>;
    // 点-面残差直接约束的状态维度: [pos, rot, rot_ext, pos_ext] 共 12 维
    using Vector12d = Eigen::Matrix<double, 12, 1>;
    using Matrix21d = Eigen::Matrix<double, 21, 21>;
    // IMU 噪声向量及其协方差 Q 的维度, 块顺序: [陀螺白噪声(0) 加计白噪声(3) 陀螺零偏游走(6) 加计零偏游走(9)]
    using Matrix12d = Eigen::Matrix<double, 12, 12>;
    using Matrix21x12d = Eigen::Matrix<double, 21, 12>;
    using Matrix23x12d = Eigen::Matrix<double, 23, 12>;

    // 误差状态协方差 P_ 与信息矩阵 H_ 的维度(23 x 23)
    using Matrix23d = Eigen::Matrix<double, 23, 23>;
    // 误差状态增量的维度, 即 State::operator- 的返回值(update 迭代中的 delta)
    using Vector23d = Eigen::Matrix<double, 23, 1>;
    // 递推时使用的增量维度: 重力那一段按完整 3 维旋转描述, 见 State::operator+=(const Vector24d &)
    using Vector24d = Eigen::Matrix<double, 24, 1>;
    // 重力切空间相关的小矩阵类型: 3x2 是切空间基/雅可比, 2x3 是反向映射
    using Matrix3x2d = Eigen::Matrix<double, 3, 2>;
    using Matrix2x3d = Eigen::Matrix<double, 2, 3>;

    // SO3 右雅可比 J_r(phi); Sophus 只提供左雅可比, 由 J_r(phi) = J_l(-phi) = J_l(phi)^T 求得
    Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &inp);

    /**
     * @brief 观测函数在一次迭代内累加出的"观测信息", 由 LIOBuilder::sharedUpdateFunc() 负责填充
     *
     * 该回调每轮迭代先清零, 再遍历所有通过卡方检验的点-面残差累加:
     *   H += J^T * r_info * J           12x12 观测信息矩阵
     *   b += J^T * r_info * residual    12x1  观测信息向量
     * 12 维就是状态块 [pos, rot, rot_ext, pos_ext], 与 P_ID / R_ID / ER_ID / EP_ID 对齐;
     * IESKF::update() 会把它们加到 23 维 H_ / b_ 的左上 12x12 块上, 其余维度只有先验信息
     */
    struct SharedState
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Matrix12d H;         // 观测信息矩阵, 只覆盖前 12 维状态
        Vector12d b;         // 观测信息向量 Sigma J^T R^-1 r
        size_t iter_num = 0; // 迭代轮次, 由 IESKF::update() 每轮自增
        // NOTE: iter_num 只写不读 —— 全工程(含 LIOBuilder::sharedUpdateFunc)没有任何地方读取它, 对结果无影响
    };

    /**
     * @brief 名义状态(nominal state): 只保存"大信号"本身, 误差量与其不确定度都在 IESKF::P_ 里
     *
     * 成员含义:
     *   pos      世界系下 IMU(机体)中心的位置
     *   rot      世界系 <- 机体系的旋转 R_wb
     *   rot_ext  机体系 <- 雷达系的固定旋转外参(工程里对应配置项 r_il)
     *   pos_ext  雷达原点在机体系下的位置(配置项 p_il)
     *   vel      世界系下的速度
     *   bg / ba  陀螺零偏 / 加计零偏(机体系)
     *   g        世界系下的重力向量, 模长恒为 GRAVITY, 其不确定度只有 2 个自由度
     * 世界系下的点坐标: p_w = rot * (rot_ext * p_l + pos_ext) + pos, 见 LIOBuilder::lidarToWorld()
     */
    struct State
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d rot_ext = Eigen::Matrix3d::Identity();
        Eigen::Vector3d pos_ext = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba = Eigen::Vector3d::Zero();
        Eigen::Vector3d g = Eigen::Vector3d(0.0, 0.0, -GRAVITY); // 默认竖直向下, 模长锁定为 GRAVITY

        // 由给定方向确定重力向量: 只取方向, 模长固定为 GRAVITY
        // IMU 初始化时用静止段平均加速度的反方向作为重力方向, 见 LIOBuilder::initializeImu()
        void initG(const Eigen::Vector3d &gravity_dir)
        {
            g = gravity_dir.normalized() * GRAVITY;
        }

        // 流形加法(盒加) State = State + delta, 23 维误差状态; 各块的叠加方式见 ieskf.cpp 中的定义
        void operator+=(const Vector23d &delta);

        // 24 维增量版本的流形加法, 重力段是一个完整的 3 维旋转向量;
        // 目前只在 IESKF::predict() 中调用, 且那里该段恒为 0(见 ieskf.cpp 中的 NOTE)
        void operator+=(const Vector24d &delta);

        // 流形减法(盒减) "this ⊖ other", 返回 23 维误差状态;
        // IESKF::update() 中的 delta = x_ - predict_x 即"当前迭代点相对预测均值"的误差
        Vector23d operator-(const State &other);

        // 重力球面流形在 g 处的切空间基 Bx(3x2): 两列正交归一且都与 g 正交,
        // 重力扰动写为旋转向量 u = Bx * θ, 再通过 g <- exp(u) * g 施加
        Matrix3x2d getBx() const;

        // 重力向量对 2 维切空间扰动的雅可比 ∂g/∂θ = -hat(g) * Bx (由 hat(u) * g = -hat(g) * u 得到)
        Matrix3x2d getMx() const;

        // 施加过一次 2 维增量 res 之后重新线性化的雅可比: bu = Bx * res 即那次的旋转向量,
        // 用 exp(bu) 与 bu 处的左雅可比(转置)对基础雅可比做修正; update() 组装 J / L 时使用
        Matrix3x2d getMx(const Eigen::Vector2d &res) const;

        // 把 3 维重力之差反解成 2 维切空间扰动的一阶关系: δθ ≈ Nx * δg, Nx = Bx^T * hat(g) / GRAVITY^2
        // predict() 中用于组装重力块自身的转移矩阵 F(21,21) = Nx * Mx
        Matrix2x3d getNx() const;
    };
    /**
     * @brief 一次 IMU 测量的输入(只用于 predict 递推)
     */
    struct Input
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d acc;  // 加计读数(机体系); 静止时约为 -R^T * g, LIOBuilder 中已归一到 9.81 量纲
        Eigen::Vector3d gyro; // 陀螺读数(机体系, rad/s)
        Input() = default;
        // NOTE: 下面两个构造函数的形参是非 const 引用, 无法绑定临时对象或 const 变量;
        //       现有代码用的是默认构造 + 逐字段赋值, 因此没有触发这个限制
        Input(Eigen::Vector3d &a, Eigen::Vector3d &g) : acc(a), gyro(g) {}
        Input(double a1, double a2, double a3, double g1, double g2, double g3) : acc(a1, a2, a3), gyro(g1, g2, g3) {}
    };

    // 观测函数类型: 入参为当前迭代的名义状态与该轮要填充的观测信息;
    // 由 LIOBuilder::loadConfig() 以 lambda 形式注册(内部转调 sharedUpdateFunc), update() 每轮迭代回调一次
    using measure_func = std::function<void(State &, SharedState &)>;

    /**
     * @brief 迭代误差状态卡尔曼滤波: 名义状态在流形上迭代, 误差状态只在每次线性化的瞬间局部存在
     *
     * 与"线性化一次、求解一次"的普通 EKF 不同, update() 会在预测均值附近反复线性化观测方程:
     * 每轮都用当前名义状态重建残差与雅可比, 直到误差状态增量收敛(或达到 max_iter_), 从而减小线性化误差。
     *
     * 两个阶段:
     *   predict(): 用一段 IMU 数据递推名义状态, 并按误差状态模型传播协方差 P_ = F P F^T + G Q G^T
     *   update() : 用一帧雷达点-面残差做迭代观测更新, 结束后同时得到后验名义状态与后验 P_
     * 调用关系: LIOBuilder 持有本类对象, undistortCloud() 中逐个 IMU 区间调用 predict(),
     *           process() 中每帧点云调用一次 update()
     */
    class IESKF
    {
    public:
        // 误差状态各块的起始下标: 位置 0 / 姿态 3 / 外参旋转 6 / 外参平移 9 / 速度 12 /
        // 陀螺零偏 15 / 加计零偏 18 / 重力(2 维) 21
        // 外部(如 LIOBuilder 传播点协方差)靠它们索引 kf_.P().block<3, 3>(kf::IESKF::R_ID, ...)
        static int P_ID, R_ID, ER_ID, EP_ID, V_ID, BG_ID, BA_ID, G_ID;
        // 默认构造: max_iter_ / eps_ / x_ 取类内初始值
        // NOTE: P_ 与 func_ 没有初始值 —— P_ 在使用前必须由外部赋值(LIOBuilder::initializeImu 里 setIdentity),
        //       否则内容未定义; func_ 未注册就调用 update() 会抛 std::bad_function_call
        IESKF();

        // 直接指定最大迭代次数的构造; NOTE: 该构造同样不初始化 P_, 需外部赋值后才能 predict
        IESKF(size_t max_iter) : max_iter_(max_iter) {}

        // NOTE: setMaxIter 在全工程中未被调用 —— 配置项 opti_max_iter 只读进了 LIOConfig 却没有接进来,
        //       max_iter_ 实际恒为默认值 5
        void setMaxIter(int max_iter) { max_iter_ = max_iter; }

        State &x() { return x_; }

        // NOTE: change_x 与下面的 change_P 在全工程中均未被调用
        void change_x(State &x) { x_ = x; }

        Matrix23d &P() { return P_; }

        // 注册观测函数(见 measure_func), 由 LIOBuilder::loadConfig() 调用
        void set_share_function(measure_func func) { func_ = func; }

        void change_P(Matrix23d &P) { P_ = P; }

        // IMU 递推一步: 名义状态在 dt 上积分, 误差状态协方差随之传播到同一时刻
        void predict(const Input &inp, double dt, const Matrix12d &Q);

        // 用注册的观测函数迭代求解, 得到后验名义状态与后验协方差
        void update();

    private:
        // 迭代更新的最大轮数(update 中 for 循环的上限)
        size_t max_iter_ = 5;
        // 迭代收敛阈值: 本轮的误差状态增量满足判据就提前退出(判据的写法见 ieskf.cpp)
        double eps_ = 0.001;
        // 名义状态(流形量), 滤波真正维护的对象
        State x_;
        // 23x23 误差状态协方差, 分块顺序同 P_ID ~ G_ID
        Matrix23d P_;
        // 观测函数, 由 set_share_function() 注册, update() 每轮迭代回调一次
        measure_func func_;
        // 迭代中临时使用的后验信息矩阵与信息向量(H_ = 先验信息 + 观测信息)
        Matrix23d H_;
        Vector23d b_;
        // 误差状态转移矩阵 F_(23x23), predict() 中按 IMU 误差模型填充
        Matrix23d F_;
        // 噪声驱动矩阵 G_(23x12): 把 12 维 IMU 噪声映射到误差状态上
        Matrix23x12d G_;
    };
} // namespace kf