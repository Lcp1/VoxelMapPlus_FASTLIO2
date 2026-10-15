#include "ieskf.h"

// 误差状态迭代卡尔曼滤波的实现, 状态的定义与分块顺序见 ieskf.h
// NOTE: 本文件用到 std::atan2 / std::fabs 却没有显式 include <cmath>, 依赖其它头文件的间接包含
namespace kf
{

    /**
     * SO3 右雅可比 J_r(phi) = J_l(-phi) = J_l(phi)^T
     * 误差姿态按右扰动约定 R = R_hat * exp(delta) 定义, 所以姿态相关的雅可比都是右雅可比;
     * Sophus 只提供左雅可比, 这里用转置关系换算
     * 用途: predict() 里角度积分对零偏/噪声的雅可比, 以及 update() 里组装 J / L 的姿态块
     */
    Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &inp)
    {
        return Sophus::SO3d::leftJacobian(inp).transpose();
    }

    /**
     * 流形加法(盒加) x += delta: 把 23 维误差状态叠加到名义状态上
     *   欧氏量 pos / pos_ext / vel / bg / ba: 直接相加
     *   旋转量 rot / rot_ext: 右乘 exp(3 维扰动), 保证结果仍是正交矩阵
     *   重力 g: 先用切空间基 Bx 把 2 维扰动映射成与 g 正交的旋转向量, exp 后左乘 g,
     *           即只在切平面内旋转 g, 模长保持 GRAVITY 不变
     */
    void State::operator+=(const Vector23d &delta)
    {
        pos += delta.segment<3>(0);                                          // 位置
        rot *= Sophus::SO3d::exp(delta.segment<3>(3)).matrix();              // 姿态(右扰动)
        rot_ext *= Sophus::SO3d::exp(delta.segment<3>(6)).matrix();          // 雷达外参旋转
        pos_ext += delta.segment<3>(9);                                      // 雷达外参平移
        vel += delta.segment<3>(12);                                         // 速度
        bg += delta.segment<3>(15);                                          // 陀螺零偏
        ba += delta.segment<3>(18);                                          // 加计零偏
        g = Sophus::SO3d::exp(getBx() * delta.segment<2>(21)).matrix() * g;  // 重力: 2 维切空间扰动
    }

    /**
     * 24 维增量版本的流形加法, 前 7 个块的叠加方式与 23 维版本完全相同
     * 区别只在重力: 这里直接把 3 维旋转向量 delta.segment<3>(21) 取指数后左乘 g, 没有经 Bx 投影
     * NOTE: 该写法把重力当成有 3 个自由度, 与本文件其它地方"重力只有 2 个自由度"的约定不一致
     *       (误差状态第 21 块只有 2 维, P_ 的对应块是 2x2);
     *       该重载目前只在 predict() 中被调用, 而那里 delta 的重力段恒为 0, 故当前不影响结果
     */
    void State::operator+=(const Vector24d &delta)
    {
        pos += delta.segment<3>(0);
        rot *= Sophus::SO3d::exp(delta.segment<3>(3)).matrix();
        rot_ext *= Sophus::SO3d::exp(delta.segment<3>(6)).matrix();
        pos_ext += delta.segment<3>(9);
        vel += delta.segment<3>(12);
        bg += delta.segment<3>(15);
        ba += delta.segment<3>(18);
        g = Sophus::SO3d::exp(delta.segment<3>(21)).matrix() * g;
    }

    /**
     * 流形减法(盒减): 返回 "this ⊖ other" 的 23 维误差状态
     *   欧氏量: 直接相减
     *   rot / rot_ext: Log(other.rot^T * this.rot), 即右扰动约定下的姿态误差
     *                  (等价于 this.rot = other.rot * exp(该误差))
     *   重力: 球面流形上的误差 —— 先求把 other.g 转到 this.g 的旋转角 theta 与旋转轴,
     *         再用 other.g 处的切空间基 Bx 把该旋转向量投影成 2 维(见下面的两个分支)
     * 调用关系: IESKF::update() 中用 delta = x_ - predict_x 衡量当前迭代点相对预测均值的偏差
     */
    Vector23d State::operator-(const State &other)
    {
        Vector23d delta = Vector23d::Zero();
        delta.segment<3>(0) = pos - other.pos;                                          // 位置误差
        delta.segment<3>(3) = Sophus::SO3d(other.rot.transpose() * rot).log();          // 姿态误差(右扰动)
        delta.segment<3>(6) = Sophus::SO3d(other.rot_ext.transpose() * rot_ext).log();  // 外参旋转误差
        delta.segment<3>(9) = pos_ext - other.pos_ext;                                  // 外参平移误差
        delta.segment<3>(12) = vel - other.vel;                                         // 速度误差
        delta.segment<3>(15) = bg - other.bg;                                           // 陀螺零偏误差
        delta.segment<3>(18) = ba - other.ba;                                           // 加计零偏误差

        double v_sin = (Sophus::SO3d::hat(g) * other.g).norm();  // |g × other.g| = GRAVITY^2 * sin(theta)
        double v_cos = g.transpose() * other.g;                  // g · other.g     = GRAVITY^2 * cos(theta)
        double theta = std::atan2(v_sin, v_cos);                 // 两个重力向量的夹角, 取值 [0, pi]
        Eigen::Vector2d res;
        if (v_sin < 1e-11)
        {
            if (std::fabs(theta) > 1e-11)
            {
                // 两重力反向(theta = pi): 此时旋转轴在切平面内不唯一, 代码固定取 Bx 第一列方向
                // NOTE: 3.1415926 是硬编码的 pi 近似值, 而不是 pi 常数
                res << 3.1415926, 0;
            }
            else
            {
                // 两重力同向, 误差为 0
                res << 0, 0;
            }
        }
        else
        {
            // theta * n 是把 other.g 转到 this.g 的旋转向量(n 为旋转轴), 下面这行恰好等于
            // 用 other.g 处的切空间基把它投影成 2 维: theta / |other.g × g| * Bx^T * (other.g × g) = theta * Bx^T * n
            res = theta / v_sin * other.getBx().transpose() * Sophus::SO3d::hat(other.g) * g;
        }
        delta.segment<2>(21) = res;  // 重力误差(2 维)
        return delta;
    }

    /**
     * 重力球面流形在 g 处的切空间基 Bx(3x2)
     * 两列正交归一, 且都与 g 正交(即都落在 g 的切平面内); 于是 u = Bx * δθ 是一个与 g 正交的
     * 旋转向量, 重力方向经 g <- exp(u) * g 在切平面内旋转, 模长不变;
     * 换算成 3 维重力增量即 δg = -hat(g) * Bx * δθ, 也就是 getMx()
     * 说明: 公式按列展开后整体除以 GRAVITY 归一化, 推导中假设 |g| = GRAVITY(由 initG 保证)
     * NOTE: 表达式中除以 (GRAVITY + g[0]), 当重力指向 -x 轴(即 g[0] = -GRAVITY)时会除零;
     *       工程中重力大致朝下, 正常数据不会触发这个奇异点
     */
    Matrix3x2d State::getBx() const
    {
        Matrix3x2d res;
        // 逗号分隔按列主序填入: 每两个数是同一行的两列; 第 3 行由第 2 行把下标 1 与 2 互换得到
        res << -g[1], -g[2],
            GRAVITY - g[1] * g[1] / (GRAVITY + g[0]), -g[2] * g[1] / (GRAVITY + g[0]),
            -g[2] * g[1] / (GRAVITY + g[0]), GRAVITY - g[2] * g[2] / (GRAVITY + g[0]);
        res /= GRAVITY;  // 归一化, 使两列都是单位向量
        return res;
    }

    /**
     * 重力向量对 2 维切空间扰动的雅可比 Mx = ∂g/∂θ = -hat(g) * Bx
     * 由 hat(u) * g = -hat(g) * u (这里 u = Bx * δθ)得到, 即"在切平面里转 u"等价于重力向量变化 -hat(g) * u
     * 用途: predict() 中 F 的 (速度误差, 重力误差) 块
     */
    Matrix3x2d State::getMx() const
    {

        return -Sophus::SO3d::hat(g) * getBx();
    }

    /**
     * 迭代版的重力雅可比: 线性化参考点已经从旧 g 变成"施加过一次 2 维增量 res 之后"的 g
     * bu = Bx * res 就是那一次的旋转向量, 这里用 exp(bu) 与 bu 处的左雅可比转置(= 右雅可比)做修正;
     * res 取零时它退化为 getMx()
     * 用途: update() 中组装 J 与 L 的 (21,21) 块
     */
    Matrix3x2d State::getMx(const Eigen::Vector2d &res) const
    {
        Matrix3x2d bx = getBx();
        Eigen::Vector3d bu = bx * res;
        return -Sophus::SO3d::exp(bu).matrix() * Sophus::SO3d::hat(g) * Sophus::SO3d::leftJacobian(bu).transpose() * bx;
    }

    /**
     * 由 3 维重力之差反解 2 维切空间扰动的一阶雅可比: δθ ≈ Nx * δg
     * Nx = Bx^T * hat(g) / GRAVITY^2, 即 operator- 中大角度公式的小角度情形
     * 用途: predict() 中组装重力块自身的转移矩阵 F(21,21) = Nx * Mx
     */
    Matrix2x3d State::getNx() const
    {
        return 1 / GRAVITY / GRAVITY * getBx().transpose() * Sophus::SO3d::hat(g);
    }

    /**
     * 默认构造: 各成员取类内初始值(max_iter_ = 5, eps_ = 0.001, x_ 为单位/零状态)
     * NOTE: P_ 与 func_ 没有默认值 —— P_ 需外部先 setIdentity() / change_P() 赋值
     *       (见 LIOBuilder::initializeImu), 否则其内容未定义; func_ 未注册就调用 update()
     *       会抛 std::bad_function_call
     */
    IESKF::IESKF() = default;

    // 误差状态各块的下标常量(与 State 成员顺序一一对应), 定义在类外, 供外部按块读取 kf.P()
    int IESKF::P_ID = 0, IESKF::R_ID = 3, IESKF::ER_ID = 6, IESKF::EP_ID = 9, IESKF::V_ID = 12, IESKF::BG_ID = 15, IESKF::BA_ID = 18, IESKF::G_ID = 21;

    /**
     * IMU 递推: 用一段 IMU 测量把名义状态在 dt 上积分, 同时把协方差传播到同一时刻
     *
     * 名义状态(先攒成 24 维增量再一次性叠加, 避免中间态被反复读写):
     *   pos += vel * dt                              匀速积分
     *   rot *= exp((gyro - bg) * dt)                 角速度去偏后积分, 右乘更新
     *   vel += (rot * (acc - ba) + g) * dt           比力去偏后转到世界系, 再加上重力
     *   外参 / 零偏 / 重力在预测阶段视为常量, 它们的不确定度由 F_ 与 Q 传播
     * NOTE: 这里调用的是 24 维版本的 operator+=, 其中重力段恒为 0, 因此重力方向在预测中不变
     *
     * 误差状态协方差: P_ = F_ * P_ * F_^T + G_ * Q * G_^T
     *   F_ 是离散化的误差状态转移矩阵(对右扰动误差状态线性化), 各非零块的含义逐行见下;
     *   G_ 把 12 维 IMU 噪声映射到误差状态, Q 由调用方按同样的块顺序给出
     *   (LIOBuilder::loadConfig 里填的是陀螺/加计白噪声与两个零偏的随机游走)
     * 调用关系: LIOBuilder::undistortCloud() 中每个 IMU 区间调用一次, 点云去畸变时复用积分出的位姿
     */
    void IESKF::predict(const Input &inp, double dt, const Matrix12d &Q)
    {
        Vector24d delta = Vector24d::Zero();
        delta.segment<3>(0) = x_.vel * dt;                                  // 位置增量
        delta.segment<3>(3) = (inp.gyro - x_.bg) * dt;                      // 姿态增量(旋转向量)
        delta.segment<3>(12) = (x_.rot * (inp.acc - x_.ba) + x_.g) * dt;    // 速度增量(比力 + 重力)
        F_.setIdentity();
        F_.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity() * dt;                                    // 位置误差 <- 速度误差
        F_.block<3, 3>(3, 3) = Sophus::SO3d::exp(-(inp.gyro - x_.bg) * dt).matrix();                 // 姿态误差自身: 被 exp(-w*dt) 反向旋转到新切空间
        F_.block<3, 3>(3, 15) = -rightJacobian((inp.gyro - x_.bg) * dt) * dt;                        // 姿态误差 <- 陀螺零偏误差(右雅可比 * dt)
        F_.block<3, 3>(12, 3) = -x_.rot * Sophus::SO3d::hat(inp.acc - x_.ba) * dt;                   // 速度误差 <- 姿态误差(比力方向偏转)
        F_.block<3, 3>(12, 18) = -x_.rot * dt;                                                       // 速度误差 <- 加计零偏误差
        F_.block<3, 2>(12, 21) = x_.getMx() * dt;                                                    // 速度误差 <- 重力误差(2 维)
        F_.block<2, 2>(21, 21) = x_.getNx() * x_.getMx();                                            // 重力误差自身的转移矩阵

        G_.setZero();
        G_.block<3, 3>(3, 0) = -rightJacobian((inp.gyro - x_.bg) * dt) * dt;  // 陀螺白噪声 -> 姿态误差(与零偏误差同系数)
        G_.block<3, 3>(12, 3) = -x_.rot * dt;                                 // 加计白噪声 -> 速度误差
        G_.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity() * dt;             // 陀螺零偏随机游走 -> 零偏
        G_.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity() * dt;             // 加计零偏随机游走 -> 零偏
        x_ += delta;                                                          // 名义状态先在流形上积分
        P_ = F_ * P_ * F_.transpose() + G_ * Q * G_.transpose();              // 误差状态协方差传播(线性化 + 噪声注入)
    }

    /**
     * 迭代更新: 在预测均值附近反复线性化点-面残差, 直到误差状态增量收敛
     *
     * 记 predict_x 为预测(先验)均值, x_ 为当前迭代点, 每轮迭代做四件事:
     *   1) 回调 func_(x_, shared_data): LIOBuilder::sharedUpdateFunc 用当前名义状态把雷达点投到世界系,
     *      查体素地图构建点-面残差, 累加出 12x12 的观测信息 H 与 12x1 的 b(见 SharedState)
     *   2) delta = x_ ⊖ predict_x: 当前迭代点相对先验的 23 维误差状态(流形减法)
     *   3) J = ∂delta/∂(x_ 的局部扰动): 把"迭代点处的误差参数化"换算到"先验处的误差参数化",
     *      只有两个 SO3 块与重力块不是单位阵
     *   4) 组装正规方程并求解, 得到本轮的误差状态增量, 再叠加到名义状态上
     *
     * 正规方程(信息形式, 等价于最小化 ||x ⊖ predict_x||^2_{P^-1} + Σ ||r(x)||^2_{R^-1}):
     *   H_ = J^T * P^-1 * J + H_obs      P^-1 是先验信息矩阵, H_obs 只占前 12 维(其余维度只有先验约束)
     *   b_ = J^T * P^-1 * delta + b_obs
     *   delta_next = -H_^-1 * b_
     * 收敛判据: 本轮增量的最大分量小于 eps_ 时提前退出(见下面的 NOTE)
     *
     * 迭代结束后:
     *   L 与迭代内的 J 同构, 用收敛后的 delta 求值;
     *   H_ 是收敛点处的后验信息矩阵, 其逆即该参数化下的后验协方差, 再用 L 变换回
     *   与 predict() 递推、与点协方差传播(见 LIOBuilder 里读 kf.P() 分块)一致的参数化
     */
    void IESKF::update()
    {
        State predict_x = x_;               // 缓存预测均值(先验名义状态), 迭代过程中 x_ 会被反复更新
        SharedState shared_data;            // 观测回调的累加器(每轮由 func_ 自己清零)
        shared_data.iter_num = 0;           // 迭代计数清零
        Vector23d delta = Vector23d::Zero();
        for (size_t i = 0; i < max_iter_; i++)
        {
            func_(x_, shared_data);         // 在当前迭代点处重建残差与雅可比, 填充 shared_data.H / shared_data.b
            H_.setZero();
            b_.setZero();
            delta = x_ - predict_x;         // 当前迭代点相对先验的误差状态(流形减法)
            Matrix23d J = Matrix23d::Identity();                                                // 欧氏量块(pos/外参/速度/零偏)的雅可比是单位阵
            J.block<3, 3>(3, 3) = rightJacobian(delta.segment<3>(3));                            // 姿态块的参数化换算(右扰动)
            J.block<3, 3>(6, 6) = rightJacobian(delta.segment<3>(6));                            // 外参旋转块
            J.block<2, 2>(21, 21) = x_.getNx() * predict_x.getMx(delta.segment<2>(21));          // 重力块(2 维切空间)
            b_ += (J.transpose() * P_.inverse() * delta);   // 先验信息向量项 J^T * P^-1 * delta
            H_ += (J.transpose() * P_.inverse() * J);       // 先验信息矩阵项 J^T * P^-1 * J
            H_.block<12, 12>(0, 0) += shared_data.H;        // 叠加观测信息, 只落在前 12 维 [pos, rot, rot_ext, pos_ext]
            b_.block<12, 1>(0, 0) += shared_data.b;         // 叠加观测向量
            delta = -H_.inverse() * b_;                     // 解正规方程, 得到本轮误差状态增量
            x_ += delta;                                    // 在流形上叠加到名义状态(23 维版本 operator+=)
            shared_data.iter_num += 1;                      // 记录已完成的迭代轮次
            // NOTE: 收敛判据用的是 maxCoeff() 而不是 cwiseAbs().maxCoeff() 或 norm(),
            //       当增量的所有分量都为负时会被误判为已收敛(FAST-LIO 系实现原本即如此)
            if (delta.maxCoeff() < eps_)
                break;
        }
        // NOTE: 若 max_iter_ 为 0, 上面的循环一次也不执行, H_ 会保持全零, 末尾的 H_.inverse() 将产生 NaN
        Matrix23d L = Matrix23d::Identity();                                                // 与迭代内的 J 同构, 用于把后验协方差变换回先验参数化
        L.block<3, 3>(3, 3) = rightJacobian(delta.segment<3>(3));                            // 用收敛后的 delta 求值
        L.block<3, 3>(6, 6) = rightJacobian(delta.segment<3>(6));
        L.block<2, 2>(21, 21) = x_.getNx() * predict_x.getMx(delta.segment<2>(21));
        P_ = L * H_.inverse() * L.transpose();  // 后验协方差
    }

} // namespace kf