#include "commons.h"
namespace lio
{

    /**
     * 旋转矩阵 -> 欧拉角(roll, pitch, yaw)
     *
     * 按 ZYX 约定分解 R = Rz(yaw) * Ry(pitch) * Rx(roll)(即常用的 RPY 定义):
     *   roll  = atan2(R(2,1), R(2,2))
     *   pitch = asin(-R(2,0))
     *   yaw   = atan2(R(1,0), R(0,0))
     * 未处理万向锁(pitch 接近 ±90 度时 roll/yaw 退化)
     *
     * 调用关系: 本工程内暂无调用者
     */
    Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot)
    {
        double roll = std::atan2(rot(2, 1), rot(2, 2)); // 绕 X 轴(滚转)
        double pitch = asin(-rot(2, 0));                // 绕 Y 轴(俯仰)
        double yaw = std::atan2(rot(1, 0), rot(0, 0));  // 绕 Z 轴(偏航)
        return Eigen::Vector3d(roll, pitch, yaw);
    }

    /**
     * 两个点之间的欧氏距离平方
     * 只统计 x/y/z, 返回平方值(省一次开方), 需要真实距离时自行开方
     *
     * 调用关系: 本工程内暂无调用者
     */
    float sq_dist(const pcl::PointXYZINormal &p1, const pcl::PointXYZINormal &p2)
    {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    }

    /**
     * 计算雷达系下一点的位置协方差(量测噪声模型)
     *
     * 把点写成"沿视线方向的距离 + 两个正交角度"的量测, 假设测距与测角误差独立:
     *   range_var     = range_inc * range_inc          测距方差
     *   direction_var = diag(sin(角度)^2, sin(角度)^2) 垂直于视线的两个方向上的角度方差(2x2)
     *   cov = d * range_var * d^T + A * direction_var * A^T
     * 其中 d 是视线单位向量, A = range * hat(d) * N,
     * N 的每一列是垂直于 d 的单位基向量, hat(d) 是 d 的反对称矩阵,
     * A 的作用是把单位球切平面上的角度扰动映射为三维位置扰动
     *
     * 参数:
     *   pb         雷达系下的点坐标(注意: 会被原地修改)
     *   range_inc  测距噪声(按方差使用, 即 range_inc 的平方)
     *   degree_inc 测角噪声, 单位为度(内部经 DEG2RAD 转成弧度后取 sin 平方)
     *   cov        输出: 3x3 的位置协方差
     *
     * 调用关系: 由 LIOBuilder::process() 对每个点调用(首帧建图与正常帧各一处);
     *           求出的协方差在 process()/sharedUpdateFunc() 中随状态一起传播到世界系
     *
     * NOTE: pb 是引用, 调用者传入的坐标可能被就地改写(z 恰为 0 时被改成 0.001);
     *       另 DEG2RAD 未在本工程内定义, 由外部依赖头文件提供
     */
    void calcBodyCov(Eigen::Vector3d &pb, const double &range_inc, const double &degree_inc, Eigen::Matrix3d &cov)
    {
        if (pb[2] == 0)
        {
            pb[2] = 0.001; // 兜底: 避免下面构造 base_vector1 时除以 0(只覆盖恰好为 0 的情况)
        }
        double range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]); // 点到原点的距离
        double range_var = range_inc * range_inc; // 测距方差
        Eigen::Matrix2d direction_var;
        direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
            pow(sin(DEG2RAD(degree_inc)), 2); // 两个角度方向上的对角方差(度转弧度后取 sin 平方)
        Eigen::Vector3d direction(pb);
        direction.normalize(); // 视线单位向量
        Eigen::Matrix3d direction_hat;
        direction_hat << 0, -direction(2), direction(1), direction(2), 0,
            -direction(0), -direction(1), direction(0), 0; // direction 的反对称矩阵
        Eigen::Vector3d base_vector1(1, 1,
                                     -(direction(0) + direction(1)) / direction(2)); // 与 direction 正交的向量
        base_vector1.normalize();
        Eigen::Vector3d base_vector2 = base_vector1.cross(direction); // 与 direction 正交, 与 base_vector1 一起构成切平面基
        base_vector2.normalize();
        Eigen::Matrix<double, 3, 2> N; // 3x2, 每列是切平面内的一个单位基向量
        N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
            base_vector1(2), base_vector2(2);
        Eigen::Matrix<double, 3, 2> A = range * direction_hat * N; // 角度扰动 -> 位置扰动的雅可比
        cov = direction * range_var * direction.transpose() +
              A * direction_var * A.transpose(); // 测距项 + 测角项
    }
}
