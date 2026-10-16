// 把"只有位姿"的数据集(如 KITTI 的 /pose_imu, 10Hz)合成成 sensor_msgs/Imu(默认 100Hz), 供 LIO 使用。
// 做法: 位姿序列局部二次拟合得到速度/加速度(比直接差分抗噪), 姿态用 slerp 插值, 角速度取段内常值;
// 再把世界系加速度转到机体系并减去重力得到比力 f = R^T (a - g)。
// 注意: 加速度是由 10Hz 位姿二次微分得来的, 噪声较大, 配置里应把 na(加速度计噪声)调大, 让点云主导状态。
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <deque>
#include <Eigen/Eigen>
#include <sophus/so3.hpp>

namespace
{
    struct PoseSample
    {
        double t = 0.0;
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        Eigen::Vector3d v = Eigen::Vector3d::Zero(); // 局部拟合得到的速度
        Eigen::Vector3d a = Eigen::Vector3d::Zero(); // 局部拟合得到的加速度(世界系)
    };

    class PoseToImu
    {
    public:
        PoseToImu() : nh("~")
        {
            nh.param<std::string>("pose_topic", pose_topic, "/pose_imu");
            nh.param<std::string>("imu_topic", imu_topic, "/synthetic_imu");
            nh.param<std::string>("imu_frame", imu_frame, "imu");
            nh.param<double>("rate", rate, 100.0);
            nh.param<double>("gravity", gravity, 9.81);
            nh.param<int>("poly_win", poly_win, 5);        // 拟合窗口(单边点数): 2*poly_win+1 个位姿
            nh.param<bool>("input_is_transform", input_is_transform, false);
            nh.param<bool>("publish_velocity", publish_velocity, true);

            pub = nh.advertise<sensor_msgs::Imu>(imu_topic, 1000);
            if (publish_velocity)
                vel_pub = nh.advertise<geometry_msgs::TwistStamped>(imu_topic + "/velocity", 100);
            if (input_is_transform)
                sub_t = nh.subscribe(pose_topic, 2000, &PoseToImu::transformCB, this);
            else
                sub_p = nh.subscribe(pose_topic, 2000, &PoseToImu::poseCB, this);
            ROS_INFO("pose_to_imu: %s -> %s @ %.0fHz (poly_win=%d, frame=%s)", pose_topic.c_str(), imu_topic.c_str(), rate, poly_win, imu_frame.c_str());
        }

    private:
        void poseCB(const geometry_msgs::PoseStamped::ConstPtr &msg)
        {
            addSample(msg->header.stamp.toSec(), msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,
                      msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
        }
        void transformCB(const geometry_msgs::TransformStamped::ConstPtr &msg)
        {
            addSample(msg->header.stamp.toSec(), msg->transform.translation.x, msg->transform.translation.y, msg->transform.translation.z,
                      msg->transform.rotation.x, msg->transform.rotation.y, msg->transform.rotation.z, msg->transform.rotation.w);
        }

        void addSample(double t, double x, double y, double z, double qx, double qy, double qz, double qw)
        {
            PoseSample s;
            s.t = t;
            s.p = Eigen::Vector3d(x, y, z);
            s.q = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
            if (!buf.empty() && t <= buf.back().t)
            {
                ROS_WARN("pose_to_imu: non-monotonic pose time, clear buffer");
                buf.clear();
            }
            buf.push_back(s);
            while (buf.size() > static_cast<size_t>(2 * poly_win + 2))
                buf.pop_front();
            if (buf.size() < static_cast<size_t>(2 * poly_win + 2))
                return;
            // 缓冲区有 2*poly_win+2 个样本(idx 0..2k+1)后, 对中间的 k 与 k+1 各做一次对称窗口拟合(无滞后),
            // 再输出 [k, k+1] 之间的IMU。输出比位姿晚 poly_win 个样本(0.5s), 这是无滞后估计的代价;
            // LIO 会等IMU时间追上点云时刻, 只是延迟处理, 不会死锁
            const size_t n = poly_win;
            fitLocal(n);
            fitLocal(n + 1);
            emitSegment(n, n + 1);
        }

        // 对 buf[i] 用"以 i 为右端"的历史窗口做局部二次拟合, 得到平滑的速度/加速度。
        // 只用历史样本是为了不引入额外时延(否则 LIO 的 IMU 会落后于点云, 同步会卡住)
        void fitLocal(size_t i)
        {
            const PoseSample &c = buf[i];
            const int k = poly_win;
            int n = 2 * k + 1;
            Eigen::MatrixXd A(n, 3);
            Eigen::MatrixXd B(n, 3);
            int row = 0;
            for (int jj = -k; jj <= k; jj++)
            {
                const PoseSample &s = buf[i + jj];
                double dt = s.t - c.t;
                A(row, 0) = 1.0;
                A(row, 1) = dt;
                A(row, 2) = 0.5 * dt * dt;
                B.row(row) = s.p.transpose();
                row++;
            }
            Eigen::Matrix3d AtA = A.transpose() * A;
            Eigen::MatrixXd coef = (AtA + Eigen::Matrix3d::Identity() * 1e-9).inverse() * A.transpose() * B;
            buf[i].v = coef.row(1).transpose(); // dp/dt at center
            buf[i].a = coef.row(2).transpose(); // d2p/dt2 at center
        }

        void emitSegment(size_t i, size_t j)
        {
            const PoseSample &s0 = buf[i];
            const PoseSample &s1 = buf[j];
            double dt = s1.t - s0.t;
            if (dt <= 0.0)
                return;
            // 段内角速度(机体系): log(R0^T R1)/dt
            Eigen::Matrix3d dR = s0.q.conjugate().toRotationMatrix() * s1.q.toRotationMatrix();
            Eigen::Vector3d omega = Sophus::SO3d(dR).log() / dt;
            // 重力: 位姿所在的世界系是 ENU 定向(z 向上), 直接取竖直向下即可
            Eigen::Vector3d g_world(0.0, 0.0, -gravity);
            int n_steps = std::max(1, static_cast<int>(std::round(dt * rate)));
            for (int m = 0; m < n_steps; m++)
            {
                double u = static_cast<double>(m) / n_steps;
                double t = s0.t + u * dt;
                Eigen::Quaterniond q = s0.q.slerp(u, s1.q);
                Eigen::Vector3d a_world = (1.0 - u) * s0.a + u * s1.a;
                Eigen::Vector3d f_body = q.conjugate() * (a_world - g_world);

                if (publish_velocity)
                {
                    // 机体系速度: 常速度对加速度计不可观测, 但位姿的导数可以直接给出, 供 LIO 初始化用
                    Eigen::Vector3d v_world = (1.0 - u) * s0.v + u * s1.v;
                    Eigen::Vector3d v_body = q.conjugate() * v_world;
                    geometry_msgs::TwistStamped tw;
                    tw.header.stamp = ros::Time().fromSec(t);
                    tw.header.frame_id = imu_frame;
                    tw.twist.linear.x = v_body.x();
                    tw.twist.linear.y = v_body.y();
                    tw.twist.linear.z = v_body.z();
                    vel_pub.publish(tw);
                }

                sensor_msgs::Imu msg;
                msg.header.stamp = ros::Time().fromSec(t);
                msg.header.frame_id = imu_frame;
                msg.linear_acceleration.x = f_body.x();
                msg.linear_acceleration.y = f_body.y();
                msg.linear_acceleration.z = f_body.z();
                msg.angular_velocity.x = omega.x();
                msg.angular_velocity.y = omega.y();
                msg.angular_velocity.z = omega.z();
                msg.orientation = geometry_msgs::Quaternion(); // 定位姿由LIO自己估计
                pub.publish(msg);
            }
        }

        ros::NodeHandle nh;
        ros::Publisher pub, vel_pub;
        ros::Subscriber sub_p, sub_t;
        bool publish_velocity = true;
        std::deque<PoseSample> buf;

        std::string pose_topic, imu_topic, imu_frame;
        double rate = 100.0, gravity = 9.81;
        int poly_win = 5;
        bool input_is_transform = false;
    };
} // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pose_to_imu");
    PoseToImu node;
    ros::spin();
    return 0;
}
