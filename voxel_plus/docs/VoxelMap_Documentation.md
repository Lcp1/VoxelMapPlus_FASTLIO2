# VoxelMapPlus FAST-LIO 体素地图系统文档

## 1. 系统概述

VoxelMapPlus 是 FAST-LIO2 系统中的核心地图表示模块，采用**体素化平面特征**来高效表示环境结构。与传统点云地图相比，体素地图通过将空间划分为三维网格单元（体素），并在每个体素内提取平面特征，实现了**内存占用低、计算效率高、适合实时定位与建图**的特点。

## 2. 核心组件

### 2.1 `VoxelGrid` 类 ([voxel_map.h:22-85](voxel_plus/src/map_builder/voxel_map.h#L22-L85))

表示单个体素单元，包含以下关键成员：

- **平面参数**：`plane` (包含均值 `mean`、法向量 `norm` 和协方差 `cov`)
- **点云数据**：`temp_points` (临时存储原始点云，用于平面拟合)
- **状态标志**：`is_init` (是否初始化)、`is_plane` (是否为平面)
- **合并机制**：`group_id` (用于合并后统一管理)、`merged` (是否已被合并)

**关键方法**：

- `addPoint()`/`pushPoint()`：添加点云数据，触发平面更新或合并
- `updatePlane()`：[第97-136行](voxel_plus/src/map_builder/voxel_map.cpp#L97-L136) - 基于PCA计算平面参数
- `merge()`：[第138-186行](voxel_plus/src/map_builder/voxel_map.cpp#L138-L186) - 与相邻体素合并逻辑

### 2.2 `VoxelMap` 类 ([voxel_map.h:87-145](voxel_plus/src/map_builder/voxel_map.h#L87-L145))

管理全局体素地图，核心特性：

- **空间索引**：使用哈希表 `featmap` 存储有效体素 ([第192行](voxel_plus/src/map_builder/voxel_map.cpp#L192))
- **LRU缓存**：`cache` 链表实现最近最少使用策略，控制内存占用 ([第209行](voxel_plus/src/map_builder/voxel_map.cpp#L209))
- **数据处理**：`build()` 和 `update()` 方法处理新点云
- **残差计算**：`buildResidual()` 为优化提供测量值 ([第258-277行](voxel_plus/src/map_builder/voxel_map.cpp#L258-L277))

## 3. 数据流详解

### 3.1 点云处理流程

```mermaid
graph TD
    A[原始点云] --> B{转换到体素坐标}
    B -->|index()| C[定位体素位置]
    C --> D{体素是否存在？}
    D -->|存在| E[更新体素 cache]
    D -->|不存在| F[创建新体素]
    E --> G[addPoint/pushPoint]
    F --> G
    G --> H{点数是否达阈值？}
    H -->|是| I[updatePlane]
    H -->|否| J[等待新数据]
    I --> K{是否为有效平面？}
    K -->|是| L[尝试合并相邻体素]
    K -->|否| M[标记为非平面]
```

### 3.2 关键步骤说明

1. **体素定位** (`index` 方法)：
   ```cpp
   Eigen::Vector3d idx = (point / voxel_size).array().floor();
   ```
   将三维点坐标按 `voxel_size` 大小离散化为整数体素坐标 ([第195-197行](voxel_plus/src/map_builder/voxel_map.cpp#L195-L197))

2. **平面检测** (`updatePlane` 方法)：
   - 计算点云协方差矩阵 ([第103行](voxel_plus/src/map_builder/voxel_map.cpp#L103))
   - 特征分解获取主成分 ([第104-106行](voxel_plus/src/map_builder/voxel_map.cpp#L104-L106))
   - 最小特征值 < `plane_thresh` → 判定为平面 ([第107-112行](voxel_plus/src/map_builder/voxel_map.cpp#L107-L112))

3. **体素合并** (`merge` 方法)：
   - 检查6邻域体素 ([第140-146行](voxel_plus/src/map_builder/voxel_map.cpp#L140-L146))
   - 法向量角度差 < `merge_thresh_for_angle` (0.1) **AND** 平面距离 < `merge_thresh_for_distance` (0.04) → 允许合并 ([第159-160行](voxel_plus/src/map_builder/voxel_map.cpp#L159-L160))
   - 合并后更新平面参数 ([第166-183行](voxel_plus/src/map_builder/voxel_map.cpp#L166-L183))

## 4. 功能参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `voxel_size` | - | 体素物理尺寸（需在配置文件设置） |
| `max_point_thresh` | - | 体素最大点数阈值（超出则停止更新） |
| `update_point_thresh` | - | 平面更新所需最小点数 |
| `plane_thresh` | - | PCA最小特征值阈值（判定平面） |
| `merge_thresh_for_angle` | 0.1 | 合并允许的最大法向量角度差 |
| `merge_thresh_for_distance` | 0.04 | 合并允许的最大平面距离（米） |
| `capacity` | - | 体素地图最大容量（通过LRU管理） |

> 提示：这些参数可在 [mapping.yaml](voxel_plus/config/mapping.yaml) 中配置

## 5. 与FAST-LIO2系统集成

- **前端输入**：来自 [lio_node.cpp](voxel_plus/src/lio_node.cpp) 的预处理点云
- **后端优化**：通过 `buildResidual()` 提供平面残差，用于 IESKF 优化 ([ieskf.cpp](voxel_plus/src/map_builder/ieskf.cpp))
- **数据流衔接**：
  ```
  PointCloud → VoxelMap::update() → ResidualData → IESKF::propagate()
  ```

## 6. 性能优化特点

1. **增量更新机制**：`update_enable` 标志控制动态更新频率
2. **LRU内存管理**：自动淘汰不活跃体素，控制内存峰值
3. **合并策略**：减少重复平面特征，提升地图紧凑性
4. **协方差传播**：在平面合并时考虑不确定性 ([第168行](voxel_plus/src/map_builder/voxel_map.cpp#L168))

> **注意事项**：该实现已适配 [KITTI数据集](voxel_plus/launch/mapping_kitti.launch)，通过 `mapping_port.yaml` 配置可支持不同传感器参数