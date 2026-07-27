# 推理脚本实现计划

## 1. 需求分析

参考 `test.py`（另一个项目的推理脚本），需要为当前项目编写一个推理脚本，用于：

### 参考脚本核心功能
- **数据采集**：订阅ROS2话题获取图像（头部、左手、右手）和关节状态
- **推理请求**：通过websocket连接到策略服务器，发送观测数据获取动作
- **动作执行**：Action Chunking + 统一插值策略
- **跨Chunk过渡**：平滑过渡相邻Chunk的动作
- **位置保持**：后台线程持续保持当前位置

### 当前项目架构
- 服务器端：`scripts/serve_policy.py` + `src/openpi/serving/websocket_policy_server.py`
- 客户端：`openpi_client.websocket_client_policy.WebsocketClientPolicy`
- 通信协议：Websocket + msgpack

## 2. 实现方案

### 2.1 文件结构

创建新文件：`scripts/inference.py`

### 2.2 核心模块设计

#### 2.2.1 数据采集模块（参考test.py的RobotObservationNode）
- 订阅ROS2话题：
  - `/camera/head_color` → 头部图像
  - `/camera/hand_left_color` → 左手图像
  - `/camera/hand_right_color` → 右手图像
  - `/hal/arm_joint_state` → 关节状态
  - `/hal/left_ee_data` → 左夹爪状态
  - `/hal/right_ee_data` → 右夹爪状态

#### 2.2.2 观测构建模块
- 将原始数据转换为策略期望的格式：
  ```python
  {
      'video': {
          'observation.images.top_head': ...,
          'observation.images.hand_left': ...,
          'observation.images.hand_right': ...,
      },
      'state': {
          'left_arm_joint_position': ...,
          'right_arm_joint_position': ...,
          'left_effector_position': ...,
          'right_effector_position': ...,
      },
      'language': {
          'annotation.human.task_description': ...,
      }
  }
  ```

#### 2.2.3 推理客户端模块
- 使用 `WebsocketClientPolicy` 连接到服务器
- 发送观测数据获取动作预测

#### 2.2.4 动作执行模块（参考test.py的Action Chunking）
- 提取前N步原始动作
- 跨Chunk过渡插值（五次样条）
- Chunk内插值（五次样条）
- 发布关节控制指令

#### 2.2.5 位置保持模块
- 后台线程以20Hz持续发送保持指令

### 2.3 关键参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CONTROL_FREQ | 20 | 控制频率(Hz) |
| EXECUTION_HORIZON | 8 | 每次执行原始步数 |
| INTERPOLATION_FACTOR | 5 | Chunk内插值倍数 |
| TRANSITION_INTERPOLATION_FACTOR | 5 | Chunk间过渡插值倍数 |
| ACTION_HORIZON | 32 | 模型预测总步数 |

## 3. 实施步骤

### 步骤1：创建推理脚本框架
- 创建 `scripts/inference.py`
- 导入必要依赖（numpy, cv2, rclpy, PolicyClient等）
- 定义核心类 `RobotInferenceNode`

### 步骤2：实现数据采集功能
- 实现ROS2订阅回调
- 存储最新图像和状态数据
- 实现数据就绪检查

### 步骤3：实现观测构建功能
- 处理图像（resize到640x480）
- 处理关节状态（分离左右臂）
- 构建符合策略输入格式的观测字典

### 步骤4：实现推理客户端
- 初始化 `WebsocketClientPolicy`
- 连接到策略服务器
- 发送观测获取动作

### 步骤5：实现动作执行（Action Chunking）
- 实现跨Chunk过渡插值
- 实现Chunk内五次样条插值
- 实现动作发布

### 步骤6：实现位置保持
- 启动后台保持线程
- 持续发送当前位置指令

### 步骤7：实现主循环
- 推理-执行循环
- 统计和日志输出
- 异常处理

## 4. 依赖与兼容性

### 已存在的依赖
- `numpy`
- `cv2`
- `rclpy`
- `sensor_msgs`
- `cv_bridge`
- `openpi_client`（通过 `packages/openpi-client`）

### 需要确保的配置
- ROS2环境已配置
- 策略服务器已启动（通过 `scripts/serve_policy.py`）
- 机器人硬件已连接并发布所需话题

## 5. 风险与注意事项

### 风险点
1. **数据格式不匹配**：观测数据格式必须与策略期望完全一致
2. **网络延迟**：websocket通信可能存在延迟，影响实时控制
3. **插值稳定性**：五次样条插值在极端情况下可能产生振荡

### 缓解措施
1. 严格按照 `g01_policy.py` 中的格式要求构建观测
2. 添加超时机制和性能统计
3. 使用平滑参数限制插值幅度

## 6. 验证方案

### 功能验证
1. **数据采集验证**：运行脚本，检查是否能正确接收图像和关节数据
2. **推理连接验证**：检查是否能成功连接到策略服务器
3. **动作执行验证**：检查机器人是否能执行推理得到的动作
4. **过渡平滑验证**：检查Chunk间过渡是否平滑无跳变

### 性能指标
- 推理频率：> 5Hz
- 动作执行延迟：< 50ms
- 过渡平滑度：关节速度变化率 < 阈值
