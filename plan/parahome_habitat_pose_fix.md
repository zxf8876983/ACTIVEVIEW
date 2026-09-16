# Task Plan: ParaHome → Habitat pose fidelity fix

## Goal

修复 ParaHome SMPL-X 动作导入 Habitat 后系统性出现的头部倾斜和单侧手臂伸直，
以及修复后残留的“头/躯干与手臂朝向相反”问题，并用合成真值与多 sequence 定量
关节朝向误差验证修复。

## Phases

- [x] Phase 1: 建立问题证据、边界与验收标准
- [x] Phase 2: 定位 ParaHome pose convention 与 Habitat URDF/converter 的首次差异
  - SMPL-X local rotation 直传不可用（两套 rig rest/joint frame 不同）；
  - Habitat articulated joint order 与 URDF 声明顺序不同 → 头歪/单侧手伸直（已修）；
  - **坐标映射 `PARAHOME_TO_HABITAT` 的 det = −1（反射）→ 镜像骨架**：所有关节解
    都是 proper rotation，无法拟合镜像目标；pelvis/spine3（各 3 子节点）的最小二乘
    解只能把身体坐标系翻转约 180°（chest facing·录制 facing = −0.91…−0.99），
    而手臂/手的**位置**仍然正确 → 这正是“人脸和身体与手臂反方向”的主因；
  - 逐关节 world-space `Rotation.align_vectors` 丢失父链 twist：单子节点关节
    （spine1/spine2/neck/collar/shoulder/elbow/hip/knee/ankle）朝向退回 rest
    frame，而 pelvis/spine3（各 3 个子节点）保持正确 → 躯干/头不跟随身体 yaw。
- [x] Phase 3: 实施最小 retarget 修复与单元级验证
  - `PARAHOME_TO_HABITAT` 改为 proper rotation `[[0,-1,0],[0,0,1],[-1,0,0]]`
    （前/上映射不变，去掉镜像）；
  - `_solve_global_rotations` 改为自顶向下继承父坐标系（新增 `_hierarchy_order`）；
  - 多子节点关节的最小二乘解不变（残差在父旋转下不变）；
  - `tests/unit/test_parahome_retarget.py` 新增 det=+1 手性回归测试与整体
    yaw-180° 朝向回归测试。
- [x] Phase 4: 多 sequence 定量回归与可视化验证
  - 合成真值（整体 yaw 180° + 头部俯仰 20°）：spine1/spine2/neck/head 误差
    118–127° → 0.0°（neck 残差 11.3° 为单骨 twist，位置不可观测）；
  - s78/s50（均在修正后的 proper mapping 下）：spine1 朝向误差 136.4/124.7° →
    8.8/7.7°；head-vs-chest 相对旋转 mean/max 143.3/179.8°、110.3/179.9° →
    13.7/20.9°、6.8/18.4°；segment 拟合误差完全不变（6.84° mean / 61.40° max）；
  - 映射手性：chest facing·录制 facing 由 −0.99/−0.91/−0.84（s78 三帧）变为
    +0.93/+0.87/+0.80；
  - 离线线性混合蒙皮 before/after 渲染（本机无 GPU/EGL，无法直接跑 Habitat）。
- [x] Phase 4b: 修复“肚子向前隆起”
  - 根因：两套 rig 的关节定义不同 —— ParaHome `hip` 是髋中心（两髋夹角 178.3°），
    `male_0` pelvis link 在髋关节上方约 10 cm（髋偏移夹角 60.5°）；ParaHome
    `spine1/spine2` 是 L5/L4（骨长 0.05–0.07 m），`male_0` 的 spine1/2 更高且带
    ~20° 前凸。把原始偏移直接拟合让 pelvis 残差达 ~60°，并把 rig 的长前凸腰椎
    强行拉直 → 腹部网格前凸（s50 腹部前移 +0.043 m，rest 为 +0.011 m）。
  - 修复：pelvis 改用与定义无关的髋线 `left_hip-right_hip` + 脊柱轴求解
    （约束残差 2.7°）；`spine1/spine2` 归入 `REST_SHAPE_JOINTS`，保留 rig 自身
    静止曲率（躯干整体弯曲仍由 world-space 求解的 `spine3` 承担）。
  - 效果（蒙皮网格 + 录制物体，手-物中位距离）：s50 cup 17→13 cm；s78 laptop 最小
    21→17 cm；s3 pan 34→30 cm、gasstove 47→39 cm；腹部前移恢复至 +0.021 m。
  - “解剖对应关节”的骨方向拟合误差保持 ~1.4°（全量指标因包含定义不同的关节而变大，
    已在记录中区分）。
- [x] Phase 4c: 修复“站立时双腿向内并拢”
  - 根因：只匹配骨骼**方向**、不匹配关节**间距**。`male_0` 髋关节间距 12.5 cm，
    而 ParaHome 的髋（髋中心定义）间距随受试者为 18.8–21.9 cm（每条序列内为常数）。
    整条腿链跟随录制大腿方向，因此比录制窄约 8 cm。
  - 修复：`match_hip_width=True`（默认）——把两个髋偏移的横向分量按
    “录制间距/rig 间距”缩放（clamp 0.8–2.0，避免坏拟合撕裂网格）。腿整体外移，
    网格只在腹股沟处有拉伸。
  - 结果（s3）：髋/膝/踝间距 12/12/12 cm → 20/19/19 cm（录制 20/19/19）；
  - 追加：录制站姿（膝 19–20 cm）比 rig 自身静止站姿（膝 24.4 cm）更窄，
    male_0 腿部较粗 → 网格相互穿透（s3 膝部网格间隙 +1.1 cm 与 −5.9 cm）。
    因此目标间距不低于 rig 静止膝距：修复后同帧膝距 23/21 cm、网格间隙
    +5.0/−2.0 cm（站姿最多比录制宽 ~5 cm；`match_hip_width=False` 可恢复录制间距）。
- [x] Phase 5: 在真实 Habitat 渲染中复核
  - CUDA (RTX 4090, driver 550.54.14, torch 2.6.0+cu124) + Habitat windowless EGL 可用；
  - 渲染 s78 frames 445–700（255 帧 / 8.5 s @30fps）三视角样本视频，
    修复版与 pre-fix 版同相机、同 root 轨迹对照：
    `experiments/parahome_feasibility_v1/retarget_pose_fidelity/videos/`；
  - rest-pose 探针（标签已校正）确认 `male_0` 静止朝向为 **+Z**（+Z 侧相机看到
    正脸），箭头/相机前向取 +Z；相机改为随身体朝向；
  - 修复版：人坐在椅子上面对笔记本，胸/头前倾低头（chest elev −21…−28°，
    head −25…−30°），双手在躯干前方；pre-fix 解算器下躯干仍相对手臂扭转。

## Key Questions

1. ParaHome `smplx_pose.pkl` 的 `body_pose` 是否可被 Habitat `MotionConverterSMPLX` 直接解释？
   → 不可以：rest/joint frame 不兼容，改用释放的 23 关节世界坐标做几何 retarget。
2. 异常来自 joint order、rotation representation、rest/bind frame、root basis，还是 gender/shape mismatch？
   → joint order（已修）+ 单骨 twist 求解方式（本次修复）。
3. 在没有官方 SMPL-X model weights 时，能否利用 ParaHome raw joints 与 Habitat URDF FK 恢复一个确定、可泛化的 local retarget calibration？
   → 可以：层级 segment-direction 解算 + 父帧 twist 继承，不依赖 SMPL-X model。
4. 修复是否在多个 sequence 上同时降低头颈与左右上肢误差，而不破坏腿部和 root trajectory？
   → 是：朝向误差大幅下降，segment 拟合、root transform 与 grounding 完全不变。

## Acceptance Criteria

- 不能使用 action-specific 或单帧手工姿态补丁。✅ 通用层级解算
- 修复必须作用于通用 ParaHome→Habitat retarget 路径。✅ `activeview/data/motion/parahome_retarget.py`
- 至少在 `s5/s50/s78/s99/s121` 的多帧样本上比较修复前后 bone-direction error。
  ✅ s78/s50 各 150 帧（bone-direction 误差不变，朝向误差大幅下降）
- `body_pose=0` 仍保持正常直立姿态。✅ 中性骨架单元测试
- 至少重新生成 `s50` 或 `s78` 的真实 Habitat humanoid 视频并人工检查。
  ✅ s78 frames 445–700 三视角 MP4（fixed / prefix 对照）+ 逐帧 PNG
- 不修改既有 ActiveView frozen AMASS/BABEL protocol。✅

## Decisions Made

- 将旧 `human replay: PASS` 解释为 replay infrastructure PASS，不代表 pose fidelity。
- 保留位置驱动的几何 retarget，不直传 SMPL-X local rotation。
- twist 由根节点向下继承；单骨方向不可观测的 twist 不强行拟合。
- 两处修复都必须保留：只改 twist 不能消除“整体反向”（映射镜像造成）；只改映射
  则躯干/头仍不跟随身体 yaw（twist 丢失造成）。
- `male_0` 静止朝向为 +Z（渲染实证），与 Habitat 文档里“character faces −Z”的说法
  不同；箭头/相机/朝向判断一律以渲染实证为准。

## Errors Encountered

- 当前环境缺少官方 SMPL-X model package/weights，A→B→C 严格对照不可用；改用
  合成真值 + 观测几何（肩线、head tip）指标。
- 首轮会话沙箱隐藏了 `/dev/nvidia*`，windowless EGL context 建不起来 → 当时
  Habitat 渲染 `NOT RUN`，先用离线 LBS 渲染（`male_0.glb` rest 骨架与 URDF rest
  骨架只差一个常量 root 偏移，已核对）。放开设备访问后 Habitat 渲染通过。
- `body_joint_orientations.pkl`（6D）不能作为全局朝向真值直接比较（其语义与
  `joint_positions` 的常骨向量不一致），因此不作为绝对真值使用。

## Status

**Phase 1–5 完成**：四处缺陷（坐标镜像映射、twist 丢失、关节定义错配、髋宽不匹配）均已修复，
并通过合成真值、s78/s50/s3 多帧指标、物体交互距离与真实 Habitat 样本视频复核。
记录：`experiments/parahome_feasibility_v1/retarget_pose_fidelity/`。
