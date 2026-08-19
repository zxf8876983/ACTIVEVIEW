# ACTIVEVIEW Motion Assets & Data Infrastructure Tools

面向 ACTIVEVIEW 项目的动作数据资产（BABEL / AMASS）准备与前置基础设施工具集。

---

## 1. 统一数据目录与路径规则 (Data Root & Path Resolution)

项目采用源码仓库与运行时数据物理隔离规范：
- **源码仓库路径**：`code/ActiveView/`
- **外部数据根目录**：`data/ActiveView/`（默认基于仓库位置解析为相对路径 `../../data/ActiveView`）
- **环境变量覆盖**：可通过环境变量 `ACTIVEVIEW_DATA_ROOT` 指定绝对路径。

数据目录布局：
```text
data/ActiveView/
├── datasets/
│   ├── babel/babel_v1.0_release/       # BABEL annotations (train, val, extra_train, extra_val)
│   └── amass/                          # AMASS 解压后的 mocap npz 文件
├── assets/
│   └── motions/raw/                    # motion_asset_manifest.json (元数据清单)
├── cache/
│   ├── babel_selection/                # 动作筛选产物与候选清单
│   └── amass_download/                 # AMASS 下载状态与文件索引
├── logs/data_preparation/              # 数据准备日志
└── tmp/downloads/                      # 下载临时归档文件
```

---

## 2. 老人典型动作 5 类定义 (Elderly Action Taxonomy)

聚焦面向居家助老场景的 5 类基础与关键动作：
1. **`standing`**：自然站立、静止站立、起立站定；
2. **`sitting`**：坐姿、坐下、椅子端坐；
3. **`bending`**：弯腰、拾物、弯腰捡起地面物品；
4. **`reaching`**：前伸伸手、向上触及、触物交互；
5. **`fall_related`**：高置信跌倒（fall down, trip, collapse）与需人工确认的倒地姿态（lie on floor）。

---

## 3. 核心工具运行指南 (Workflow & Commands)

### 3.1 运行 BABEL 动作筛选器
解析 BABEL 标注并生成完整候选池与第一轮 Motion Feasibility Set：
```bash
python -m tools.motion_assets.select_babel_elderly_actions
```
产物位于 `cache/babel_selection/`：
- `elderly_action_candidates.csv` / `.json`
- `feasibility_manifest.json` / `.csv`
- `summary.json`
- `required_amass_subdatasets_feasibility.txt`

---

### 3.2 AMASS 自动下载流程 (Portal Downloader)

> **安全与合规要求**：
> - 必须在环境变量中设置 `AMASS_EMAIL` 和 `AMASS_PASSWORD`；
> - 绝对禁止将账号与密码硬编码进源码、配置文件或提交到 Git 仓库。

#### 步骤 1: 运行 Dry-Run 预检 (验证登录与资源映射)
```bash
python -m tools.motion_assets.download_amass_required --dry-run
```

#### 步骤 2: 执行实际下载与解压
```bash
python -m tools.motion_assets.download_amass_required --execute
```

---

### 3.3 手动下载应急方案 (Manual Download Fallback)
若因网络阻断或网站验证机制调整导致自动下载受阻，系统会自动生成：
- `cache/amass_download/manual_download_required.md`
- `cache/amass_download/manual_download_required.json`

用户可根据清单手动从 AMASS 官网下载对应子库压缩包并解压至 `datasets/amass/<SUBDATASET>/`。

---

### 3.4 索引本地 AMASS 文件与 Habitat 兼容性检查
扫描本地 AMASS 数据，建立 BABEL `feat_p` 到真实本地文件的映射，并校验 Habitat `MotionConverterSMPLX` 所需字段（`trans`, `root_orient`, `poses`, `frame_rate`）：
```bash
python -m tools.motion_assets.index_amass_files
```
产物：
- `cache/amass_download/amass_file_index.json`
- `cache/amass_download/habitat_motion_compatibility.csv`
- `assets/motions/raw/motion_asset_manifest.json`

---

### 3.5 运行单元测试
```bash
python -m unittest tools.motion_assets.tests.test_motion_assets
```

---

## 4. 安全与数据保护承诺
1. **禁止第三方数据入库**：BABEL annotations 与 AMASS npz/tar.bz2 等数据只存放在外部数据目录，严禁提交进 Git 仓库；
2. **凭据安全**：代码中仅通过 `os.environ` 读取账号信息，禁止在任何文件、日志或 commit 中输出真实密码。
