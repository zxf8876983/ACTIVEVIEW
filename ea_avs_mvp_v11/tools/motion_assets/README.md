# ACTIVEVIEW Motion Assets & Data Infrastructure Tools

面向 ACTIVEVIEW 项目的动作数据资产（BABEL / AMASS）准备与前置基础设施工具集。

---

## 1. 统一数据目录与路径规则 (Data Root & Path Resolution)

项目采用源码仓库与运行时数据物理隔离规范：
- **源码仓库路径**：`<workspace>/code/ActiveView/`
- **外部数据根目录**：`../../data/ActiveView/`（默认基于源码仓库相对解析）
- **环境变量覆盖**：可通过环境变量 `ACTIVEVIEW_DATA_ROOT` 指定绝对路径。

数据目录布局：
```text
data/ActiveView/
├── datasets/
│   ├── babel/babel_v1.0_release/       # BABEL annotations (train, val, extra_train, extra_val)
│   └── amass/                          # AMASS 解压后的 mocap npz 文件
├── assets/
│   └── motions/raw/                    # motion_asset_manifest.json (标准化元数据清单)
├── cache/
│   ├── babel_selection/                # 动作筛选产物与候选清单
│   └── amass_download/                 # AMASS 下载状态与文件索引
├── logs/data_preparation/              # 数据准备日志
└── tmp/hf_downloads/                   # Hugging Face 下载临时归档
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

## 3. 标准推荐流水线 (Primary Recommended Workflow)

### 步骤 1: 运行 BABEL 动作筛选器
解析 BABEL 标注并生成完整候选池与第一轮 17 条 Motion Feasibility Set：
```bash
python -m tools.motion_assets.select_babel_elderly_actions
```
产物位于 `cache/babel_selection/`：
- `elderly_action_candidates.csv` / `.json`
- `feasibility_manifest.json` / `.csv`
- `summary.json`
- `required_amass_subdatasets_feasibility.txt`

---

### 步骤 2: 从 Hugging Face 选择性下载 AMASS 子数据集 (推荐主路径)
直接从 Hugging Face `tuguobin/AMASS` 镜像选择性下载 Feasibility Set 所需的 5 个子库（`BMLrub`, `CMU`, `EKUT`, `EyesJapanDataset`, `KIT`）并自动解压：
```bash
# 预检下载计划 (Dry-Run)
python -m tools.motion_assets.download_amass_hf --dry-run

# 执行实际下载与安全解压
python -m tools.motion_assets.download_amass_hf
```

---

### 步骤 3: 建立本地 AMASS 索引与 Schema 兼容性验证
扫描本地已解压的 AMASS 数据，建立 BABEL `feat_p` 到真实本地 NPZ 文件的映射，并校验 Habitat `MotionConverterSMPLX` 所需字段（支持 `explicit_root_orient` 与 `standard_amass` 两种模式）：
```bash
python -m tools.motion_assets.index_amass_files
```
- **退出状态**：17/17 全部通过返回 `exit code 0` (PASS)，存在缺失返回 `exit code 1` (INCOMPLETE)。
- **产物**：
  - `cache/amass_download/amass_file_index.json`
  - `cache/amass_download/habitat_motion_compatibility.csv`
  - `assets/motions/raw/motion_asset_manifest.json`

---

### 步骤 4: 运行单元测试套件
```bash
python -m unittest tools.motion_assets.tests.test_motion_assets
```

---

## 4. 备用下载路径 (Optional / Legacy Fallback)

### AMASS 官方 Portal 网页认证下载器
若需从官方 AMASS 站点下载，可使用：
```bash
# 需要在环境变量中设置 AMASS_EMAIL 和 AMASS_PASSWORD
python -m tools.motion_assets.download_amass_required --dry-run
python -m tools.motion_assets.download_amass_required --execute
```

---

## 5. 安全与数据保护承诺
1. **禁止第三方数据入库**：BABEL annotations 与 AMASS npz/tar.bz2 等数据只存放在外部数据目录，严禁提交进 Git 仓库；
2. **凭据安全**：代码中仅通过 `os.environ` 读取账号信息，禁止在任何文件、日志或 commit 中硬编码真实密码；
3. **机器无关路径**：所有生成的 manifest 均采用相对于数据根目录的相对路径，不包含特定开发机的固定绝对路径。
