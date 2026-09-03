"""文件用途：暴露项目源码、运行时数据和 Habitat 资产路径解析函数。

主要输入：环境变量或仓库相对路径。
主要输出：规范化的路径对象。
项目角色：所有数据和脚本入口共享的路径基础层。
"""

from .paths import (
    get_data_root,
    get_habitat_data_root,
    get_humanoid_asset_root,
    get_humanoid_urdf_path,
    get_repo_root,
    get_experiments_root,
    get_stage_experiments_root,
    get_stage_experiment_runtime_root,
)

__all__ = [
    "get_data_root",
    "get_habitat_data_root",
    "get_humanoid_asset_root",
    "get_humanoid_urdf_path",
    "get_repo_root",
    "get_experiments_root",
    "get_stage_experiments_root",
    "get_stage_experiment_runtime_root",
]
