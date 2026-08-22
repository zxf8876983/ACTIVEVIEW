"""
动作资产与标签管理器 —— motion_manager.py
======================================

职责：
    1. 统一管理 v10.0 支持的 6 大动作类别 (standing, walking, sitting, bending, reaching, falling)；
    2. 发现并加载 data/ActiveView/assets/motions/converted/ 目录下的标准 Habitat .pkl 动作文件；
    3. 提供 MotionPlayer 实例化与动作元数据检索接口；
    4. 严格禁止在此处实现任何动作识别分类器或 ST-GCN 网络。
"""

import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.core.types import ActionClassV10

logger = logging.getLogger(__name__)


class MotionManager:
    """人体动作资产与类别索引管理器。"""

    def __init__(self, motions_dir: Optional[Union[str, Path]] = None):
        if motions_dir:
            self.motions_dir = Path(motions_dir)
        else:
            self.motions_dir = get_data_root() / "assets" / "motions" / "converted"

        self._motion_index: Dict[str, Path] = {}
        self._class_index: Dict[ActionClassV10, List[str]] = {c: [] for c in ActionClassV10}
        self.scan_motions()

    def scan_motions(self) -> Dict[str, Path]:
        """扫描并建立动作资产索引。"""
        self._motion_index.clear()
        for c in ActionClassV10:
            self._class_index[c] = []

        if not self.motions_dir.exists():
            logger.warning("Motions directory does not exist: %s", self.motions_dir)
            return self._motion_index

        pkl_files = sorted(self.motions_dir.glob("*.pkl"))
        for pkl_p in pkl_files:
            m_id = pkl_p.stem
            self._motion_index[m_id] = pkl_p

            # 匹配动作类别
            try:
                act_class = ActionClassV10.from_str(m_id)
                self._class_index[act_class].append(m_id)
            except ValueError:
                logger.debug("Could not match action class for file: %s", pkl_p.name)

        logger.info(
            "MotionManager indexed %d motion files across %d action classes from: %s",
            len(self._motion_index),
            len([k for k, v in self._class_index.items() if len(v) > 0]),
            self.motions_dir,
        )
        return self._motion_index

    def list_available_motions(self) -> List[str]:
        """列出所有可用动作 ID。"""
        return list(self._motion_index.keys())

    def get_motions_by_class(self, action_class: Union[str, ActionClassV10]) -> List[str]:
        """按动作类别获取所有动作 ID。"""
        if isinstance(action_class, str):
            act_enum = ActionClassV10.from_str(action_class)
        else:
            act_enum = action_class
        return list(self._class_index.get(act_enum, []))

    def get_motion_player(self, motion_id: str, playback_fps: float = 30.0) -> MotionPlayer:
        """根据 motion_id 实例化 MotionPlayer 播放器。"""
        if motion_id not in self._motion_index:
            # 尝试直接按文件名查找
            p = self.motions_dir / f"{motion_id}.pkl"
            if not p.exists():
                raise FileNotFoundError(f"Motion ID '{motion_id}' not found in {self.motions_dir}")
            self._motion_index[motion_id] = p

        pkl_p = self._motion_index[motion_id]
        return MotionPlayer(pkl_p, playback_fps=playback_fps)
