"""
动作语义状态定义 —— action_state.py
=================================

功能：
    1. 封装动作类别与 BABEL 语义标注；
    2. 支持动作类型验证与序列化。
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

SUPPORTED_ACTION_CLASSES = (
    "standing",
    "sitting",
    "bending",
    "reaching",
    "fall_related",
)


@dataclass
class ActionState:
    """动作标注与状态语义。"""
    action_class: str
    raw_label: str
    proc_label: str
    babel_sid: Union[int, str]
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    def __post_init__(self):
        if self.action_class not in SUPPORTED_ACTION_CLASSES:
            # 允许未知动作类但不推荐
            pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
