from dataclasses import dataclass


@dataclass
class DMPFold2Config:
    width: int = 512
    cwidth: int = 128