"""内容寻址产物存储:文件以 sha256 命名,原子写入,读取可校验。"""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "_tmp").mkdir(exist_ok=True)

    def path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha

    def verify(self, sha: str) -> bool:
        """文件存在且内容 hash 与名字一致。"""
        p = self.path(sha)
        try:
            return p.is_file() and sha256_file(p) == sha
        except OSError:
            return False

    def put_file(self, src: str | Path) -> tuple[str, int]:
        """把文件放进存储,返回 (sha256, size)。

        先复制到同一文件系统下的临时文件,再 os.replace 原子改名:
        进程在任意时刻崩溃,存储里要么没有该文件,要么是完整文件。
        """
        sha = sha256_file(src)
        size = os.path.getsize(src)
        dest = self.path(sha)
        if dest.is_file() and self.verify(sha):
            return sha, size
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.root / "_tmp" / f"{uuid.uuid4().hex}.part"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        return sha, size
