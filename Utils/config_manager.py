import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import re
from dataclasses import dataclass
from typing import List, Dict, Union


class Config:
    def __init__(self, config_path):
        # 加载环境变量
        load_dotenv()

        # 加载原始 YAML 配置
        with open(config_path, encoding="utf-8") as f:
            self.raw_config = yaml.safe_load(f)

        # 先初始化 resolved_config 为空字典
        self.resolved_config = {}

        # 递归解析配置
        self.resolved_config = self._resolve_config(self.raw_config)

        # 转换为 Path 对象
        self._init_paths()

    def _init_paths(self):
        """将解析后的配置转换为 Path 对象"""
        self.base = Path(self.resolved_config['base']['root'])
        self.data_root = Path(self.resolved_config['base']['data'])
        self.output = Path(self.resolved_config['base']['output'])

        self.input = {
            key: Path(value)
            for key, value in self.resolved_config['input'].items()
        }
        self.output_section = {
            key: Path(value)
            for key, value in self.resolved_config['output'].items()
        }

    def _resolve_config(self, config):
        """递归解析配置中的变量"""
        resolved = {}
        for key, value in config.items():
            if isinstance(value, dict):
                resolved[key] = self._resolve_config(value)
                self.resolved_config[key] = self._resolve_config(value)
            elif isinstance(value, str):
                resolved[key] = self._resolve_string(value)
                self.resolved_config[key] = self._resolve_string(value)
            else:
                resolved[key] = value
                self.resolved_config[key] = value
        return resolved

    def _resolve_string(self, s):
        """解析字符串中的嵌套变量"""
        # 先替换环境变量
        s = os.path.expandvars(s)

        # 替换 YAML 内部变量（如 ${base.data_root}）
        while True:
            matches = re.findall(r'\${([^}]+)}', s)
            if not matches:
                break
            for match in matches:
                # 从已解析的配置中获取值
                keys = match.split('.')
                current = self.resolved_config
                try:
                    for k in keys:
                        current = current[k]
                    s = s.replace(f'${{{match}}}', str(current))
                except KeyError:
                    raise ValueError(f"未定义的变量: ${{{match}}}")
        return s

    def get(self, *keys):
        """安全获取嵌套配置"""
        current = self.resolved_config
        for key in keys:
            current = current.get(key, {})
        return Path(current) if isinstance(current, str) else current
