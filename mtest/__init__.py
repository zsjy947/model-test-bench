"""model-test-bench (mtest): 一键模型测试平台。

面向 Ascend 910B + vllm(-ascend) 的性能压测 / 功能冒烟 / 长序列 / embedding / OCR
测试闭环。核心引擎以库形式提供（pipeline / suites / serve），CLI 见 ``mtest.cli``。
"""

__version__ = "0.1.0"
