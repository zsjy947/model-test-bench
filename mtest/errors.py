"""异常体系。"""


class MTestError(Exception):
    """mtest 基础异常。"""


class ConfigError(MTestError):
    """配置文件缺失 / 格式错误 / 校验不通过。"""


class ServeError(MTestError):
    """服务启动 / 停止 / 诊断过程中的错误。"""


class ServeTimeoutError(ServeError):
    """健康检查在 startup_timeout 内未就绪。"""


class NpuMonitorError(MTestError):
    """NPU 采样解析 / 执行错误（采样失败会降级，通常不抛出到顶层）。"""


class SuiteError(MTestError):
    """测试套件执行错误。"""
