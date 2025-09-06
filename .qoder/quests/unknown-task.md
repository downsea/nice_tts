# nice_tts Dataclass TypeError 修复设计

## 问题概述

在执行 `uv run nice-tts check-gpu` 命令时遇到 TypeError：

```
TypeError: non-default argument 'errors' follows default argument
```

错误发生在 `src/nice_tts/core/pipeline.py` 文件第48行的 `BatchResult` dataclass 定义中。

## 问题分析

### 根本原因

Python dataclass 有一个重要的限制：**没有默认值的字段不能跟在有默认值的字段后面**。

当前的 `BatchResult` dataclass 定义存在问题：

```python
@dataclass  
class BatchResult:
    """Result of processing a batch of files."""
    
    files_processed: List[FileResult]
    total_files: int
    successful_files: int
    failed_files: int
    total_processing_time: float = 0.0  # 有默认值
    errors: List[Exception]  # 没有默认值，但跟在有默认值的字段后面
```

### 错误触发链

1. 导入 `pipeline.py` 模块
2. Python 尝试处理 `@dataclass` 装饰器
3. 在 `_process_class` 函数中检查字段定义
4. 发现 `errors` 字段违反了 dataclass 字段排序规则
5. 抛出 TypeError

## 解决方案

### 方案一：为 errors 字段提供默认值（推荐）

将 `errors` 字段改为使用 `field(default_factory=list)` 提供默认值：

```python
from dataclasses import dataclass, field

@dataclass  
class BatchResult:
    """Result of processing a batch of files."""
    
    files_processed: List[FileResult]
    total_files: int
    successful_files: int
    failed_files: int
    total_processing_time: float = 0.0
    errors: List[Exception] = field(default_factory=list)
```

### 方案二：重新排序字段

将有默认值的字段移到最后：

```python
@dataclass  
class BatchResult:
    """Result of processing a batch of files."""
    
    files_processed: List[FileResult]
    total_files: int
    successful_files: int
    failed_files: int
    errors: List[Exception]
    total_processing_time: float = 0.0
```

## 技术实现

### 推荐实现（方案一）

**优势：**
- 保持字段逻辑顺序不变
- 提供合理的默认值，避免 None 检查
- 符合 Python dataclass 最佳实践
- 不影响现有代码调用

**实现步骤：**

1. 导入 `field` 函数：
```python
from dataclasses import dataclass, field
```

2. 修改 `BatchResult` 类：
```python
@dataclass  
class BatchResult:
    """Result of processing a batch of files."""
    
    files_processed: List[FileResult]
    total_files: int
    successful_files: int
    failed_files: int
    total_processing_time: float = 0.0
    errors: List[Exception] = field(default_factory=list)
```

### 代码兼容性检查

检查现有代码中对 `BatchResult` 的使用：

1. **创建实例时：** 当前代码在创建 `BatchResult` 时总是显式传递 `errors` 参数，因此兼容
2. **访问字段时：** 所有字段访问保持不变
3. **类型注解：** 保持不变

## 测试验证

### 验证步骤

1. 修复 dataclass 定义
2. 运行命令验证修复：
   ```bash
   uv run nice-tts check-gpu
   ```
3. 确认不再出现 TypeError
4. 验证核心功能正常工作

### 测试用例

```python
# 测试默认值
result = BatchResult(
    files_processed=[],
    total_files=0,
    successful_files=0,
    failed_files=0
    # errors 使用默认值 []
)
assert result.errors == []

# 测试显式传递
errors_list = [Exception("test")]
result = BatchResult(
    files_processed=[],
    total_files=0,
    successful_files=0,
    failed_files=0,
    errors=errors_list
)
assert result.errors == errors_list
```

## 风险评估

### 低风险
- 仅修改字段默认值定义
- 不影响现有 API 接口
- 向后兼容现有代码

### 无影响区域
- CLI 命令接口
- 其他 dataclass 定义
- 业务逻辑流程

## 实施建议

1. **立即修复**：这是一个阻塞性错误，需要立即修复
2. **使用方案一**：提供默认值比重排字段更安全
3. **测试验证**：修复后立即验证 `check-gpu` 命令
4. **代码审查**：检查其他 dataclass 定义是否有类似问题