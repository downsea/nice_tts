# Nice-TTS 项目重构设计

## 概述

Nice-TTS 是一个基于 AI 的命令行工具，专注于音频转录和会议摘要生成，特别针对中文语言处理进行优化。本重构设计旨在提升代码架构、可维护性、扩展性和用户体验。

### 当前架构问题分析

1. **单体式模块设计**：transcription.py 和 llm.py 职责过于集中，缺乏模块化
2. **配置管理混乱**：环境变量加载逻辑分散，缺乏统一配置管理
3. **错误处理不一致**：不同模块的错误处理方式不统一
4. **测试覆盖率低**：缺乏系统性单元测试
5. **扩展性限制**：难以添加新的转录引擎或 LLM 提供商
6. **CLI 逻辑耦合**：命令行逻辑与业务逻辑耦合过紧

## 重构目标

- 提升代码模块化和可维护性
- 增强系统扩展性和配置管理
- 统一错误处理和日志记录
- 完善测试覆盖率
- 改善用户体验和性能

## 架构设计

### 整体架构图

```mermaid
graph TB
    subgraph "CLI Layer"
        CLI[CLI Interface]
        Commands[Commands Handler]
    end
    
    subgraph "Service Layer"
        PS[Pipeline Service]
        CS[Config Service]
        LS[Logging Service]
    end
    
    subgraph "Business Layer"
        TE[Transcription Engine]
        LLM[LLM Engine]
        FileM[File Manager]
    end
    
    subgraph "Provider Layer"
        WhisperP[Whisper Provider]
        OpenAIP[OpenAI Provider]
        ClaudeP[Claude Provider]
    end
    
    subgraph "Infrastructure"
        Storage[File Storage]
        Cache[Model Cache]
        Config[Configuration]
    end
    
    CLI --> Commands
    Commands --> PS
    PS --> TE
    PS --> LLM
    PS --> FileM
    PS --> CS
    PS --> LS
    
    TE --> WhisperP
    LLM --> OpenAIP
    LLM --> ClaudeP
    
    WhisperP --> Cache
    FileM --> Storage
    CS --> Config
```

### 核心组件设计

#### 1. 配置管理系统

```mermaid
classDiagram
    class ConfigService {
        +load_config() dict
        +validate_config(config) bool
        +get_transcription_config() TranscriptionConfig
        +get_llm_config() LLMConfig
        -_load_env_files() dict
        -_merge_configs(global, local) dict
    }
    
    class TranscriptionConfig {
        +model_name: str
        +language: str
        +device: str
        +cache_dir: str
    }
    
    class LLMConfig {
        +api_key: str
        +base_url: str
        +model_name: str
        +max_tokens: int
        +temperature: float
    }
    
    ConfigService --> TranscriptionConfig
    ConfigService --> LLMConfig
```

#### 2. 转录引擎抽象

```mermaid
classDiagram
    class TranscriptionEngine {
        <<interface>>
        +transcribe(audio_path, config) TranscriptionResult
        +supports_language(language) bool
        +get_supported_formats() list
    }
    
    class WhisperEngine {
        +transcribe(audio_path, config) TranscriptionResult
        +supports_language(language) bool
        +get_supported_formats() list
        -_load_model(model_name) Model
        -_detect_device() str
    }
    
    class TranscriptionResult {
        +text: str
        +language: str
        +confidence: float
        +processing_time: float
    }
    
    TranscriptionEngine <|-- WhisperEngine
    WhisperEngine --> TranscriptionResult
```

#### 3. LLM 引擎抽象

```mermaid
classDiagram
    class LLMEngine {
        <<interface>>
        +refine_text(text, config) RefinementResult
        +summarize_text(text, config) SummaryResult
        +count_tokens(text) int
    }
    
    class OpenAIProvider {
        +refine_text(text, config) RefinementResult
        +summarize_text(text, config) SummaryResult
        +count_tokens(text) int
        -_create_client(config) Client
        -_chunk_text(text, max_tokens) list
    }
    
    class ClaudeProvider {
        +refine_text(text, config) RefinementResult
        +summarize_text(text, config) SummaryResult
        +count_tokens(text) int
        -_create_client(config) Client
    }
    
    class RefinementResult {
        +refined_text: str
        +chunks_processed: int
        +tokens_used: int
    }
    
    class SummaryResult {
        +summary_markdown: str
        +metadata: dict
        +tokens_used: int
    }
    
    LLMEngine <|-- OpenAIProvider
    LLMEngine <|-- ClaudeProvider
    OpenAIProvider --> RefinementResult
    OpenAIProvider --> SummaryResult
    ClaudeProvider --> RefinementResult
    ClaudeProvider --> SummaryResult
```

#### 4. 处理流水线

```mermaid
graph LR
    subgraph "Pipeline Service"
        A[Input Validation] --> B[File Discovery]
        B --> C[Processing Queue]
        C --> D[Stage 1: Transcription]
        D --> E[Stage 2: Refinement]
        E --> F[Stage 3: Summarization]
        F --> G[Output Generation]
    end
    
    subgraph "State Management"
        H[Progress Tracking]
        I[Skip Logic]
        J[Error Recovery]
    end
    
    C --> H
    D --> I
    E --> I
    F --> I
    H --> J
```

## 新模块结构

### 目录结构设计

```
src/nice_tts/
├── __init__.py
├── main.py                     # CLI 入口
├── cli/                        # CLI 层
│   ├── __init__.py
│   ├── commands.py             # 命令处理器
│   └── validators.py           # 参数验证器
├── core/                       # 核心业务层
│   ├── __init__.py
│   ├── pipeline.py             # 处理流水线
│   ├── config.py               # 配置管理
│   └── exceptions.py           # 异常定义
├── engines/                    # 引擎抽象层
│   ├── __init__.py
│   ├── transcription/
│   │   ├── __init__.py
│   │   ├── base.py             # 转录引擎基类
│   │   └── whisper.py          # Whisper 实现
│   └── llm/
│       ├── __init__.py
│       ├── base.py             # LLM 引擎基类
│       ├── openai_provider.py  # OpenAI 实现
│       └── claude_provider.py  # Claude 实现
├── utils/                      # 工具层
│   ├── __init__.py
│   ├── file_manager.py         # 文件管理
│   ├── logger.py               # 日志管理
│   └── progress.py             # 进度显示
└── tests/                      # 测试
    ├── __init__.py
    ├── unit/
    ├── integration/
    └── fixtures/
```

### 配置系统设计

```mermaid
graph TD
    subgraph "Configuration Sources"
        A[Default Config]
        B[Global ~/.env]
        C[Local .env]
        D[CLI Arguments]
        E[Environment Variables]
    end
    
    subgraph "Configuration Merger"
        F[Config Loader]
        G[Validation Layer]
        H[Type Conversion]
    end
    
    subgraph "Configuration Objects"
        I[AppConfig]
        J[TranscriptionConfig]
        K[LLMConfig]
        L[OutputConfig]
    end
    
    A --> F
    B --> F
    C --> F
    D --> F
    E --> F
    
    F --> G
    G --> H
    H --> I
    I --> J
    I --> K
    I --> L
```

## 数据流设计

### 处理流水线数据流

```mermaid
sequenceDiagram
    participant CLI
    participant Pipeline
    participant TranscriptionEngine
    participant LLMEngine
    participant FileManager
    participant Logger
    
    CLI->>Pipeline: process_files(input_files, config)
    Pipeline->>Logger: log_start(batch_info)
    
    loop For each audio file
        Pipeline->>FileManager: check_existing_outputs(file)
        FileManager-->>Pipeline: existing_stages
        
        alt Transcription needed
            Pipeline->>TranscriptionEngine: transcribe(audio_file)
            TranscriptionEngine-->>Pipeline: TranscriptionResult
            Pipeline->>FileManager: save_transcription(result)
        end
        
        alt Refinement needed
            Pipeline->>LLMEngine: refine_text(transcription)
            LLMEngine-->>Pipeline: RefinementResult
            Pipeline->>FileManager: save_refinement(result)
        end
        
        alt Summarization needed
            Pipeline->>LLMEngine: summarize_text(refined_text)
            LLMEngine-->>Pipeline: SummaryResult
            Pipeline->>FileManager: save_summary(result)
        end
        
        Pipeline->>Logger: log_file_complete(file_status)
    end
    
    Pipeline->>Logger: log_batch_complete(batch_status)
    Pipeline-->>CLI: BatchResult
```

### 错误处理流程

```mermaid
graph TD
    A[Operation Start] --> B{Try Execute}
    B -->|Success| C[Log Success]
    B -->|Error| D[Catch Exception]
    
    D --> E{Error Type}
    E -->|FileNotFound| F[Log File Error]
    E -->|NetworkError| G[Log Network Error]
    E -->|ConfigError| H[Log Config Error]
    E -->|APIError| I[Log API Error]
    
    F --> J{Retry Logic}
    G --> J
    H --> K[Fatal Error]
    I --> J
    
    J -->|Can Retry| L[Retry Operation]
    J -->|Cannot Retry| M[Skip File]
    
    L --> B
    M --> N[Continue Next]
    K --> O[Exit Process]
    
    C --> P[Continue]
    N --> P
    P --> Q[Operation Complete]
```

## 接口设计

### CLI 接口优化

```python
# 改进的 CLI 命令结构
@app.command()
def process(
    input_path: Path,
    output_dir: Path = Path("output"),
    config_file: Optional[Path] = None,
    transcription_model: str = "large-v3-turbo",
    language: str = "zh",
    llm_provider: str = "openai",
    force_reprocess: bool = False,
    parallel_jobs: int = 1,
    verbose: bool = False
) -> None:
    """处理音频文件并生成转录和摘要"""
    
@app.command()
def config(
    action: str = typer.Argument(..., help="Action: show, validate, init"),
    config_file: Optional[Path] = None
) -> None:
    """配置管理命令"""
    
@app.command() 
def list_models() -> None:
    """列出可用的转录和 LLM 模型"""
```

### 服务接口

```python
# 核心服务接口
class ProcessingPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.transcription_engine = self._create_transcription_engine()
        self.llm_engine = self._create_llm_engine()
        self.file_manager = FileManager(config.output)
        self.logger = Logger(config.logging)
    
    async def process_batch(
        self, 
        input_files: List[Path], 
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """批量处理音频文件"""
        
    async def process_single(
        self, 
        input_file: Path,
        progress_callback: Optional[Callable] = None
    ) -> FileResult:
        """处理单个音频文件"""
```

## 测试策略

### 测试架构

```mermaid
graph TB
    subgraph "测试层级"
        A[单元测试 Unit Tests]
        B[集成测试 Integration Tests] 
        C[端到端测试 E2E Tests]
        D[性能测试 Performance Tests]
    end
    
    subgraph "测试覆盖"
        E[配置管理 Config Management]
        F[转录引擎 Transcription Engines]
        G[LLM 引擎 LLM Engines]
        H[文件管理 File Management]
        I[CLI 接口 CLI Interface]
    end
    
    A --> E
    A --> F
    A --> G
    A --> H
    B --> F
    B --> G  
    B --> I
    C --> I
```

### 测试用例设计

#### 单元测试覆盖

1. **配置管理测试**
   - 配置文件加载和合并
   - 环境变量处理
   - 配置验证逻辑

2. **转录引擎测试**  
   - 模型加载和缓存
   - 音频格式支持验证
   - GPU/CPU 设备检测

3. **LLM 引擎测试**
   - Token 计数准确性
   - 文本分块算法  
   - API 调用处理

4. **文件管理测试**
   - 文件路径处理
   - 输出文件生成
   - 跳过逻辑验证

## 性能优化

### 并发处理设计

```mermaid
graph LR
    subgraph "并发策略"
        A[文件级并发] --> B[I/O 密集型操作]
        C[模型级并发] --> D[GPU 资源管理]  
        E[网络级并发] --> F[LLM API 调用]
    end
    
    subgraph "资源管理"
        G[连接池管理]
        H[模型缓存策略]
        I[内存使用监控]
    end
    
    B --> G
    D --> H
    F --> G
    H --> I
```

### 缓存策略

1. **模型缓存**：Whisper 模型本地缓存和版本管理
2. **结果缓存**：处理结果的智能缓存和失效策略  
3. **配置缓存**：运行时配置对象缓存

### 内存优化

1. **流式处理**：大文件的分块流式处理
2. **懒加载**：按需加载模型和资源
3. **垃圾回收**：及时释放不需要的对象

## 向后兼容性

### 迁移策略

1. **配置文件迁移**
   - 自动检测旧版本配置格式
   - 提供配置迁移工具
   - 保持向后兼容的环境变量名

2. **CLI 接口兼容**
   - 保持现有命令行参数
   - 添加新功能时使用可选参数
   - 提供弃用警告机制

3. **输出格式兼容**  
   - 维持现有输出文件格式
   - 新功能通过可选参数启用

## 扩展性设计

### 插件架构

```mermaid
classDiagram
    class EngineRegistry {
        +register_transcription_engine(name, engine_class)
        +register_llm_engine(name, engine_class)
        +get_transcription_engine(name) TranscriptionEngine
        +get_llm_engine(name) LLMEngine
        +list_available_engines() dict
    }
    
    class PluginManager {
        +load_plugins(plugin_dir) 
        +discover_plugins() list
        +validate_plugin(plugin) bool
    }
    
    EngineRegistry --> PluginManager
```

### 新引擎集成

1. **转录引擎扩展**
   - Azure Speech Services
   - Google Speech-to-Text  
   - 自定义本地模型

2. **LLM 引擎扩展**
   - 智谱 ChatGLM
   - 百度文心一言
   - 本地 Ollama 模型

## 部署和分发

### 打包策略

```mermaid
graph TB
    subgraph "分发方式"
        A[PyPI Package]
        B[Docker Image] 
        C[Standalone Executable]
        D[Conda Package]
    end
    
    subgraph "依赖管理"
        E[Core Dependencies]
        F[Optional Dependencies] 
        G[System Dependencies]
    end
    
    A --> E
    B --> E
    B --> F
    B --> G
    C --> E
    C --> G
    D --> E
    D --> F
```

### 配置管理

1. **多环境配置**：开发、测试、生产环境配置分离
2. **默认配置**：合理的默认值减少配置负担  
3. **配置验证**：启动时配置完整性检查

## 监控和日志

### 日志系统设计

```mermaid
graph LR
    subgraph "日志级别"
        A[DEBUG] --> B[INFO]
        B --> C[WARNING] 
        C --> D[ERROR]
        D --> E[CRITICAL]
    end
    
    subgraph "输出目标"
        F[控制台输出]
        G[文件日志]
        H[结构化日志]
    end
    
    subgraph "日志内容"
        I[处理进度]
        J[性能指标]
        K[错误信息] 
        L[配置状态]
    end
    
    B --> F
    C --> F
    D --> F
    E --> F
    
    A --> G
    B --> G
    C --> G
    D --> G
    E --> G
    
    I --> H
    J --> H
    K --> H
    L --> H
```

### 监控指标

1. **性能指标**
   - 文件处理速度  
   - API 响应时间
   - 内存和 CPU 使用率

2. **业务指标**
   - 处理成功率
   - 错误分布统计
   - 模型使用情况

3. **系统指标**
   - 磁盘空间使用
   - 网络请求状态
   - 配置加载状态