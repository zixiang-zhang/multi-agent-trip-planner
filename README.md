# SmartTrip-基于 Multi-Agent 架构的智能旅行规划系统🌍

> 面向个性化旅行规划场景，设计并实现基于 Multi-Agent 架构的智能旅行规划系统。系统可根据用户输入的目的地、出行日期、交通方式、住宿偏好和兴趣标签，自动完成景点检索、天气分析、酒店推荐、预算估算与多日行程编排，并通过前端地图化展示结果，支持行程编辑与导出。

---

## ✨ 项目亮点

- 🤖 **多智能体协作架构** : 基于 LangGraph 构建 `Supervisor + 多角色 Agent` 协同流程，包含任务拆解、景点检索、天气分析、酒店推荐、行程规划等核心角色
- ⚡ **三路并发检索** : 景点、天气、酒店三个 Agent 并发执行，缩短端到端规划耗时
- 🗺️ **高德地图集成** : 通过 MCP (Model Context Protocol) 接入高德地图服务，支持 POI 搜索、POI 详情补全、周边检索、天气查询等真实工具调用
- 🧠 **真实工具调用 + 结构化输出** : Agent 不是直接“编结果”，而是基于真实 MCP 数据做筛选、规划，并通过 Pydantic 约束结构化输出
- 📅 **智能行程生成** : PlannerAgent 负责生成每日行程骨架，结合用户偏好自动规划多日安排
- 💰 **预算估计** : 自动汇总景点门票、住宿、餐饮、交通费用，生成预算明细
- 🍜 **真实餐饮补全** : 餐饮不是由 Planner 直接生成文本，而是由 MealService 基于真实周边检索补全
- 🖼️ **景点配图** : 集成 Unsplash API 为景点匹配高质量图片
- 🗺️ **可视化地图** : 前端交互式地图展示景点位置与每日路线
- 📄 **导出功能** : 支持导出行程为图片或 PDF 格式，方便保存和分享
- 📊 **可观测与评测** : 集成 LangSmith Trace 与离线评测脚本，持续统计约束满足率、MCP 命中率与平均耗时

---

## 🏗️ 技术栈

### 后端

```text
FastAPI + Pydantic              # Web 框架与数据验证
├── LangChain                   # LLM 应用框架
├── LangGraph                   # 多智能体工作流编排
├── OpenAI-Compatible API       # 大语言模型服务（支持 DeepSeek / Qwen / OpenAI 等兼容接口）
├── MCP (Model Context Protocol)# 工具调用协议
│   └── amap-mcp-server         # 高德地图 MCP 服务
├── Unsplash API                # 图片服务
└── Uvicorn                     # ASGI 服务器
```

**关键技术亮点：**
- **多智能体系统设计** - Supervisor 负责任务拆解，景点 / 天气 / 酒店 / 规划 Agent 各司其职
- **LangGraph 工作流编排** - 基于共享 State 在节点之间传递上下文，并支持并发检索
- **MCP 协议集成** - 通过标准化协议接入高德地图真实工具能力
- **共享异步客户端** - LLM 客户端、MCP Client、服务实例全局复用，提升吞吐与稳定性
- **结构化输出** - 利用 Pydantic 模型约束 LLM 输出，降低解析失败率
- **容错降级机制** - MCP 优先，异常时通过局部回退与后处理保障结果可返回
- **轻量骨架规划** - PlannerAgent 只生成每日骨架，避免大 JSON 输出导致的超时问题

### 前端

```text
Vue 3 + TypeScript              # 前端框架
├── Ant Design Vue              # UI 组件库
├── Vite                        # 构建工具
├── 高德地图 JS API             # 地图可视化
├── html2canvas + jsPDF         # 导出功能
└── Axios                       # HTTP 客户端
```

---

## 📂 项目结构

```text
multi-agent-trip-planner/
├── backend/                         # 后端服务（核心）
│   ├── app/
│   │   ├── agents/                  # 多智能体与后处理模块
│   │   │   └── langgraph_agents.py  # Supervisor / Attraction / Weather / Hotel / Planner / Validator / MealService
│   │   ├── api/                     # API 路由
│   │   │   ├── routes/
│   │   │   │   ├── trip.py          # 行程规划 API
│   │   │   │   ├── poi.py           # POI 与图片 API
│   │   │   │   └── map.py           # 地图相关 API
│   │   │   └── main.py              # FastAPI 应用入口
│   │   ├── models/
│   │   │   └── schemas.py           # Pydantic 数据模型
│   │   ├── services/
│   │   │   ├── llm_service.py       # LLM 服务封装
│   │   │   ├── amap_service.py      # 高德地图服务封装
│   │   │   └── unsplash_service.py  # Unsplash 图片服务
│   │   ├── tools/
│   │   │   └── amap_mcp_tools.py    # 共享异步 AMap MCP Client
│   │   ├── workflows/               # LangGraph 工作流实现
│   │   │   ├── trip_planner_graph.py
│   │   │   └── trip_planner_state.py
│   │   ├── config.py                # 配置管理
│   │   └── logging_config.py        # request_id / run_id 日志上下文
│   ├── evals/                       # 离线评测脚本与报告
│   ├── logs/                        # 运行日志（自动创建）
│   ├── .env.example                 # 环境变量模板
│   └── requirements.txt             # Python 依赖
├── frontend/                        # 前端界面（演示）
│   ├── src/
│   │   ├── services/                # API 调用封装
│   │   ├── views/                   # 页面组件
│   │   ├── types/                   # TypeScript 类型定义
│   │   └── main.ts
│   └── package.json
└── README.md
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- 高德地图 API 密钥（后端 Web 服务 Key + 前端 JS Key）
- LLM API 密钥（OpenAI 或其他兼容 OpenAI 的 LLM）
- Unsplash API Key（可选，用于景点配图）

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/multi-agent-trip-planner.git
```

### 2. 后端配置与启动

#### 2.1 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

#### 2.2 配置环境变量

复制 `.env.example` 为 `.env` 并填写必要配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 模型配置（支持 OpenAI 兼容接口）
LLM_MODEL_ID="deepseek-chat"
LLM_API_KEY="your-api-key"
LLM_BASE_URL="https://api.deepseek.com/v1"
LLM_TIMEOUT=60

# 高德地图 API（必填）
AMAP_API_KEY="your-amap-api-key"

# Unsplash API（可选，用于景点配图）
UNSPLASH_ACCESS_KEY="your-unsplash-access-key"
UNSPLASH_SECRET_KEY="your-unsplash-secret-key"

# LangSmith / LangChain Trace（可选）
LANGCHAIN_TRACING=true
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="multi-agent-trip-planner"

# 并发与日志（可选）
LLM_LIGHT_MAX_CONCURRENCY=3
LLM_HEAVY_MAX_CONCURRENCY=1
AMAP_MCP_MAX_CONCURRENCY=3
LOG_LEVEL=INFO
```

**获取 API Key：**
- 高德地图：https://lbs.amap.com/
- DeepSeek：https://www.deepseek.com/
- Unsplash：https://unsplash.com/developers
- LangSmith：https://smith.langchain.com/

#### 2.3 启动后端服务

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

服务将运行在 `http://localhost:8000`

访问 API 文档：`http://localhost:8000/docs`

### 3. 前端配置与启动

#### 3.1 安装依赖

```bash
cd frontend
npm install
```

#### 3.2 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 后端 API 地址
VITE_API_BASE_URL=http://localhost:8000

# 高德地图 Web API Key（可选）
VITE_AMAP_WEB_KEY=your-amap-web-key

# 高德地图 Web 端 JS API Key
VITE_AMAP_WEB_JS_KEY=your-amap-web-js-key
```

#### 3.3 启动开发服务器

```bash
npm run dev
```

打开浏览器访问 `http://localhost:5173`

---

## 📖 使用说明

### 基本流程

1. **填写行程需求**
   - 选择目的地城市
   - 设置出发/返回日期
   - 选择交通方式和住宿偏好
   - 勾选旅行偏好（历史文化、自然风光、美食等）
   - 输入额外要求（可选）

2. **生成行程计划**
   - 点击“开始规划我的旅行”
   - 系统将调用多智能体协作生成行程
   - 当前主流程为：`Supervisor -> 景点/天气/酒店并发检索 -> Planner 骨架规划 -> Validator 修正 -> MealService 补餐饮`
   - 预计耗时约 30-60 秒（取决于行程天数和模型 / 工具响应）

3. **查看与编辑**
   - 查看每日详细行程
   - 地图展示景点位置与路线
   - 支持编辑景点顺序和信息
   - 查看预算明细和天气信息

4. **导出行程**
   - 导出为图片（PNG）
   - 导出为 PDF 文档

---

### 调试技巧

1. **启用 LangSmith 追踪**
   ```env
   LANGCHAIN_TRACING=true
   ```
   查看 Agent 调用链路：https://smith.langchain.com/

2. **查看后端日志**
   - `backend/logs/backend.out.log`
   - `backend/logs/backend.err.log`

3. **前端调试**
   - 打开浏览器开发者工具
   - Network 面板查看 API 请求
   - Console 面板查看错误信息

4. **运行离线评测**
   ```bash
   cd backend
   python evals/eval_runner.py --limit 5 --no-gate
   ```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.