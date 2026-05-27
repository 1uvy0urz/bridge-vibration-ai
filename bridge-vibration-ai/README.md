# 桥梁振动检测 AI 科研平台

这是一个基于 Python + Streamlit 的桥梁振动检测网页系统。平台支持上传手机加速度传感器导出的 CSV、XLS、XLSX 文件，自动完成加速度数据读取、FFT 频谱分析、主频识别、AI 工程检测结论生成，并支持导出专业 PDF 检测报告。

## 功能介绍

- 上传 CSV / XLS / XLSX 振动数据
- 自动识别时间列和加速度列
- 支持三轴合加速度分析
- 自动绘制时域图和频域图
- 使用 Plotly 提供缩放、hover、深色交互图表
- 自动识别主频
- 基于真实 FFT 结果生成 AI 工程检测结论
- 输出风险等级：安全、注意、风险
- 导出包含图表和结论的 PDF 检测报告
- 深色科技风、毛玻璃、蓝紫渐变仪表盘 UI

## 项目结构

```text
bridge-vibration-ai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── services/
│   ├── __init__.py
│   └── bridge_ai.py
└── .github/
    └── workflows/
        └── streamlit-check.yml
```

## 安装方式

进入项目目录：

```bash
cd C:\Users\user\bridge-vibration-ai
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果你的电脑提示找不到 `pip` 或 `streamlit`，请先安装 Python 3.10 或更新版本，并勾选 `Add Python to PATH`。

## 本地运行方式

```bash
streamlit run app.py
```

启动后浏览器访问：

```text
http://localhost:8501
```

## 数据格式

推荐表格至少包含时间列和一个加速度列：

```csv
time,acc_x,acc_y,acc_z
0.00,0.02,0.01,9.81
0.01,0.04,0.03,9.84
0.02,0.01,0.02,9.79
```

常见可识别列名：

- 时间列：`time`、`timestamp`、`seconds`、`时间`
- 加速度列：`acc_x`、`acc_y`、`acc_z`、`acceleration`、`加速度`

如果没有时间列，需要在网页左侧输入正确采样率。

## GitHub 上传方式

1. 打开 [GitHub](https://github.com/) 并登录。
2. 点击右上角 `+`，选择 `New repository`。
3. 仓库名建议填写：

```text
bridge-vibration-ai
```

4. 不要勾选自动创建 README，因为本项目已经有 README。
5. 创建仓库后，在本地项目目录运行：

```bash
git init
git add .
git commit -m "init bridge vibration ai platform"
git branch -M main
git remote add origin https://github.com/你的用户名/bridge-vibration-ai.git
git push -u origin main
```

把 `你的用户名` 替换成你的 GitHub 用户名。

## Streamlit Cloud 部署方式

1. 打开 [Streamlit Community Cloud](https://streamlit.io/cloud)。
2. 使用 GitHub 账号登录。
3. 点击 `Create app` 或 `New app`。
4. 选择刚上传的 GitHub 仓库：

```text
bridge-vibration-ai
```

5. Branch 选择：

```text
main
```

6. Main file path 填写：

```text
app.py
```

7. 点击 `Deploy`。
8. 等待安装依赖并启动应用。

部署成功后，Streamlit Cloud 会自动生成一个公网网址，通常类似：

```text
https://你的应用名.streamlit.app
```

这个网址就是可以分享给别人访问的公网网站。

## GitHub Actions 自动检查

项目已经包含：

```text
.github/workflows/streamlit-check.yml
```

每次 push 到 GitHub 后，它会自动：

- 安装 Python
- 安装 `requirements.txt`
- 检查 `app.py` 和 AI 服务代码是否能正常编译

## 重要说明

本系统适合教学、科研展示、初步数据分析和原型验证。正式桥梁结构安全评估需要结合专业传感器、现场工况、结构图纸、长期监测数据和工程规范。
