<div align="center">

<h1>RWKV-RM-Copilot</h1>

[![Code License](https://img.shields.io/badge/LICENSE-Apache2.0-green.svg?style=for-the-badge)](https://github.com/Ourboros-Alignment-Team/RWKV-Development-Tools/tree/main/LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-yellow.svg?style=for-the-badge)](https://img.shields.io/badge/python-3.11-yellow.svg?style=for-the-badge)
[![QQ Group](https://img.shields.io/badge/qq%20group-873610818-blue?style=for-the-badge)](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=zcGtQcxps3ZEtGwV0-qdHFF2RULZOnQ4)

</div>

`RWKV-RM-Copilot` 是面向 `RWKV-Development-Tools` 在线学习的强化学习奖励模型。

## 使用方法
创建虚拟环境：
```bash
conda create [训练环境名] python=3.11.3
```


安装后端依赖：
```bash
pip install -r requirements.txt
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
```

## 预训练
在`train_config.json`中配置读取预训练模型的路径，然后运行：
```bash
deepspeed --master_port [使用的端口] --num_gpus 1 train_model.py
```
或
```bash
deepspeed ---master_port [使用的端口] --include localhost:[你的gpu] train_model.py
```
在不同进程中使用同一端口会报错。

## 简单测试
*如果你使用gradio5.4.0，你就不能使用deepspeed0.9.3😅。*

创建一个测试环境：
```bash
conda create [测试环境名] --clone [训练环境名]
pip install gradio==5.4.0
```

配置`eval_config.json`，然后运行：
```bash
python evaluate.py
```

## 启动在线学习服务
```bash
deepspeed --num_gpus 1 online_api.py
```