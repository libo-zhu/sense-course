# 第三方来源与复现改动

1. **MSH-Net**，Wei等，IEEE TGRS 61，2023，DOI 10.1109/TGRS.2023.3265650。作者代码 https://github.com/shicaiwei123/TGRS-MSHNet ，commit `c1b5f0409ca5e545b4f767070caab95fdd6fdc63`，MIT，许可证保留在upstream/LICENSE。upstream/loss/kd/jda.py的距离计算做内存优化、带宽保护和reshape修正，原JDA文件保留在references/original_jda.py以供数值/梯度核对。
2. **S2FL数据**，Hong等，ISPRS JP&RS 178，2021，68–80，DOI 10.1016/j.isprsjprs.2021.05.011。数据入口 https://github.com/danfenghong/ISPRS_S2FL 。代码仓库不重分发原始大数据或原论文PDF，获取脚本保留公开来源与哈希清单。数据不因本仓库MIT代码许可而自动获得MIT许可。
3. **随机模态丢弃思想**，Neverova、Wolf、Taylor、Nebout，ModDrop，arXiv:1501.00102v2。当前moddrop是课程适配：相同教师结构，每样本50%概率随机丢一种模态，不是原手势识别系统的完整复现，也不声称发明ModDrop。
4. **IEEEtran**类用于报告排版，来源为TeX Live所附官方IEEEtran类，保持原文件及其许可头部。模板说明 https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/ 。
5. 原论文与发布代码并非处处一致：本文按论文表VI对照数值，并披露训练样本数、随机翻转、融合参数化和模型选择的本地差异。未删改阴性结果，也未把流程测试作为准确率复现。

自动化工具用于代码适配、图表与材料整理；数据来自实际保存的预测，参考文献与方法来源按上文列出。署名成员按交付责任核对并承担内容责任。
