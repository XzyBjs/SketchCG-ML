## 环境配置
17年的tensorflow代码和默认的cuda不匹配，需要[and-cuda]重新安装
新版本tf对应的kearas3已经没有tf.nn.rnn_cell.RNNCell接口，需要降级到2.15.1，和keras2

## 数据集
有多种格式，且没有说明文档说明一个数据点的vector代表什么意思，通过draw_three进行调试和阅读，发现
sketchMLP使用的是[x, y, down, up]，而sketchrnn用的是[\delta x , \delta y, up]
用一个脚本把sketchMLP的数据集转换到sketch-rnn的格式，保存成npz文件
