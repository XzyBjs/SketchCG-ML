import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 简单GPU测试
if tf.config.list_physical_devices('GPU'):
    print("GPU设备名称:", tf.config.list_physical_devices('GPU'))
    
    # 创建一个在GPU上运行的简单计算
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("矩阵乘法结果:", c)
        print("计算设备:", c.device)
else:
    print("未检测到GPU")