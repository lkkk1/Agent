import numpy as np
import matplotlib.pyplot as plt

# 梯度下降的计算过程
# 1. 初始化参数
# 2. 计算梯度
# 3. 更新参数
# 4. 重复迭代

# x**2 = 2 , 找f(x) = 2 , 可能对应，系数取到某个值时，最匹配对应的标签2，找这样的一个系数点
# 均方误差，放大取正
def J(x):
	return (x**2 - 2)**2

def gradient(x):
	return 4*x**3-8*x

# 学习率的设置 -- 影响梯度下降的变更步长，
# 如果设置较大如0.2 , 会导致更新步长太大，高纬大幅度震荡，无法到达梯度低点，即无法收敛
# 如果设置较小如0.01，虽然可以逐步收敛，且收敛过程平滑，没有震荡，但是达到收敛的训练次数暴增，计算成本大

# 学习率设置合理，如0.1，可以快速收敛，虽然有小幅度震荡，但可以平衡计算次数和收敛速度。
alpha = 0.1
# alpha = 0.2
# alpha = 0.01
x = 1

# 记录x, y 的坐标变更
x_array = []
y_array = []

# 重复迭代n次 或者 梯度小于某个值，就停止迭代

# ：= 海象运算法，允许在表达式中进行赋值，通常结合循环使用
while np.abs(grad:= gradient(x)) > 1e-10:
	y = J(x)
	x_array.append(x)
	y_array.append(y)
	print(f'x={x}, f(x)={y}, gradient={grad}')
	x = x-alpha*grad
print(len(x_array))

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(x_array, y_array)

# 4*x**3 = 8*x 最小值，应该在根号2附近
x_list = np.arange(0.9, 2, 0.01)
ax[0].plot(x_list, J(x_list), color='blue')
ax[0].plot(x_array, y_array, color='r')
ax[0].scatter(x_array, y_array, color='r')

# 因为第一次梯度下降很多，所以为了直观展现后续的变化，对第一个点做舍弃后，对后续的变化做放大展示
x_array2 = x_array[1:]
y_array2 = y_array[1:]
x_list2 = np.arange(1.399, 1.425, 0.001)

ax[1].plot(x_list2, J(x_list2), color='blue')
ax[1].plot(x_array2, y_array2, color='r')
ax[1].scatter(x_array2, y_array2, color='r')

plt.show()



