import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

X = np.linspace(-3,3, 300).reshape(-1,1)
Y = np.sin(X) + np.random.uniform(-0.5,0.5,300).reshape(-1,1)

fig, ax = plt.subplots(2,1, figsize=(5,10))

ax[0].scatter(X,Y, color='y')

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# model = LinearRegression()

# L2正则化，平方消除过大的参数带来的过拟合风险，一般不会对参数赶尽杀绝，而是在保留参数的前提下，转一个很小的值，对结果影响小
# model = Ridge(alpha=0.1)

# L1正则化，绝对值消除过大的参数带来的过拟合风险，可能会对影响很小的参数赶尽杀绝，直接参数值清0 -- 有特征过滤的效果
model = Lasso(alpha=0.001)
poly = PolynomialFeatures(degree=20)
poly_train_X = poly.fit_transform(train_X)
poly_test_X = poly.fit_transform(test_X)

model.fit(poly_train_X, train_Y)
y_predict = model.predict(poly_test_X)
loss = mean_squared_error(test_Y, y_predict)

ax[0].plot(X, model.predict(poly.transform(X)), color='red')
ax[0].text(-3,1, f'测试误差：{loss:.4f}', color='red')

# bar 绘制直方图，range表示绘制的直方图参数范围，因为poly指定是构建20次方项，再加一个常数项，所以参数个数是21，参数从model.coef_获取
ax[1].bar(np.arange(21).reshape(-1), model.coef_, color='blue')

plt.show()