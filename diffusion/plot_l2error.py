import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_model_error(model, x_bounds, y_bounds, device='cuda'):
    # 解包上下界
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    # 生成网格点
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    xx, tt = np.meshgrid(x, y)
    
    # 创建张量
    X = torch.DoubleTensor(np.stack([xx.ravel(), tt.ravel()], axis=1)).to(device)
    
    # 计算目标值
    target_y = (torch.exp(-X[:,1]) * torch.sin(np.pi * X[:,0])).flatten()[:, None].cpu()
    
    # 计算模型的输出
    with torch.no_grad():
        model_output = model(X).cpu().numpy()
    
    # 计算误差
    y = target_y.numpy()
    y = y - model_output
    Z = np.abs(y.reshape(xx.shape))
    
    # 绘制三维图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    surf = ax.scatter(xx, tt, c=Z, cmap='OrRd', s=10)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Abs. error')
    
    # 计算L2误差
    outputs = np.abs(y)
    error = np.sqrt(np.sum(outputs**2)) / np.sqrt(len(y))
    print('error', error)
    error = "{:.4f}".format(error)
    plt.figtext(0.45, 0, f"L2 error: {error}", ha='center', fontsize=10)
    plt.show()

    return error
    
# 使用示例
# plot_model_error(pinns_all.model, (-1, 1), (0, 1), device='cuda')
