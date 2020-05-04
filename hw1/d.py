import numpy as np
from matplotlib import pyplot as plt

data_size = [60, 160, 320]
mu, sigma = 0, 0.04
_min, _max = 0, 1
degree = 14

def create_data(num):
    eps = np.random.normal(mu, sigma, (num,))
    x = np.random.uniform(_min, _max, (num,))
    y = np.sin(2 * np.pi * x) + eps
    
    return x, y

def least_squares_method(x, y):

    return np.linalg.inv(x.T @ x) @ x.T @ y

def mse(x, y):
    return np.mean((y - x) ** 2) 

def kfold_validation(train_input, train_target, fold):
    idx = np.arange(len(train_input))
    losses = np.zeros((fold,))

    for i in range(fold):
        select = idx % fold == i
        val_x = train_input[select]
        train_x = train_input[~select]
        val_y = train_target[select]
        train_y = train_target[~select]
        weight = least_squares_method(train_x, train_y)
        predict = val_x @ weight
        loss = mse(predict, val_y)
        losses[i] = loss

    return losses.mean()
    
def loo_validation(train_input, train_target):
    idx = np.arange(len(train_input))
    losses = np.zeros((len(train_input),))
    
    for i in idx:
        select = idx == i
        val_x = train_input[select]
        train_x = train_input[~select]
        val_y = train_target[select]
        train_y = train_target[~select]
        try:
            weight = least_squares_method(train_x, train_y)
        except:
            continue
        predict = val_x @ weight
        loss = mse(predict, val_y)
        losses[i] = loss

    return losses.mean()

def draw(ax, _input, target, model_x, model_y):
    
    ax.plot(model_x, model_y)
    ax.scatter(_input, target, marker='o', color=["green"])
    ax.set_xlim(_min, _max)

if __name__ == '__main__':
    
    for size in data_size:
        train_size = round(size * 0.75)
        test_size = size - train_size

        x, y = create_data(train_size + test_size)
        train, test = x[:train_size], x[train_size:]
        train_target = y[:train_size]
        test_target = y[train_size:]
        show = np.arange(_min, _max + .01, .01)
    
        plt.figure(figsize=(8, 6))
        train_input = np.ones((train_size, degree + 1))
        test_input = np.ones((test_size, degree + 1))
        show_input = np.ones((show.shape[0], degree + 1))

        for d in range(1, degree+1):
            show_input[:, d] = show ** d
            train_input[:, d] = train ** d
            test_input[:, d] = test ** d
    
        weight = least_squares_method(train_input, train_target)
        model_x = show_input[:, 1]
        model_y = show_input @ weight

        kfold_loss = kfold_validation(train_input, train_target, 5)
        loo_loss = loo_validation(train_input, train_target)

        ## plot train
        ax = plt.subplot(121)
        ax.set_title("Train")
        train_predict = train_input @ weight
        train_loss = mse(train_predict, train_target)
        draw(ax, train_input[:, 1], train_target, model_x, model_y)
        
        ## plot test
        ax2 = plt.subplot(122)
        ax2.set_title("Test")
        test_predict = test_input @ weight
        test_loss = mse(test_predict, test_target)
        draw(ax2, test_input[:, 1], test_target, model_x, model_y)

        string_loss = f"Train Loss: {train_loss}\nKfold Loss: {kfold_loss}\nLoo Loss: {loo_loss}\nTest Loss:{test_loss}"
        plt.gcf().text(0.3, 0.1, string_loss, fontsize=14, verticalalignment="bottom")
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(f"d_{size}.png")