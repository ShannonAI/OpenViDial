import math

class Optim():
    
    def __init__(self, optimizer, d_model, warm_up_step):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warm_up_step = warm_up_step
        self.n_current_step = 0
        self.init_lr = math.pow(self.d_model, -0.5)

    def step_and_updata_lr(self):
        self.updata_lr()
        self.optimizer.step()

    def get_lr(self):
        return min(math.pow(self.n_current_step, -0.5), math.pow(self.warm_up_step, -1.5)*self.n_current_step)
    
    def updata_lr(self):
        self.n_current_step += 1
        lr = self.init_lr * self.get_lr()
        for para in self.optimizer.param_groups:
            para['lr'] = lr
    
    def zero_grad(self):
        self.optimizer.zero_grad()