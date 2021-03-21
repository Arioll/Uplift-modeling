import torch
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Data(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(Data, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return self.X.shape[0] # >>> your solution here <<<
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class Uplift_NN(torch.nn.Module):
    def __init__(self, input_size,hid_size):
            super(Uplift_NN, self).__init__()
            self.input_size = input_size
            self.hid_size = hid_size
            self.hid_size.insert(0,input_size)
            layers = []
            for i in range(len(hid_size)-1):
                layers.append(torch.nn.Linear(hid_size[i], hid_size[i+1], bias=True))
                layers.append(torch.nn.BatchNorm1d(hid_size[i+1]))
                layers.append(torch.nn.LeakyReLU(0.05))
            layers.append(torch.nn.Linear(hid_size[-1], 1, bias=True))
            layers.append(torch.nn.Sigmoid())
            self.net = torch.nn.Sequential(*layers)
            
                        
    def forward(self, x):
            output = self.net(x)
            return output
        
        
class Classifier_NN():
    def __init__(self, input_size,hid_size,epoch=5,lr=1e-4):
        self.model = Uplift_NN(input_size,hid_size)
        self.epoch = epoch
        self.lr=lr
    
    def fit(self,X_train,y_train):
        self.model.train()
        trainset = Data(X_train,y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True,num_workers=0, drop_last=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_function = torch.nn.BCELoss()
        for iter_i in range(self.epoch):
            for inputs, labels in trainloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_function(outputs[:,0], labels)
                loss.backward()
                optimizer.step()
        return self
                
    def predict_proba(self,X_test):
        self.model.eval()
        X_test = torch.from_numpy(X_test.astype(np.float32))
        prob_1 = self.model(X_test)
        prob_0 = torch.ones_like(prob_1)
        prob_0 -= prob_1
        return torch.cat((prob_0, prob_1), 1).detach().numpy()
    
    def predict(self,X_test):
        self.model.eval()
        X_test = torch.from_numpy(X_test.astype(np.float32))
        prob_1 = self.model(X_test)
        out = (prob_1>0.5).int()
        return out.detach().numpy()[:,0]
