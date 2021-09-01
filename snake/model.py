# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as func
# import os

# class Linear_QNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__() #initializes a super init
#         self.linear_1 = nn.Linear(input_size,hidden_size)
#         self.linear_2 = nn.Linear(hidden_size,output_size)
    
#     def forward(self,tensor):
#         tensor = func.relu(self.linear_1(tensor))
#         tensor = self.linear_2(tensor)
#         return tensor

#     def save(self,file_name = "model.pth"):
#         model_folder_path = "./model"
#         if not os.path.exists(model_folder_path): #making it only if it doesnt exist
#             os.makedirs(model_folder_path)
#         file = os.path.join(model_folder_path,file_name) #joins it together to make the actual file path
#         torch.save(self.state_dict(),file) #saving the file
    
# class QTrainer:
#     def __init__(self,model,lr,gamma):
#         self.lr = lr
#         self.model = model
#         self.gamma = gamma
#         self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
#         self.criterion = nn.MSELoss()
    
#     def train_step(self,state,action,reward,new_state,game_over):
#         #here you make them all tensors so that you can handle either individual or multiple values
#         state = torch.tensor(state, dtype=torch.float)
#         new_state = torch.tensor(new_state, dtype=torch.float)
#         action = torch.tensor(action, dtype=torch.long)
#         reward = torch.tensor(reward, dtype=torch.float)

#         if len(state.shape) == 1:
#             #puts it in the format (1, tensor)
#             torch.unsqueeze(state,0)
#             torch.unsqueeze(new_state,0)
#             torch.unsqueeze(action,0)
#             torch.unsqueeze(reward,0)
#             game_over = (game_over,)
        
#         #you want the predicted q values along with the current state
#         pred = self.model(state)
#         target = pred.clone() #cloning the prediction
#         for i in range(len(game_over)): #running across all items
#             Q_new = reward[i]
#             if not game_over[i]: #making sure the ith game is not over
#                 Q_new = reward[i] + self.gamma * torch.max(self.model(new_state[i]))
#             target[i][torch.argmax(action).item()] = Q_new 
#         self.optimizer.zero_grad()
#         loss= self.criterion(target,pred)
#         loss.backward()
#         self.optimizer.step()
#         #Q_new = reward * gamma + max(next predicted q value)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
