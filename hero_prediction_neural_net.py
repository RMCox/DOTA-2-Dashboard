def run_neural_network(model_type, *args):
    import pickle
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import pandas as pd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    user_input = [item for item in args]

    hero_checker = pd.read_csv(r"dota-2-matches\hero_names.csv")
    
    
    def func(x):
        if x == 113:
            return x-4
        elif x > 107:
            return x-3
        elif x > 23:
            return x-2
        else:
            return x-1
    
    hero_checker['hero_id'] = hero_checker['hero_id'].apply(func)
    hero_indices = hero_checker['hero_id'][hero_checker['localized_name'].isin(user_input)]
    
    class LSTM_Classifier(nn.Module):
        def __init__(self, input_size = 990, hidden_size = 110, num_layers = 3, num_classes = 110): # 990 here, the ds[0][0].size
            super(LSTM_Classifier, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = True
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional)
            if self.bidirectional:
                self.fc = nn.Linear(hidden_size*2, num_classes)
            else:
                self.fc = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x):
            x = x.view(-1, 1, 990).float().to(device) # 990 here, the ds[0][0].size
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            if self.bidirectional:
                h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
                c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
            out, _ = self.lstm(x, (h0, c0))

            out = self.fc(out[:, -1, :])
            return F.log_softmax(out, dim=1)

    class linear_net(nn.Module):
        def __init__(self):
            super(linear_net, self).__init__()
            self.layer1 = nn.Linear(9 * 110, 300)
            self.layer2 = nn.Linear(300, 200)
            self.layer3 = nn.Linear(200, 150)
            self.layer4 = nn.Linear(150, 110)
            
            self.dropout = nn.Dropout(p=0.2)
        def forward(self, x):
            x = x.float().to(device)
            x = x.view(-1, 9 * 110)
            x = self.dropout(F.relu(self.layer1(x)))
            x = self.dropout(F.relu(self.layer2(x)))
            x = self.dropout(F.relu(self.layer3(x)))
            x = self.layer4(x)
            return F.log_softmax(x, dim=1)
        
    if model_type == "LSTM":
        # defining the model and loading it's state
        model = LSTM_Classifier()
        torch_input = torch.tensor(np.eye(110)[hero_indices])
        model.load_state_dict(torch.load(r"LSTM_Classifier"))
        output = model(torch_input)
        probabilities = F.softmax(output, dim=1)
        predictions = torch.topk(output, k=10).indices.tolist()[0]
        
        hero_outputs = hero_checker['localized_name'][hero_checker['hero_id'].isin(predictions)].tolist()
        probability_output = probabilities[0][predictions].tolist()
        return(hero_outputs, probability_output)
        
    elif model_type == "FFNN":
        # defining the model and loading it's state
        # need different input for FFNN
        # use 110 since 110 heroes in DOTA
        model = linear_net()
        torch_input = torch.LongTensor(np.eye(110)[hero_indices.tolist()])
        model.load_state_dict(torch.load(r"linear_net"))
        output = model(torch_input)
        probabilities = F.softmax(output, dim=1)
        predictions = torch.topk(output, k=10).indices.tolist()[0]
        hero_outputs = hero_checker['localized_name'][hero_checker['hero_id'].isin(predictions)].tolist()
        probability_output = probabilities[0][predictions].tolist()
        print(predictions)
        return(hero_outputs, probability_output)
        
run_neural_network("FFNN", 'Abaddon', 'Alchemist', 'Ancient Apparition', 'Anti-Mage', 'Arc Warden', 'Axe', 'Bane', 'Batrider', 'Beastmaster')
        