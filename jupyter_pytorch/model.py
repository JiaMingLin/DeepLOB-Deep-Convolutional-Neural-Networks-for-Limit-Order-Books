from common import *


from brevitas.nn import QuantLinear
from brevitas.nn import QuantLSTM


class deeplob(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
#             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        
        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(device)
        c0 = torch.zeros(1, x.size(0), 64).to(device)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)  
        
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        
#         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)
        
        return forecast_y

class LOB_LSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=64, num_layers=1, output_size=3,
                quant = False, quant_type='int', w_bit=8, acc_bit=16, a_bit=8, i_bit=8, o_bit=8, r_bit=8, b_bit=8,
                no_brevitas = True
    ):
        super(LOB_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.quant = quant
        if quant:
            self.rnn = QuantLSTM(
                input_size, hidden_size, num_layers, batch_first=True,
                weight_quant = weight_quantizer[f'{quant_type}{w_bit}'],
                io_quant = act_quantizer[f'{quant_type}{o_bit}'],
                sigmoid_quant = act_quantizer[f'u{quant_type}{a_bit}'],
                tanh_quant = act_quantizer[f'{quant_type}{a_bit}'],
                cell_state_quant = act_quantizer[f'{quant_type}{r_bit}'],
                gate_acc_quant = act_quantizer[f'{quant_type}{acc_bit}'],
                bias_quant = bias_quantizer[f'int{b_bit}']
            )
            self.fc = QuantLinear(hidden_size, output_size, 
                                weight_quant=weight_quantizer['int8']
                                )

        else:
            self.rnn = QuantLSTM(
                input_size, hidden_size, num_layers, batch_first=True,
                weight_quant = NoneWeightQuant,
                io_quant = NoneActQuant,
                gate_acc_quant = NoneActQuant,
                sigmoid_quant = NoneActQuant,
                tanh_quant = NoneActQuant,
                cell_state_quant = NoneActQuant,
                bias_quant = NoneBiasQuant
            )
            self.fc = QuantLinear(hidden_size, output_size, 
                                weight_quant=NoneWeightQuant
                             )

        if no_brevitas:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        outputs, (h_n, _) = self.rnn(x)
        outputs = outputs[:,-1,:]
        # outputs = self.relu(outputs)
        out = self.fc(outputs)
        return out

def lob_model(model_name, feature_num=40, output_size=3, num_layers=1, hidden_size=64,
              quant=False, quant_type='int',
              w_bit=8, acc_bit=16, a_bit=8, i_bit=8, o_bit=8, r_bit=8, b_bit=8,
              no_brevitas=False):
    if model_name == 'deeplob':
        return deeplob(3)
    elif model_name == 'lob_lstm':
        return LOB_LSTM(feature_num, hidden_size, num_layers, output_size, 
                        quant, quant_type, 
                        w_bit, acc_bit, a_bit, i_bit, o_bit, r_bit, b_bit,
                        no_brevitas)
    else:
        raise ValueError('Model not found')