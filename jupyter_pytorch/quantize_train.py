from common import *
from model import deeplob
from train_val import *
from opts import parser
from model import lob_model
from pathlib import Path
from brevitas import config
from data.EMG.dataset import *
from data.LOB.dataset import *
from torch.utils.tensorboard import SummaryWriter
    
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
# N, D = X_train.shape
    
    # please change the data_path to your local path
# data_path = '/nfs/home/zihaoz/limit_order_book/data'

def main(exp_setting):
    with open(os.path.join('exp_cases', exp_setting+'.yml'), 'r') as file:
        settings = yaml.safe_load(file)

    dataset = settings['dataset']
    exp_name = settings['exp_name']
    learning_rate = settings['learning_rate']
    epochs = settings['epochs']

    num_layers = settings['num_layers']
    # quantization
    quant = settings['quant']
    quant_type = settings['quant_type']
    # qat = settings['qat']
    fp_model = settings.get('fp_model', None)
    w_bit = settings['w_bit']
    acc_bit = settings['acc_bit']
    a_bit = settings['a_bit']
    i_bit = settings['i_bit']
    o_bit = settings['o_bit']
    r_bit = settings['r_bit']
    no_brevitas = settings['no_brevitas']

    feature_num = 40
    output_size=3
    hidden_size = 64
    if dataset == 'LOB':
        train_loader, val_loader, test_loader, tmp_loader = lob_dataset(batch_size=256)
    elif dataset == 'EMG':
        feature_num = 8
        output_size=8
        hidden_size = 128
        train_loader, val_loader, test_loader, tmp_loader = emg_dataset()

    for x, y in tmp_loader:
        print(x)
        print(y)
        print(x.shape, y.shape)
        break

    writer = SummaryWriter(f'log/{exp_name}')
    Path('saved_models').mkdir(parents=True, exist_ok=True)

    model = lob_model('lob_lstm', 
                      feature_num=feature_num, output_size=output_size, num_layers=num_layers, hidden_size = hidden_size,
                      quant = quant, quant_type=quant_type, 
                      w_bit = w_bit, acc_bit = acc_bit, a_bit = a_bit, i_bit = i_bit, 
                      o_bit = o_bit, r_bit = r_bit)
    model.to(device)

    summary(model, (1, 1, 100, feature_num))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, criterion, optimizer, 
          train_loader, val_loader, exp_name, writer,
          epochs=epochs)

    config.IGNORE_MISSING_KEYS = True
    model.load_state_dict(torch.load(f'best_val_model_{exp_name}.pt'))
    model.to(device)

    test_model(model, criterion, test_loader)

if __name__ == "__main__":

    args = parser.parse_args()
    exp_settings = args.settings
    for setting in exp_settings:
        main(setting)
