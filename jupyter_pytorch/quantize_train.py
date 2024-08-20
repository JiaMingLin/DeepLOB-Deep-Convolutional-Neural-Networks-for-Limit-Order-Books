from common import *
from dataset import Dataset
from model import deeplob
from train_val import *
from opts import parser
from model import lob_model
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
# N, D = X_train.shape
    
    # please change the data_path to your local path
# data_path = '/nfs/home/zihaoz/limit_order_book/data'

dec_data = np.loadtxt('../data/Train_Dst_NoAuction_DecPre_CF_7.txt')
dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

dec_test1 = np.loadtxt('../data/Test_Dst_NoAuction_DecPre_CF_7.txt')
dec_test2 = np.loadtxt('../data/Test_Dst_NoAuction_DecPre_CF_8.txt')
dec_test3 = np.loadtxt('../data/Test_Dst_NoAuction_DecPre_CF_9.txt')
dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
print(dec_train.shape, dec_val.shape, dec_test.shape)

def main(exp_setting):
    with open(os.path.join('exp_cases', exp_setting+'.yml'), 'r') as file:
        settings = yaml.safe_load(file)

    batch_size = 64

    exp_name = settings['exp_name']
    # learning_rate = setting['learning_rate']
    

    dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
    dataset_val = Dataset(data=dec_val, k=4, num_classes=3, T=100)
    dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=100)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=12)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=12)

    print(dataset_train.x.shape, dataset_train.y.shape)

    tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)

    # quantization
    quant = settings['quant']
    # qat = settings['qat']
    fp_model = settings.get('fp_model', None)
    w_bit = settings['w_bit']
    acc_bit = settings['acc_bit']
    a_bit = settings['a_bit']
    i_bit = settings['i_bit']
    o_bit = settings['o_bit']
    r_bit = settings['r_bit']
    no_brevitas = settings['no_brevitas']

    for x, y in tmp_loader:
        print(x)
        print(y)
        print(x.shape, y.shape)
        break

    model = lob_model('lob_lstm', 
                      quant = quant, w_bit = w_bit, acc_bit = acc_bit, i_bit = i_bit, 
                      o_bit = o_bit, r_bit = r_bit)
    model.to(device)

    summary(model, (1, 1, 100, 40))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses, val_losses = train(model, criterion, optimizer, 
                                        train_loader, val_loader, exp_name, epochs=50)

    model = torch.load(f'best_val_model_{exp_name}')

    test_model(model, test_loader)

if __name__ == "__main__":

    args = parser.parse_args()
    exp_settings = args.settings
    for setting in exp_settings:
        main(setting)