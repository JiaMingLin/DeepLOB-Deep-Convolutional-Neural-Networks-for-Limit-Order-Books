from common import *

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization"""

        self.k = k
        self.num_classes = num_classes
        self.T = T
            
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:,self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

def lob_dataset(batch_size=64):
    dec_data = np.loadtxt('./data/LOB/zscore/training/Train_Dst_NoAuction_ZScore_CF_7.txt')
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    dec_test1 = np.loadtxt('./data/LOB/zscore/testing/Test_Dst_NoAuction_ZScore_CF_7.txt')
    dec_test2 = np.loadtxt('./data/LOB/zscore/testing/Test_Dst_NoAuction_ZScore_CF_8.txt')
    dec_test3 = np.loadtxt('./data/LOB/zscore/testing/Test_Dst_NoAuction_ZScore_CF_9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    # dec_test = dec_test[:, :int(np.floor(dec_test.shape[1] * 0.01))]
    print(dec_train.shape, dec_val.shape, dec_test.shape)

    dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
    dataset_val = Dataset(data=dec_val, k=4, num_classes=3, T=100)
    dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=100)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=12)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=12)

    print(dataset_train.x.shape, dataset_train.y.shape)
    tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)
    return train_loader, val_loader, test_loader, tmp_loader