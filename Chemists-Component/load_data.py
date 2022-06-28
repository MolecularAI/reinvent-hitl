import os
from rdkit.Chem import AllChem as Chem
from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from molecular_features import descriptorsX, morganX
import time
class Dataset2D:
    def __init__(self, file, y_field=None, id_field=None, ext='sdf'):
        self.smiles = []
        self.moles = []
        self.Y = [] if y_field is not None else None
        self.id = []
        temp_id = 1
        if ext == 'sdf':
            suppl = Chem.SDMolSupplier(file, strictParsing=False)
            for i in suppl:
                if i is None:
                    continue
                smi = Chem.MolToSmiles(i, isomericSmiles=False)
                if smi is not None and smi != '':
                    self.smiles.append(smi)
                    self.moles.append(i)
                    if y_field is not None:
                        self.Y.append(i.GetProp(y_field))
                    if id_field is not None:
                        self.id.append(i.GetProp(id_field))
                    else:
                        self.id.append('id{:0>5}'.format(temp_id))
                        temp_id += 1
        
        elif ext == 'csv':
            # df = pd.read_csv(file)
            df=file
            try:
                df['moles'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
            except KeyError:
                df['SMILES'] = df['canonical']
                df['moles'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
            df = df.dropna()
            self.smiles = df['SMILES'].tolist()
            self.moles = df['moles'].tolist()
            self.Y = df[y_field].tolist() if y_field is not None else None
            self.id = df[id_field].tolist() if id_field is not None else np.arange(len(self.smiles))
            
        else:
            raise ValueError('file extension not supported!')
            
                    
        assert(len(self.smiles) == len(self.moles) == len(self.id))
        if self.Y is not None:
            assert(len(self.smiles) == len(self.Y))
            self.Y = np.array(self.Y)

    
    def __getitem__(self, index):
        if self.Y is not None:
            ret = self.id[index], self.smiles[index], self.moles[index], self.Y[index]
        else:
            ret = self.id[index], self.smiles[index], self.moles[index]
        return ret
    
    def __len__(self):
        return len(self.smiles)
    
    def __add__(self, other):
        pass


class DataStructure:
    def __init__(self, dataset, feat_fn, y_transforms=None, num_proc=1):
        self.dataset = dataset
        self.feat_fn = feat_fn
        self.Y = dataset.Y
        self.id = dataset.id
        self.num_proc = num_proc
        self.feat_names = []
        self.name_to_idx = {}

        x_s = []
        for fname in self.feat_fn.keys():
            f = self.feat_fn[fname]
            with Pool(self.num_proc) as p:
                arr = np.array(p.map(f, self.dataset.moles))
            x_s.append(arr)
            length = arr.shape[1]
            names = list('{}_{}'.format(fname, x+1) for x in range(length))
            self.feat_names += names
        x_s = tuple(x_s)
        self.X_ = np.concatenate(x_s, axis=1)
        
        # remove any nan rows
        nans = np.isnan(self.X_)
        mask = np.any(nans, axis=1)
        self.X_ = self.X_[~mask, :]
        self.name_to_idx = dict(zip(self.feat_names, range(len(self.feat_names))))
        self.id = list(self.id[j] for j in range(len(mask)) if not mask[j])
        if self.Y is not None:
            self.Y = self.Y[~mask]
        if y_transforms is not None:
            for t in y_transforms:
                self.Y = np.array(list(map(t, self.Y)))
    
    def __len__(self):
        return self.X_.shape[0]
    
    @property
    def shape(self):
        return self.X_.shape
    
    def X(self, feats=None):
        '''
        Use a list of to select feature columns
        '''
        if feats is None:
            return self.X_
        else:
            mask = list(map(lambda x: self.name_to_idx[x], feats))
            return self.X_[:, mask]




def load_data(train_data_path, test_data_path, output_dir, train_num, N0, y_field, id_field=None, ext='csv', sampling=True, normalization=False):
    print(train_data_path)
    train_data=pd.read_csv(train_data_path)
    if test_data_path:
        test_data=pd.read_csv(test_data_path)
    train_data['class']=(train_data[y_field]>=0.5).astype(int)

    if sampling:
        print('sampling')
        # sample train_num samples from train_data
        train_data, _ =train_test_split(train_data, train_size=train_num/train_data.shape[0],stratify=train_data['class'])
        train_data=train_data.reset_index(drop=True)
        # sample test_num samples from test_data
        # _, test_data=train_test_split(test_data, test_size=test_num/test_data.shape[0],stratify=test_data['activity'])

    if test_data_path is None:
        print('splitting')
        train_data, test_data =train_test_split(train_data, test_size=0.2,stratify=train_data[y_field])

    L=train_data.sample(n=int(N0)).index
    #print('training data have {} active'.format(train_data.loc[train_data[y_field]>=0.5].shape))
    #print('training data have {} inactive'.format(train_data.loc[train_data[y_field]<0.5].shape))
    #print('testing data have {} active'.format(test_data.loc[test_data[y_field]>=0.5].shape))
    #print('testing data have {} inactive'.format(test_data.loc[test_data[y_field]<0.5].shape))
    train_ds=Dataset2D(train_data, y_field=y_field, id_field=id_field, ext=ext)
    test_ds=Dataset2D(test_data, y_field=y_field, id_field=id_field, ext=ext)
    start=time.time()
    train_str = DataStructure(train_ds, dict(physchem=morganX), num_proc=8)
    test_str = DataStructure(test_ds, dict(physchem=morganX), num_proc=8)
    print("trainsformation spend {} s".format(time.time()-start))

    # X contains features
    X_train = train_str.X()
    y_train = train_str.Y
    y_train=y_train.reshape(-1, 1)
    X_test = test_str.X()
    y_test = test_str.Y
    y_test=y_test.reshape(-1,1)

    smiles_train= np.array([ item[1] for item in train_str.dataset])
    id_train= np.array([item[0] for item in train_str.dataset])

    if normalization:
        print('Normalizing...')
        #Normalization, speed up training process
        scaler = StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
        modelfile=output_dir+"/models/scaler.pkl"
        try:
            os.mkdir(output_dir + '/models')
        except FileExistsError:
            pass
        with open(modelfile, "wb+") as f:
            pickle.dump(scaler, f)
    U=np.setdiff1d(np.arange(X_train.shape[0]),L)
    X_L, X_U, y_L, y_U=X_train[L], X_train[U], y_train[L], y_train[U]
    return X_train, X_test, y_train, y_test, smiles_train, id_train, X_L, X_U, y_L, y_U, L , U