import copy
import numpy as np

def get_loader(**kwargs):
    class LimitedDataset(kwargs['full_dataset']):
        def __init__(self,
                     examples_per_class=None,
                     epc_seed=None,
                     **kwargs):

            super(LimitedDataset, self).__init__(**kwargs)
            
            train_labels = copy.deepcopy(getattr(self, 'train_labels'))
            
            samp_ind = []
            
            for i in range(max(train_labels)+1):
                np.random.seed(epc_seed)
                
                i_ind = np.where(train_labels == i)[0]
                
                i_ind = np.random.choice(i_ind, examples_per_class, replace=False)
                samp_ind += i_ind.tolist()

            self.targets = train_labels[samp_ind]
            self.data = self.data[samp_ind,]
    
    del kwargs['full_dataset']
    return LimitedDataset(**kwargs)
