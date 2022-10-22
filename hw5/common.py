import torch
from tqdm import tqdm


class TaggerDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, dtype=None):
        self.x = x
        self.y = y
        self.dtype = dtype

    def __getitem__(self, index):
        return torch.as_tensor(self.x[index], dtype=self.dtype), torch.as_tensor(self.y[index], dtype=self.dtype)
    
    def __len__(self):
        return self.x.shape[0]


class TorchTrainable:
    def fit(self, loader, optim, crit, *, epochs=5, device='cpu'):
        """
        :param loader - data loader
        :param optim - optimizer
        :param crit - criterion
        :param epochs
        :param device
        :param eval_params - predict() parameters for evaluation
        """        
        self.dev = device
        self.train()
        
        for ep in range(epochs):
            sum_loss, items = 0.0, 0
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {ep + 1}/{epochs}')
            for i, batch in pbar:
                inputs, labels = batch[0].to(self.dev), batch[1].to(self.dev)
                optim.zero_grad()
                outputs = self(inputs)
                loss = crit(outputs, labels)
                loss.backward()
                optim.step()

                sum_loss += loss.item()
                items += len(labels)
                pbar.set_postfix({'cumulative loss per item': sum_loss / items})

                # evaluate
                # if (i + 1 == len(loader)) and eval_params:
                #     self.eval()
                #     pbar.set_postfix()
                #     self.train()
        self.trained = True
        print('\nDone.')

    def predict(self, loader):
        if not hasattr(self, 'trained'):
            raise AttributeError('Model is not trained.')
        self.eval()
        for i, batch in enumerate(loader):
            inputs, labels = batch[0].to(self.dev), batch[1].to(self.dev)
            outputs = self(inputs)
            predicts = torch.cat([predicts, outputs]) if i > 0 else outputs
        return predicts.detach().cpu()
