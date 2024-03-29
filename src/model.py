
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
    
class Regressor(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(14, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.drop_layer = nn.Dropout(p=0.05)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = F.mse_loss(logits, y.view(-1,1))
        self.log('rmse_train', torch.sqrt(loss), on_step=True, on_epoch=False, logger=True, prog_bar=True)
        
        return {'loss': loss}
      
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        val_loss = F.mse_loss(logits, y.view(-1,1))
        #self.log('rmse_val', torch.sqrt(val_loss), on_step=True, on_epoch=False, logger=True, prog_bar=True)
        
        return {'val_loss': val_loss}
    
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([torch.sqrt(x['loss']) for x in outputs]).mean()
        
        self.log('step', self.current_epoch) # something wierd with loggers so you have to set step manually
        self.log('avg_rmse_train', avg_train_loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return None

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([torch.sqrt(x['val_loss']) for x in outputs]).mean()
        
        self.log('step', self.current_epoch) # something wierd with loggers so you have to set step manually
        self.log('avg_rmse_val', avg_val_loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return None
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        return optimizer
