import torch.nn as nn
import torch
import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from .modules import SpatialAttentionBlock

class SatUNet(L.LightningModule):
    def __init__(self, in_channels, out_channels, optim_params):
        super().__init__()

        channels = [16, 32, 64, 128, 256, 512]

        self.optim_params = optim_params
        self.criterion = nn.BCEWithLogitsLoss()
        self.acc = BinaryAccuracy()
        self.pr = BinaryPrecision()
        self.rc = BinaryRecall()
        self.f1 = BinaryF1Score()

        self.conv1 = self.contract_block(in_channels, channels[1])
        self.conv2 = self.contract_block(channels[1], channels[2])
        self.conv3 = self.contract_block(channels[2], channels[3])
        self.conv4 = self.contract_block(channels[3], channels[4])
        self.conv5 = self.contract_block(channels[4], channels[5])

        self.satt5 = SpatialAttentionBlock(channels[5], channels[5], channels[4])
        self.satt4 = SpatialAttentionBlock(channels[4], channels[4], channels[3])
        self.satt3 = SpatialAttentionBlock(channels[3], channels[3], channels[2])
        self.satt2 = SpatialAttentionBlock(channels[2], channels[2], channels[1])
        self.satt1 = SpatialAttentionBlock(channels[1], channels[1], channels[0])

        self.upconv5_1 = self.expand_block(channels[5], channels[5], 1, 0)
        self.upconv4_1 = self.expand_block(channels[4], channels[4], 1, 0)
        self.upconv3_1 = self.expand_block(channels[3], channels[3], 1, 0)
        self.upconv2_1 = self.expand_block(channels[2], channels[2], 1, 0)
        self.upconv1_1 = self.expand_block(channels[1], channels[1], 1, 0)
        
        self.upconv5_2 = self.expand_block(channels[5], channels[4], 2, 1)
        self.upconv4_2 = self.expand_block(channels[4], channels[3], 2, 1)
        self.upconv3_2 = self.expand_block(channels[3], channels[2], 2, 1)
        self.upconv2_2 = self.expand_block(channels[2], channels[1], 2, 1)
        self.upconv1_2 = self.expand_block(channels[1], channels[0], 2, 1)

        self.last = nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x, *args, **kwargs):
        
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        u5 = self.upconv5_1(conv5)
        attention5 = self.satt5(g=u5, x=conv5)
        upconv5 = self.upconv5_2(u5+attention5)

        u4 = self.upconv4_1(upconv5)
        attention4 = self.satt4(g=u4, x=conv4)
        upconv4 = self.upconv4_2(u4+attention4)

        u3 = self.upconv3_1(upconv4)
        attention3 = self.satt3(g=u3, x=conv3)
        upconv3 = self.upconv3_2(u3+attention3)

        u2 = self.upconv2_1(upconv3)
        attention2 = self.satt2(g=u2, x=conv2)
        upconv2 = self.upconv2_2(u2+attention2)

        u1 = self.upconv1_1(upconv2)
        attention1 = self.satt1(g=u1, x=conv1)
        upconv1 = self.upconv1_2(u1+attention1)
        
        last = self.last(upconv1)
        return last

    def contract_block(self, in_channels, out_channels, stride=2):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return contract

    def expand_block(self, in_channels, out_channels, stride, out_pad):
        expand = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=out_pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return expand

    def epoch_step(self, batch):
        inputs, labels = batch
        labels = labels.float()
        outputs = self(inputs).squeeze(1)

        return {'loss': self.criterion(outputs, labels), 
                'acc': self.acc(outputs, labels), 
                'precision': self.pr(outputs, labels), 
                'recall': self.rc(outputs, labels),
                'f1': self.f1(outputs, labels)}

    def log_losses(self, losses, mode="train"):
        for loss_name, loss_value in losses.items():
            self.log(f"{mode}_{loss_name}", loss_value, 
                    on_epoch=True, on_step=False, logger=True, 
                    prog_bar=(loss_name == "loss"))

    def training_step(self, batch, batch_idx):
        losses = self.epoch_step(batch)
        self.log_losses(losses, mode="train")
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        losses = self.epoch_step(batch)
        self.log_losses(losses, mode="valid")
        return losses['loss'] 

    def configure_optimizers(self):
        optim_type = self.optim_params["type"]
        assert optim_type in ['Adam', 'SGD'], "Invalid optimizer type"
        m_params = self.optim_params[optim_type]
        if optim_type == 'Adam':
            return torch.optim.Adam(self.parameters(), 
                        lr=m_params['lr'],
                        weight_decay=m_params['weight_decay'])
        else:
            return torch.optim.SGD(self.parameters(), 
                       lr=m_params['lr'],
                       momentum=m_params['momentum'], 
                       weight_decay=m_params['weight_decay'])


if __name__ == '__main__':
    model = SatUNet(in_channels=4,
                    out_channels=1, 
                    optim_params={'type': 'Adam', 'lr': 1e-3, 'weight_decay': 1e-5})

    print(sum(p.numel() for p in model.parameters()))
    x = torch.randn(1, 4, 384, 384)
    print(model(x).shape)
   