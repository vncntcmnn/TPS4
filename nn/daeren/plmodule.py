import pytorch_lightning as pl
from .module import TransformerAutoEncoder, SwapNoiseMasker
from torch.optim import Adam


class DAERen(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.noise_maker = SwapNoiseMasker(self.haprams['swap'])
        self.dae = TransformerAutoEncoder(self.hparams['dae'])

    def forward(self, x):
        denoised = self.dae(x)
        return denoised

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_corrputed, mask = self.noise_maker.apply(x)
        loss = self.dae.loss(x_corrputed, x, mask)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer

