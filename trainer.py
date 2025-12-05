import torch
import os
import time
from tqdm import tqdm

class Trainer():
    def __init__(self, cfg, logger):
        super().__init__()

        self.device = torch.device(cfg['devices'] if torch.cuda.is_available() else "cpu")

        self.exp = logger
        self.num_nodes = cfg['num_nodes']
        if cfg['precision'] == 16 or cfg['precision'] == 'bf16-mixed' or cfg['precision'] == 'bf16-true' or cfg['precision'] == 'bf16':
            self.precision = torch.bfloat16
        else:
            self.precision = torch.float32
        self.max_epochs = cfg['max_epochs']
        self.check_val_every_n_epoch = cfg['check_val_every_n_epoch']
        
        '''
        fast_dev_run: false 
        accumulate_grad_batches: 1
        profiler: false
        detect_anomaly: false
        enable_model_summary: true # nn info
        deterministic: false
        benchmark: true
        '''
                            
    def fit(self, model, datamodule, ckpt_path, early_stop):
        model.to(self.device)

        optim_params = model.optim_params
        criterion = model.criterion
        acc = model.acc
        pr = model.pr
        rc = model.rc
        f1 = model.f1

        optimizer = model.configure_optimizers()

        losses, accs, prs, rcs, f1s = [], [], [], [], []

        if self.precision == torch.bfloat16:
            use_amp = True
        else:
            use_amp = False

        start_epoch = 0
        min_loss = 2
        if os.path.exists(os.path.join(ckpt_path, 'last.ckpt')):
            checkpoint = torch.load(os.path.join(ckpt_path, 'last.ckpt'), weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_states'])#[0])
            min_loss = checkpoint['loss']
            start_epoch = checkpoint['epoch']+1

        patience_counter = 0
        start_time = time.time()

        for epoch in range(start_epoch, self.max_epochs):
            print('Epoch {}'.format(epoch))

            if epoch % self.check_val_every_n_epoch == 0:
                phases = ["train", "valid"]
            else:
                phases = ["train"]

            for phase in phases:
                if phase == 'train':
                    model.train()
                    dataloader = datamodule.train_dataloader()
                else:
                    model.eval()
                    dataloader = datamodule.val_dataloader()

                running_loss = 0.0
                running_acc = 0.0
                running_p = 0.0
                running_r = 0.0
                running_f1 = 0.0
                dataset_size = 0.0

                for x, y in tqdm(dataloader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    

                    if phase == 'train':
                        with torch.autocast(device_type='cuda', enabled = use_amp, dtype=self.precision):
                            outputs = model(x).squeeze(1)
                            loss = criterion(outputs, y.to(self.precision))
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    else:
                        with torch.autocast(device_type='cuda', enabled = use_amp, dtype=self.precision):
                            with torch.no_grad():
                                outputs = model(x).squeeze(1)
                                loss = criterion(outputs, y.to(self.precision))
                                

                    running_acc  += acc(outputs, y)*len(y)
                    running_p  += pr(outputs, y)*len(y)
                    running_r  += rc(outputs, y)*len(y)
                    running_f1  += f1(outputs, y)*len(y)
                    running_loss += loss.detach()*len(y)
                    dataset_size += len(y)

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_acc / dataset_size
                epoch_p = running_p / dataset_size
                epoch_r = running_r / dataset_size
                epoch_f1 = running_f1 / dataset_size

                if phase == "valid":
                    print('Loss: {:.4f}'.format(epoch_loss))
                    print('Acc: {:.4f}%'.format(epoch_acc*100))
                    print('Precision: {:.4f}%'.format(epoch_p*100))
                    print('Recall: {:.4f}%'.format(epoch_r*100))
                    print('F1: {:.4f}%'.format(epoch_f1*100))

                if phase=='train':
                    self.exp.log_metrics({'train_loss': epoch_loss, 'train_f1': epoch_f1, 'train_acc': epoch_acc, 'train_precision': epoch_p, 'train_recall': epoch_r}, epoch=epoch)
                elif phase=='valid':
                    self.exp.log_metrics({'valid_loss': epoch_loss, 'valid_f1': epoch_f1, 'valid_acc': epoch_acc, 'valid_precision': epoch_p, 'valid_recall': epoch_r}, epoch=epoch)

                losses.append(epoch_loss)
                accs.append(epoch_acc)
                prs.append(epoch_p)
                rcs.append(epoch_r)
                f1s.append(epoch_f1)

            torch.save({
                'epoch': epoch,
                'loss': epoch_loss,
                'state_dict': model.state_dict(),
                'optimizer_states': optimizer.state_dict()
                }, os.path.join(ckpt_path, 'last.ckpt'))

            if epoch_loss<min_loss and phase=='valid':
                patience_counter = 0
                print("Loss improved by: {:.4f}".format((min_loss-epoch_loss).item()))
                min_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'loss': epoch_loss,
                    'state_dict': model.state_dict(),
                    'optimizer_states': optimizer.state_dict()
                    }, '{}/epoch={}.ckpt'.format(ckpt_path, epoch))
            else:
                patience_counter += 1
                if patience_counter >= early_stop["patience"]:
                    break

        end_time = time.time() - start_time
        print('Elapsed time: {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))

        return losses, accs, prs, rcs, f1s
