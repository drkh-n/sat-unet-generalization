from lightning.pytorch.callbacks import  EarlyStopping


class ResumedEarlyStopping(EarlyStopping):
    def setup(self, trainer, pl_module, stage=None):
        print(f"EarlyStopping run on stage: {stage}")
        trainer.should_stop = False