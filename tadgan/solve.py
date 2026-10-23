#!/usr/bin/env python
# coding: utf-8


from tadGAN import *
from dataload import *
from lstm_tadgan import LTadGAN
from TadAGAN import *

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build_model(self):
        model_cls = TadAGAN if self.model_name == 'tadagan' else TadGAN
        self.model = model_cls(
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            learning_rate=self.lr,
            squ_size=self.win_size,
            dataset_name=self.dataset,
            path=self.model_save_path
        )
        if torch.cuda.is_available():
            self.model.cuda()

    def train(self):

        print("======================TRAIN MODE======================")
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.fit(self.train_loader,self.vali_loader,epochs=self.num_epochs)
        #EarlyStopping(C_loss_x,C_loss_z,Genereation_loss,self.model,path)


    def test(self):
        print("======================TEST MODE======================")
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_best_model.pth')),strict=False)
        self.model.eval()
        self.model.test(self.test_loader)





