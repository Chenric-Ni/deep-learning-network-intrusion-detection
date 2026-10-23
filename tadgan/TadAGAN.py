import logging
import os
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import torch.backends.cudnn as cudnn


# 用于实现 TimeDistributed 层：对时间步的每个样本独立应用一个子模块
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x shape: (batch, time_steps, features)
        batch_size, time_steps = x.size(0), x.size(1)
        # 将时间步和 batch 合并
        x_reshaped = x.contiguous().view(batch_size * time_steps, -1)
        y = self.module(x_reshaped)
        # 恢复成 (batch, time_steps, output_dim)
        y = y.view(batch_size, time_steps, -1)
        return y
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq, embed_dim)
        attn_output, _ = self.attn(x, x, x)  # 自注意力输出
        # 残差连接
        x = x + self.dropout(attn_output)
        # 归一化
        x = self.norm(x)
        return x
# Encoder 部分：由双向 LSTM、Flatten、全连接层和 Reshape 组成
class Encoder(nn.Module):
    def __init__(self, input_shape, encoder_reshape_shape, lstm_units=100, dense_units=20):
        """
        input_shape: (seq_length, num_features)
        lstm_units: LSTM 隐藏单元数（单向）
        dense_units: 全连接层输出单元数（应与 encoder_reshape_shape 的乘积相等）
        encoder_reshape_shape: 例如 (a, b)，要求 a*b == dense_units
        num_heads: 注意力头的数量
        """
        super(Encoder, self).__init__()
        self.seq_length, self.num_features = input_shape
        self.input_dropout = nn.Dropout(0.2)
        # 双向 LSTM：输出特征维度为 2*lstm_units
        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=lstm_units,
                            batch_first=True,
                            bidirectional=True)
        
        self.flatten = nn.Flatten()
        # LSTM 输出 shape: (batch, seq_length, 2*lstm_units) → 展平后维度为 seq_length*2*lstm_units
        self.fc = nn.Linear(self.seq_length * 2 * lstm_units, int(np.prod(encoder_reshape_shape)))
        self.encoder_reshape_shape = encoder_reshape_shape
        self.timedist = TimeDistributed(nn.Linear(2*lstm_units, dense_units))

    def forward(self, x):
        # x: (batch, seq_length, num_features)
        x = self.input_dropout(x)
        # 通过 LSTM 层
        lstm_out, _ = self.lstm(x)  # → (batch, seq_length, 2*lstm_units)
        
        x = self.flatten(lstm_out)  # → (batch, seq_length * 2*lstm_units)
        x = self.fc(x)               # → (batch, dense_units)
        
        # reshape 到指定形状，例如 (batch, a, b)
        if self.encoder_reshape_shape is not None:
            x = x.view(x.size(0), *self.encoder_reshape_shape)
        
        return x

# Generator 部分：从隐变量生成重构信号
class Generator(nn.Module):
    def __init__(self, fc_dim, latent_dim, generator_reshape_shape, output_dim, num_heads=4):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.generator_reshape_shape = generator_reshape_shape
        self.output_dim = output_dim
        generator_reshape_dim = int(np.prod(generator_reshape_shape))
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(fc_dim, generator_reshape_dim)
        self.lstm1 = nn.LSTM(input_size=generator_reshape_shape[-1],
                             hidden_size=64,
                             batch_first=True,
                             bidirectional=True)
        # 添加注意力块
        self.attn_block = AttentionBlock(embed_dim=128, num_heads=num_heads, dropout=0.2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lstm2 = nn.LSTM(input_size=128,
                             hidden_size=64,
                             batch_first=True,
                             bidirectional=True)
        self.timedist = TimeDistributed(nn.Linear(128, self.output_dim))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(x.size(0), *self.generator_reshape_shape)
        
        x, _ = self.lstm1(x)  # 输出形状: (batch, seq_gen, 128)
        # 在 lstm1 输出后添加注意力块

        
        x = x.transpose(1, 2)
        x = self.upsample(x)
        x = x.transpose(1, 2)
        x = self.attn_block(x)  # 输出形状依然为 (batch, seq_gen, dim)
        x, _ = self.lstm2(x)
        x = self.timedist(x)
        x = self.tanh(x)
        return x
# CriticX 部分：对输入时序信号（例如原始或重构信号）进行判别
class CriticX(nn.Module):
    def __init__(self, input_channels, seq_length, num_heads=8):
        """
        input_channels: 输入信号的通道数（例如 1 表示单变量时序数据）
        seq_length: 输入时序数据的长度
        """
        super(CriticX, self).__init__()
        self.seq_length = seq_length
        self.input_channels = input_channels
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channels, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            #nn.Flatten()
        )
        # 最后全连接层：输入维度为 64*seq_length
        self.fc = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, channels, seq_length)
        x = x.transpose(1, 2)  # → (batch, seq_length, channels)
        x = self.model(x)
        x = self.fc(x)
        return x

# CriticZ 部分：对编码得到的隐变量进行判别
class CriticZ(nn.Module):
    def __init__(self, input_dim, dense_units, seq_length):
        """
        input_dim: 隐变量（Encoder 输出）展平后的维度，即 prod(encoder_reshape_shape)
        dense_units: 全连接层单元数（JSON 中 dense_units）
        """
        super(CriticZ, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, dense_units),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dense_units, dense_units),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dense_units, 1*seq_length)
        )
        self.seq_length = seq_length

    def forward(self, z):
        z = self.model(z)
        return z.view(z.size(0), self.seq_length)



def wasserstein_loss(x, x_):
    return np.mean(x * x_)


def prune_false_positive(is_anomaly, anomaly_score, change_threshold):
    """
    消除伪阳性以减少误报，逐行处理一维输入。
    输入：
    - is_anomaly: 形状为 (squ_len,) 的一维数组，表示每个样本的异常标签。
    - anomaly_score: 形状为 (squ_len,) 的一维数组，表示每个样本的异常分数。
    - change_threshold: 用于过滤伪阳性的阈值，如果某段异常序列的最大分数小于该阈值，则认为是伪阳性并移除。

    输出：
    - is_anomaly: 更新后的异常标签数组。
    """
    squ_len = len(is_anomaly)  # 获取输入的长度
    seq_details = []
    start, end = 0, 0
    max_score = anomaly_score[0]

    # 遍历当前序列
    for j in range(1, squ_len):
        if j == squ_len or (is_anomaly[j] == 1 and is_anomaly[j - 1] == 0):  # 判断当前为异常段的结束
            end = j
            seq_details.append((start, end, max_score))
        elif is_anomaly[j] == 1 and is_anomaly[j - 1] == 0:  # 当前为异常段的开始
            start = j
            max_score = anomaly_score[j]
        elif is_anomaly[j] == 1:  # 继续更新当前异常段的最大分数
            max_score = max(max_score, anomaly_score[j])

    # 根据阈值移除伪阳性
    for seq in seq_details:
        if seq[2] < change_threshold:
            is_anomaly[seq[0]:seq[1] + 1] = 0  # 将伪阳性部分的异常标签设为0

    return is_anomaly


def find_scores(y_true, y_predict):
    """
    计算分类指标：准确率、精确率、召回率、F1 分数，支持二维输入。

    输入:
    - y_true: 形状为 (batch_size, sequence_length) 的二维数组，表示真实标签。
    - y_predict: 形状为 (batch_size, sequence_length) 的二维数组，表示预测标签。

    输出:
    - accuracy: 所有序列的整体准确率
    - precision: 所有序列的整体精确率
    - recall: 所有序列的整体召回率
    - f1_score: 所有序列的整体 F1 分数
    """
    # 将二维数组展平以计算整体指标
    y_true_flat = y_true.flatten()
    y_predict_flat = y_predict.flatten()

    # 计算分类指标
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true_flat, y_predict_flat))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true_flat, y_predict_flat))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true_flat, y_predict_flat))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true_flat, y_predict_flat))

    # 避免分母为零
    accuracy = (tp + tn) / len(y_true_flat)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


    # 返回指标值以便进一步处理
    return accuracy, precision, recall, f1_score


def initialize_weights(model):
    """
    Initialize weights for the model using Xavier initialization for linear layers
    and Kaiming initialization for GRU layers.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization for linear layers
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Bias set to zero
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:  # Input-hidden weights
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:  # Hidden-hidden weights
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)  # Bias set to zero

def dtw_distance_1d(x, x_):
    """
    计算两个一维时间序列之间的DTW距离。

    x, x_ 是形状为 (dim,) 的一维数组，表示两个时间序列的单个时间步。
    """
    n, m = x.shape[0], x_.shape[0]
    
    # 初始化DTW矩阵
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0  # 起点为0

    # 计算DTW矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(x[i - 1] - x_[j - 1])  # 使用欧几里得距离
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # 删除
                dtw_matrix[i, j - 1],  # 插入
                dtw_matrix[i - 1, j - 1]  # 匹配
            )

    # 返回DTW距离的最小值
    return dtw_matrix[n, m]


def reconstruction_loss(y, y_hat):
    """
    计算重建损失，使用DTW距离。
    
    y: 原始输入，形状为 (batch_size, seq_len, dim)
    y_hat: 重建输出，形状为 (batch_size, seq_len, dim)
    
    返回:
    - loss: 形状为 (batch_size, seq_len) 的重建损失
    """
    batch_size = y.shape[0]
    seq_len = y.shape[1]
    loss = np.zeros((batch_size, seq_len))  # 初始化损失数组

    for i in range(batch_size):
        # 对每个时间步计算DTW距离
        for t in range(seq_len):  # 遍历每个时间步
            loss[i, t] = dtw_distance_1d(y[i, t], y_hat[i, t])

    return loss  # 返回形状为 (batch_size, seq_len) 的损失

def compute_zscore(x, axis=1):
    """
    对输入数组 x 沿指定轴计算 z-score，
    保留原始时序结构。
    
    输入:
      - x: numpy 数组，形状 (..., T) 或 (batch, T, features)（此处假设 T 为时间步维度）
      - axis: 计算均值和标准差的轴，默认为1（针对每个样本沿时间维度计算）
      
    输出:
      - z: 与 x 同形状的 z-score 数组
    """
    mean_x = np.mean(x, axis=axis, keepdims=True)
    std_x = np.std(x, axis=axis, keepdims=True)
    return (x - mean_x) / (std_x + 1e-10)


class TadAGAN(nn.Module):
    def __init__(self, batch_size, input_dim, latent_dim, hidden_dim, learning_rate, squ_size, dataset_name, path):
        super(TadAGAN, self).__init__()

        # Encoder 和 Generator 互换 hidden_dim
        self.enc_seq = int(squ_size/5)
        self.input_shape = (squ_size, input_dim)
        self.encoder_reshape_shape = (self.enc_seq, latent_dim)
        self.generator_reshape_shape = (int(squ_size/2), latent_dim)
        self.encoder = Encoder(self.input_shape, self.encoder_reshape_shape)
        self.generator = Generator(int(latent_dim*self.enc_seq), latent_dim, self.generator_reshape_shape, input_dim)
        self.critic_x = CriticX(input_dim, squ_size)
        self.critic_z = CriticZ(int(latent_dim*self.enc_seq), latent_dim, squ_size)


        self.batch_size = batch_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        #self.num_layers = num_layers
        self.lr = learning_rate
        self.squ_size = squ_size
        self.dataset_name = dataset_name
        self.path = path

        # Optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.critic_x_optimizer = optim.Adam(self.critic_x.parameters(), lr=self.lr)
        self.critic_z_optimizer = optim.Adam(self.critic_z.parameters(), lr=self.lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Learning rate schedulers
        self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.generator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.generator_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.critic_x_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.critic_x_optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.critic_z_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.critic_z_optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Initialize weights
        initialize_weights(self.encoder)
        initialize_weights(self.generator)
        initialize_weights(self.critic_x)
        initialize_weights(self.critic_z)

    def wasserstein_loss(self, y_true, y_pred):
        return torch.mean(y_true * y_pred)

    def gradient_penalty(self, real, fake, critic):
        #with cudnn.flags(enabled=False):
            alpha = torch.rand_like(real, device=real.device)
            interpolated = alpha * real + (1 - alpha) * fake
            interpolated.requires_grad_(True)
            d_interpolated = critic(interpolated)
            grad_outputs = torch.ones_like(d_interpolated, device=real.device)
            gradients = torch.autograd.grad(
                outputs=d_interpolated,
                inputs=interpolated,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            gradients = gradients.reshape(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def _freeze_generator(self):
        for p in self.generator.parameters():
            p.requires_grad_(False) 
    
    def _unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def _unfreeze_generator(self):
        for p in self.generator.parameters():
            p.requires_grad_(True)  

    def _freeze_critics(self):
        for p in self.critic_x.parameters():
            p.requires_grad_(False)
        for p in self.critic_z.parameters():
            p.requires_grad_(False)

    # 在生成器训练步骤结束后，解冻判别器参数
    def _unfreeze_critics(self):
        for p in self.critic_x.parameters():
            p.requires_grad_(True)
        for p in self.critic_z.parameters():
            p.requires_grad_(True)

    def forward(self, x, z):
        z_ = self.encoder(x)
        x_ = self.generator(z)
        rec = self.generator(z_)
        cx_fake = self.critic_x(x_).detach()
        cz_fake = self.critic_z(z_).detach()
        cx_real = self.critic_x(x).detach()
        cz_real = self.critic_z(z).detach()
        return cx_fake, cz_fake, rec, cx_real, cz_real
    
    def rec(self, x):
        z_ = self.encoder(x)
        rec = self.generator(z_)
        return rec  

    def fit(self, X, X_val, epochs, iterations_critic=3):
        logging.basicConfig(filename='training.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        best_loss = float('inf')
        best_model_path = os.path.join(self.path, f'{self.dataset_name}_best_model.pth')

        for epoch in range(epochs):
            self.train()
            for i, (x, _) in enumerate(X):
                x = x.float().to(self.device)
                batch_size, squ_len, _ = x.size()
                
                for _ in range(iterations_critic):
                    self._freeze_encoder()
                    self._freeze_generator()
                    self._unfreeze_critics()
                    self.critic_x_optimizer.zero_grad()
                    noise = torch.randn_like(x) * 0.1
                    x_noise = x + noise
                    z = torch.normal(mean=0, std=1, size=(batch_size, int(self.enc_seq), self.latent_dim)).to(x.device)
                    cx_real = self.critic_x(x_noise)
                    cx = self.generator(z).detach()
                    cx_fake = self.critic_x(cx)
                    gp_x = self.gradient_penalty(x_noise, cx, self.critic_x)
                    loss_cx = self.wasserstein_loss(-0.9*torch.ones_like(cx_real), cx_real) + \
                              self.wasserstein_loss(0.9*torch.ones_like(cx_fake), cx_fake) + 10 * gp_x
                    loss_cx.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_x.parameters(), max_norm=1.0)
                    self.critic_x_optimizer.step()
                    self.critic_z_optimizer.zero_grad()
                    cz_real = self.critic_z(z)
                    cz = self.encoder(x_noise).detach()
                    cz_fake = self.critic_z(cz)
                    gp_z = self.gradient_penalty(z, cz, self.critic_z)
                    loss_cz = self.wasserstein_loss(-0.9*torch.ones_like(cz_real), cz_real) + \
                              self.wasserstein_loss(0.9*torch.ones_like(cz_fake), cz_fake) + 10 * gp_z
                    loss_cz.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_z.parameters(), max_norm=1.0)
                    self.critic_z_optimizer.step()

                for _ in range(1):
                    self.encoder_optimizer.zero_grad()
                    self.generator_optimizer.zero_grad()
                    self._freeze_critics()
                    self._unfreeze_encoder()
                    self._unfreeze_generator()
                    #noise = torch.randn_like(x) * 0.05
                    x_noise = x
                    # 修改为使 z 服从正态分布
                    z = torch.normal(mean=0, std=1, size=(batch_size, int(self.enc_seq), self.latent_dim)).to(x.device)
                    cx_fake, cz_fake, x_rec, cx_real, cz_real = self.forward(x_noise, z)
                    loss_gen_d = self.wasserstein_loss(-0.9*torch.ones_like(cx_fake), cx_fake) + 0.9*self.wasserstein_loss(torch.ones_like(cx_real), cx_real)
                    loss_gen_e = self.wasserstein_loss(-0.9*torch.ones_like(cz_fake), cz_fake) + 0.9*self.wasserstein_loss(torch.ones_like(cz_real), cz_real)
                    loss_L2norm = torch.norm(x_rec - x_noise, p=2,dim=2).mean()
                    if epoch < 0:
                        loss_gen = 15 * loss_L2norm + 1.0 * loss_L2norm
                    else:
                        loss_gen = 10 * loss_L2norm + 1 * (
                                    loss_gen_d + loss_gen_e)
                    loss_gen.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                    self.encoder_optimizer.step()
                    self.generator_optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}] - Generator Loss: {loss_gen.item():.4f}, Critic_x: {loss_cx.item():.4f}, Critic_z: {loss_cz.item():.4f}\n"
                  f"encoder_loss:{loss_gen_e.item():.4f},"
                  f"generator_loss:{loss_gen_d.item():.4f},"
                  f"L2norm:{loss_L2norm.item():.4f}"
                  )

            # Validation
            self.eval()
            batch_loss = []
            for x, _ in X_val:
                x = x.float().to(self.device)
                rec = self.rec(x)
                loss = torch.norm(rec - x, p=2,dim=2).mean()
                batch_loss.append(loss.detach().cpu().numpy())
            val_loss = np.average(batch_loss)
            print(f'Val_loss{val_loss:.6f}')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.state_dict(), best_model_path)
                print(f'Best model saved at epoch {epoch + 1}')

            self.encoder_scheduler.step(val_loss)
            self.generator_scheduler.step(val_loss)
            self.critic_x_scheduler.step(val_loss)
            self.critic_z_scheduler.step(val_loss)

        self.load_state_dict(torch.load(best_model_path))

    def predict(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32)
        z_ = self.encoder(X)
        y_hat = self.generator(z_)
        return y_hat

    def test(self, test_loader, step_size=10):
        self.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch, (sample, labels) in enumerate(test_loader):
                sample = sample.to(self.device)
                labels = labels.to(self.device)

                # Forward pass through the generator
                z_ = self.encoder(sample)
                recon = self.generator(z_)
                critic_score = self.critic_x(recon)
                critic_scores = critic_score.squeeze(1).cpu().numpy()

                # Calculate reconstruction error
                rec_error = torch.norm(recon - sample, p=2, dim=2).cpu().numpy()

                # Normalize scores using z-score
                rec_error_mean = np.mean(rec_error)
                rec_error_std = np.std(rec_error)
                critic_scores_mean = np.mean(critic_scores)
                critic_scores_std = np.std(critic_scores)

                if rec_error_std != 0:
                    rec_error_norm = (rec_error - rec_error_mean) / rec_error_std
                if critic_scores_std != 0:
                    critic_scores_norm = (critic_scores - critic_scores_mean) / critic_scores_std

                # Combine critic and reconstruction scores
                scores = rec_error

                # Store results
                all_scores.append(scores.flatten())
                all_labels.append(labels.cpu().numpy().flatten())

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        smoothing_window = int(self.batch_size * self.squ_size * 0.05)
        # 利用 Pandas 的 ewm 进行指数加权移动平均
        all_scores = pd.Series(all_scores).ewm(span=smoothing_window).mean().values

        # Apply sliding window to compute local thresholds
        window_size = len(all_scores)
        step_size = window_size
        anomalies = np.zeros_like(all_scores, dtype=bool)

        for start in range(0, len(all_scores) - window_size + 1, step_size):
            end = start + window_size
            window_scores = all_scores[start:end]
            mean = np.mean(window_scores)
            std = np.std(window_scores)
            threshold = mean
            anomalies[start:end] = window_scores > threshold

        # Calculate evaluation metrics
        accuracy, precision, recall, f1_score = find_scores(all_labels, anomalies)
        auc = roc_auc_score(all_labels, all_scores)
        
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

        return auc, accuracy, precision, recall, f1_score