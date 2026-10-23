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


class Encoder(nn.Module):
    """编码时间序列到潜在空间"""

    def __init__(self, input_dim=38, hidden_dim=100, latent_dim=20, window_size=120):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.fc_mu = nn.Linear(2 * hidden_dim, latent_dim)  # 双向LSTM输出拼接
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, x):
        # x shape: (batch_size, window_size, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        last_out = out[:, -1, :]
        mu = self.fc_mu(last_out)
        logvar = self.fc_logvar(last_out)
        return mu, logvar


class Generator(nn.Module):
    """从潜在向量重构时间序列"""

    def __init__(self, latent_dim=20, hidden_dim=100, output_dim=38, window_size=120):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.window_size = window_size

    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        h = F.relu(self.fc(z))
        # 扩展为时间序列
        h_seq = h.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.lstm(h_seq)
        reconstructed = self.output_layer(out)
        return reconstructed  # (batch, window_size, output_dim)


class Critic_x(nn.Module):
    """判别真实/生成的时间序列"""

    def __init__(self, input_dim=38, hidden_dim=100):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = torch.mean(x, dim=2)
        return self.fc(x)


class Critic_z(nn.Module):
    """判别潜在空间分布"""

    def __init__(self, latent_dim=20, hidden_dim=50):
        super(Critic_z, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = F.leaky_relu(self.fc1(z), 0.2)
        return self.fc2(h)  # (batch, 1)



def dtw_reconstruction_error_1d(x, x_):
    """
    计算两个一维时间序列之间的DTW误差。

    x, x_ 是一维数组，表示两个时间序列。
    """
    n = len(x)  # 时间序列长度

    # 初始化DTW矩阵
    dtw_matrix = np.full((n + 1, n + 1), np.inf)  # 使用无穷大初始化
    dtw_matrix[0, 0] = 0  # 起点为0

    # 计算DTW矩阵
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cost = abs(x[i - 1] - x_[j - 1])  # 使用绝对值计算距离
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # 删除
                dtw_matrix[i, j - 1],  # 插入
                dtw_matrix[i - 1, j - 1]  # 匹配
            )

    # 返回DTW距离的最小值
    return dtw_matrix[n, n]


def dtw_reconstruction_error_2d(x, x_):
    """
    计算两个二维时间序列之间的DTW误差。

    x, x_ 是形状为 (n, d) 的二维数组，表示两个时间序列，其中：
    - n 是时间步数
    - d 是特征维度数
    """
    n, d = x.shape
    m, d_ = x_.shape

    # 检查输入是否维度匹配
    if d != d_:
        raise ValueError("两个序列的特征维度 d 必须相同！")

    # 初始化DTW矩阵
    dtw_matrix = np.full((n + 1, m + 1), np.inf)  # 使用无穷大初始化
    dtw_matrix[0, 0] = 0  # 起点为0

    # 计算DTW矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 计算两个向量的欧氏距离作为成本
            cost = np.linalg.norm(x[i - 1] - x_[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # 删除
                dtw_matrix[i, j - 1],  # 插入
                dtw_matrix[i - 1, j - 1]  # 匹配
            )

    # 返回DTW距离的最小值
    return dtw_matrix[n, m]


def euclidean_distance(x, x_):
    """
    计算两个多特征向量之间的欧氏距离。

    参数：
    - x: 一维数组，表示第一个样本（多特征向量）。
    - x_: 一维数组，表示第二个样本（多特征向量）。

    返回：
    - 欧氏距离的浮点值。
    """
    if len(x) != len(x_):
        raise ValueError("输入向量的长度必须相等。")

    return np.sqrt(np.sum((x - x_) ** 2))


# 曼哈顿距离
def manhattan_distance(x, x_):
    return np.sum(np.abs(x - x_))


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

    # 输出结果
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

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


def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True, masking_window=0.01, mask=False):
    errors = torch.abs(y - y_hat)[:, 0]
    if not smooth:
        return errors

    smoothing_window = max(1, int(len(y) * smoothing_window))
    errors = pd.Series(errors.numpy()).ewm(span=smoothing_window).mean().values
    errors = torch.tensor(errors)

    if mask:
        mask_length = int(masking_window * len(errors))
        errors[:mask_length] = torch.min(errors)
    return errors


def _point_wise_error(y, y_hat):
    return torch.abs(y - y_hat)


def _area_error(y, y_hat, score_window=10):
    smooth_y = pd.Series(y.numpy()).rolling(score_window, center=True, min_periods=score_window // 2).apply(
        lambda x: torch.trapz(torch.tensor(x)).item())
    smooth_y_hat = pd.Series(y_hat.numpy()).rolling(score_window, center=True, min_periods=score_window // 2).apply(
        lambda x: torch.trapz(torch.tensor(x)).item())
    errors = torch.abs(torch.tensor(smooth_y.values) - torch.tensor(smooth_y_hat.values))
    return errors


def _dtw_error(y, y_hat, score_window=10):
    length_dtw = (score_window // 2) * 2 + 1
    half_length_dtw = length_dtw // 2

    y_pad = torch.nn.functional.pad(y, (half_length_dtw, half_length_dtw), value=0)
    y_hat_pad = torch.nn.functional.pad(y_hat, (half_length_dtw, half_length_dtw), value=0)

    similarity_dtw = []
    for i in range(len(y) - length_dtw):
        true_data = y_pad[i:i + length_dtw].flatten().numpy()
        pred_data = y_hat_pad[i:i + length_dtw].flatten().numpy()
        dist = dtw_reconstruction_error_1d(true_data, pred_data)
        similarity_dtw.append(dist)

    errors = ([0] * half_length_dtw + similarity_dtw + [0] * (len(y) - len(similarity_dtw) - half_length_dtw))
    return torch.tensor(errors)


def _compute_critic_score(critics: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """Compute an array of anomaly scores using EWMA.

    Args:
        critics (torch.Tensor): Critic values.
        alpha (float): Smoothing factor for EWMA.

    Returns:
        torch.Tensor: Array of anomaly scores.
    """
    critics = critics.clone().detach()
    critics = critics.squeeze()
    l_quantile = torch.quantile(critics, 0.25)
    u_quantile = torch.quantile(critics, 0.75)
    in_range = (critics >= l_quantile) & (critics <= u_quantile)
    critic_mean = torch.mean(critics[in_range])
    critic_std = torch.std(critics)

    z_scores = torch.abs((critics - critic_mean) / critic_std) + 1
    smoothed_scores = torch.zeros_like(z_scores)
    smoothed_scores[0] = z_scores[0]
    for t in range(1, len(z_scores)):
        smoothed_scores[t] = alpha * z_scores[t] + (1 - alpha) * smoothed_scores[t - 1]
    return smoothed_scores



class LTadGAN(nn.Module):
    def __init__(self, batch_size, input_dim, latent_dim, hidden_dim, learning_rate, squ_size, dataset_name, path):
        super(LTadGAN, self).__init__()
        # 模块初始化（修正参数传递）
        self.encoder = Encoder(input_dim=input_dim,
                               hidden_dim=hidden_dim,  # 使用传入的hidden_dim
                               latent_dim=latent_dim,
                               window_size=squ_size)

        self.generator = Generator(latent_dim=latent_dim,
                                   hidden_dim=hidden_dim,  # 使用传入的hidden_dim
                                   output_dim=input_dim,
                                   window_size=squ_size)

        self.critic_x = Critic_x(input_dim=input_dim,
                                hidden_dim=hidden_dim)  # 修正为CriticX

        self.critic_z = Critic_z(latent_dim=latent_dim,
                                hidden_dim=hidden_dim)  # 修正为CriticZ

        # 参数配置
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.window_size = squ_size  # 重命名squ_size为更规范的window_size
        self.lr = learning_rate
        self.dataset_name = dataset_name
        self.path = path

        # 优化器配置
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.critic_x_optimizer = optim.Adam(self.critic_x.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.critic_z_optimizer = optim.Adam(self.critic_z.parameters(), lr=self.lr, betas=(0.5, 0.9))

        # 设备配置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # 学习率调度（增加更灵活的配置）
        self.scheduler_config = {
            'factor': 0.5,
            'patience': 20,  # 增大耐心值
            'threshold': 1e-3,  # 设置最小改善阈值
            'min_lr': 1e-6  # 防止学习率过低
        }
        self._init_schedulers()

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ Xavier初始化 """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def _init_schedulers(self):
        """ 统一管理学习率调度器 """
        self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode='min', **self.scheduler_config)
        self.generator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.generator_optimizer, mode='min', **self.scheduler_config)
        self.critic_x_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_x_optimizer, mode='min', **self.scheduler_config)
        self.critic_z_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_z_optimizer, mode='min', **self.scheduler_config)

    def gradient_penalty(self, real, fake, critic):
        """ 带梯度裁剪的梯度惩罚 """
        alpha = torch.rand(real.size(0), 1, 1, device=self.device)  # 修正维度匹配问题
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)

        d_interpolated = critic(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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

    def forward(self, x):
        """ 统一前向传播接口 """
        z_mu, z_logvar = self.encoder(x)
        z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)  # 重参数化
        x_rec = self.generator(z)
        return x_rec, z_mu, z_logvar

    def fit_with_kfold(self, train_loader, val_loader, epochs, critic_iter=1, gp_weight=50):
        """ 训练逻辑优化 """
        best_loss = float('inf')
        best_model_path = os.path.join(self.path, f'{self.dataset_name}_lstm_best_model.pth')

        for epoch in range(epochs):
            # 训练阶段
            self.train()
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.float().to(self.device)

                # 数据增强：添加时序噪声
                x = x + torch.randn_like(x) * 0.1 * torch.std(x, dim=(1, 2), keepdim=True)

                # 训练Critic X
                for _ in range(critic_iter):
                    self.critic_x_optimizer.zero_grad()
                    z = torch.randn(x.size(0), self.latent_dim, device=self.device)
                    x_fake = self.generator(z).detach()

                    real_scores = self.critic_x(x)
                    fake_scores = self.critic_x(x_fake)
                    gp = self.gradient_penalty(x, x_fake, self.critic_x)

                    loss_cx = -0.9*torch.mean(real_scores) + 0.9*torch.mean(fake_scores) + gp_weight * gp
                    loss_cx.backward()
                    nn.utils.clip_grad_norm_(self.critic_x.parameters(), 0.5)
                    self.critic_x_optimizer.step()

                # 训练Critic Z
                for _ in range(1):
                    self.critic_z_optimizer.zero_grad()
                    z_real = torch.randn(x.size(0), self.latent_dim, device=self.device)
                    z_fake = self.encoder(x)[0].detach()

                    real_scores_z = self.critic_z(z_real)
                    fake_scores_z = self.critic_z(z_fake)
                    gp_z = self.gradient_penalty(z_real, z_fake, self.critic_z)

                    loss_cz = -0.9*torch.mean(real_scores_z) + 0.9*torch.mean(fake_scores_z) + 10 * gp_z
                    loss_cz.backward()
                    nn.utils.clip_grad_norm_(self.critic_z.parameters(), 1.0)
                    self.critic_z_optimizer.step()

                # 训练Generator & Encoder
                self.encoder_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                self._freeze_critics()

                x_rec, z_mu, z_logvar = self(x)

                # KL散度正则项
                kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

                # 重构损失
                mse_loss = F.mse_loss(x_rec, x, reduction='mean')

                # 对抗损失
                adv_loss_x = -torch.mean(self.critic_x(x_rec).detach())
                adv_loss_z = -torch.mean(self.critic_z(z_mu).detach())

                # 总损失
                total_loss = 2*mse_loss + 0.1 * kl_loss + 0.1 * (adv_loss_x + adv_loss_z)
                z_reg = 0.01 * torch.norm(z_mu, p=2, dim=1).mean()
                total_loss += z_reg
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)

                self.encoder_optimizer.step()
                self.generator_optimizer.step()
                self._unfreeze_critics()

            print(f"Epoch [{epoch + 1}/{epochs}] - Generator Loss: {total_loss.item():.4f}, Critic_x: {loss_cx.item():.4f}, Critic_z: {loss_cz.item():.4f}"
                  f"encoder_loss:{adv_loss_z.item():.4f}"
                  f"generator_loss:{adv_loss_x.item():.4f}"
                  f"mse:{mse_loss.item():.4f}")

            # 验证阶段
            self.eval()
            val_losses = []
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.float().to(self.device)
                    x_rec, _, _ = self.forward(x)
                    val_loss = F.mse_loss(x_rec, x, reduction='mean')
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)

            # 学习率调度
            self.encoder_scheduler.step(avg_val_loss)
            self.generator_scheduler.step(avg_val_loss)
            self.critic_x_scheduler.step(avg_val_loss)
            self.critic_z_scheduler.step(avg_val_loss)

            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.state_dict(), best_model_path)
                print(f"Epoch {epoch + 1}: Best model saved with val_loss {avg_val_loss:.4f}")

        # 加载最佳模型
        self.load_state_dict(torch.load(best_model_path))
        print("Training completed. Best model loaded.")

    def predict(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32)
        z_ = self.encoder(X)
        y_hat = self.generator(z_)
        return y_hat

    def test(self, test_loader, step_size=10):
        all_scores = []
        all_labels = []
        batch_rec_error = []
        with torch.no_grad():
            for batch, (sample, labels) in enumerate(test_loader):
                sample = sample.to(self.device)
                labels = labels.to(self.device)

                # Forward pass through the generator
                recon,_,_ = self.forward(sample)
                critic_score = self.critic_x(recon)
                critic_scores = critic_score.cpu().numpy()

                mse_loss = torch.mean((sample - recon) ** 2, dim=2)  # (256, 10)
                batch_rec_error = torch.tensor(mse_loss)
                # Combine critic and reconstruction scores
                scores = critic_scores
                # is_anomaly = (scores > threshold).int()

                # Store results
                all_scores.append(mse_loss.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_scores = np.concatenate([s.flatten() for s in all_scores])
        all_labels = np.concatenate([l.flatten() for l in all_labels])

        # 定义平滑窗口大小（例如：batch_size * window_size 的 5%）
        smoothing_window = int(self.batch_size * self.window_size * 0.05)
        # 利用 Pandas 的 ewm 进行指数加权移动平均
        all_scores = pd.Series(all_scores).ewm(span=smoothing_window).mean().values
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)

        # 计算F1-score并找到最佳阈值
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Best threshold: {best_threshold}")

        # 使用最佳阈值进行预测
        predicted_anomalies = (all_scores > best_threshold).astype(int)

        # Prune false positives
        # predicted_anomalies = prune_false_positive(predicted_anomalies, all_scores, threshold)

        # Compute evaluation metrics
        accuracy, precision, recall, f1_score = find_scores(all_labels, predicted_anomalies)
        auc = roc_auc_score(all_labels, all_scores)

        print(f"AUC: {auc:.2f}")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

        return auc, accuracy, precision, recall, f1_score
