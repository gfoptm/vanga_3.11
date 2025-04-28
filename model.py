import torch
import torch.nn as nn
import logging


class LSTMModel(nn.Module):
    """
    PyTorch LSTM-модель для предсказаний на бирже.

    Параметры:
      - num_layers: число LSTM-слоёв (1 или 2)
      - units1: количество нейронов в первом LSTM-слое
      - units2: количество нейронов во втором LSTM-слое (если num_layers==2)
      - dropout_rate: вероятность dropout
      - input_size: число признаков на каждом такте
      - use_genetics: если True, используется упрощённая архитектура (для генетической оптимизации),
                      если False – добавляются дополнительные кастомные слои для биржевых предсказаний.
    """

    def __init__(self, num_layers, units1, units2, dropout_rate, input_size, use_genetics=True):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.use_genetics = use_genetics
        self.dropout_rate = dropout_rate

        # Создаем LSTM-слои (bidirectional)
        if num_layers == 1:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=units1,
                batch_first=True,
                bidirectional=True
            )
            lstm_output_size = 2 * units1
        else:
            self.lstm1 = nn.LSTM(
                input_size=input_size,
                hidden_size=units1,
                batch_first=True,
                bidirectional=True
            )
            self.lstm2 = nn.LSTM(
                input_size=2 * units1,
                hidden_size=units2,
                batch_first=True,
                bidirectional=True
            )
            lstm_output_size = 2 * units2

        # Batch Normalization и Dropout после LSTM
        self.bn = nn.BatchNorm1d(lstm_output_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Полносвязные слои (fully connected)
        if self.use_genetics:
            # Упрощенная архитектура для генетической оптимизации
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_size, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            # Расширенная архитектура с дополнительными кастомными слоями
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )

    def forward(self, x):
        """
        Прямой проход по сети.
        x: тензор формы (batch_size, seq_len, input_size)
        """
        if self.num_layers == 1:
            out, _ = self.lstm(x)
        else:
            out, _ = self.lstm1(x)
            out = self.dropout(out)
            out, _ = self.lstm2(out)

        # Используем выход последнего временного шага
        out = out[:, -1, :]  # (batch, lstm_output_size)
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def build_lstm_model(num_layers, units1, units2, dropout_rate, input_size, use_genetics=True):
    """
    Создает и возвращает экземпляр LSTMModel с заданными параметрами.
    """
    logging.info(f"[Model] Building LSTM model: num_layers={num_layers}, "
                 f"units1={units1}, units2={units2}, dropout={dropout_rate:.2f}, input_size={input_size}, "
                 f"use_genetics={use_genetics}")
    return LSTMModel(num_layers, units1, units2, dropout_rate, input_size, use_genetics)


