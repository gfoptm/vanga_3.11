# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMModel(nn.Module):
    """
    Усиленная LSTM-модель с:
      - две LSTM-секции с разными hidden-размерами (units1 и units2)
      - рекуррентный dropout между слоями
      - LayerNorm
      - self-attention
      - pack_padded_sequence для переменной длины
      - residual FC-блоки
    """
    def __init__(
        self,
        input_size: int,
        units1: int,
        units2: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
        num_fc_layers: int = 2,
        fc_hidden: int = 128
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_attention = use_attention

        # первая LSTM-секция
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=units1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        # вторая LSTM-секция (dropout между слоями)
        if num_layers == 2:
            self.lstm2 = nn.LSTM(
                input_size=2*units1,
                hidden_size=units2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=dropout
            )
            lstm_out_dim = 2 * units2
        else:
            lstm_out_dim = 2 * units1

        # нормализация и общий dropout
        self.layernorm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)

        # self-attention
        if use_attention:
            self.attn_linear = nn.Linear(lstm_out_dim, lstm_out_dim)

        # residual-FC блоки
        fc_dims = [lstm_out_dim] + [fc_hidden]*num_fc_layers
        self.fc_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(fc_dims, fc_dims[1:])
        ])
        self.output_layer = nn.Linear(fc_dims[-1], 2)

        self._init_weights()
        logging.info(f"[Model] LSTMModel init: in={input_size}, u1={units1}, u2={units2}, "
                     f"layers={num_layers}, dropout={dropout}, attention={use_attention}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        batch = x.size(0)

        # pack_padded_sequence, если lengths заданы
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm1(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            h0 = x.new_zeros(2, batch, self.lstm1.hidden_size)
            c0 = x.new_zeros(2, batch, self.lstm1.hidden_size)
            out, _ = self.lstm1(x, (h0, c0))

        # вторая LSTM-секция
        if self.num_layers == 2:
            out = self.dropout(out)
            h1 = out.new_zeros(2, batch, self.lstm2.hidden_size)
            c1 = out.new_zeros(2, batch, self.lstm2.hidden_size)
            out, _ = self.lstm2(out, (h1, c1))

        # self-attention
        if self.use_attention:
            scores = torch.bmm(out, out.transpose(1,2)) / (out.size(-1)**0.5)
            weights = F.softmax(scores, dim=-1)
            out = torch.bmm(weights, out)

        # последний тайм-степ
        out = out[:, -1, :]
        out = self.layernorm(out)
        out = self.dropout(out)

        # residual FC-блоки
        x_fc = out
        for fc in self.fc_layers:
            y = F.relu(fc(x_fc))
            y = self.dropout(y)
            x_fc = x_fc + y

        return self.output_layer(x_fc)


def build_lstm_model(
    num_layers: int,
    units1: int,
    units2: int,
    dropout_rate: float,
    input_size: int,
    use_genetics: bool = True
) -> LSTMModel:
    """
    Если use_genetics=True — делаем 1 FC-слой (fc_hidden=64),
    иначе 2 FC-слоя (fc_hidden=128). Второй LSTM-скрытый размер = units2.
    """
    if use_genetics:
        fc_layers, fc_hidden = 1, 64
    else:
        fc_layers, fc_hidden = 2, 128

    model = LSTMModel(
        input_size=input_size,
        units1=units1,
        units2=units2,
        num_layers=num_layers,
        dropout=dropout_rate,
        use_attention=True,
        num_fc_layers=fc_layers,
        fc_hidden=fc_hidden
    )
    return model

