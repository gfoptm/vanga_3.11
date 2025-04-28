import logging
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from deap import base, creator, tools

from model import build_lstm_model
from config import window_size, POPULATION_SIZE, GENERATIONS

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# === ФУНКЦИЯ ДЛЯ FEATURE ENGINEERING ===
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет технические индикаторы и дополнительные фичи:
    логарифмическая доходность, волатильность, моментум и т.д.
    Возвращает DataFrame без NaN/inf.
    """
    logging.info(f"[Features] Engineering on dataframe of shape {df.shape}")
    from ta import add_all_ta_features
    df = add_all_ta_features(df, open="open", high="high", low="low",
                             close="close", volume="volume", fillna=True)

    df["log_return"]  = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    df["volatility"]  = df["log_return"].rolling(10).std().fillna(0)
    df["momentum"]    = df["close"] - df["close"].shift(10)
    df["ma_ratio"]    = df["close"] / df["trend_sma_fast"]
    df["price_range"] = df["high"] - df["low"]
    df["volume_delta"]= df["volume"].diff().fillna(0)
    df["zscore_close"]= (
        df["close"] - df["close"].rolling(20).mean()
    ) / df["close"].rolling(20).std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    logging.info(f"[Features] Final shape after dropna: {df.shape}")
    return df

# === DEAP: ГЕНЕТИЧЕСКАЯ ОПТИМИЗАЦИЯ ===
try:
    creator.FitnessMax
except:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.Individual
except:
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("num_layers",   random.choice, [1,2])
toolbox.register("units1",       random.randint, 32, 256)
toolbox.register("dropout_rate", random.uniform, 0.1, 0.5)
toolbox.register("units2",       random.randint, 32, 256)

def create_individual():
    nl = toolbox.num_layers()
    if nl == 1:
        return creator.Individual([nl, toolbox.units1(), toolbox.dropout_rate()])
    else:
        return creator.Individual([nl, toolbox.units1(),
                                   toolbox.dropout_rate(), toolbox.units2()])

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def check_bounds(ind):
    # num_layers
    ind[0] = int(round(ind[0]))
    ind[0] = max(1, min(2, ind[0]))
    # units1
    ind[1] = int(round(ind[1]))
    ind[1] = max(32, min(256, ind[1]))
    # dropout
    ind[2] = max(0.1, min(0.5, ind[2]))
    # units2 если нужно
    if ind[0] == 2:
        if len(ind) < 4:
            ind.append(random.randint(32,256))
        else:
            ind[3] = int(round(ind[3]))
            ind[3] = max(32, min(256, ind[3]))
    return ind

def evaluate_model(individual, X, y):
    """
    Кросс-валидация по TimeSeriesSplit.
    Возвращает (avg_accuracy,).
    """
    nl, u1, dr = individual[0], individual[1], individual[2]
    u2 = individual[3] if nl==2 and len(individual)>3 else 64

    tscv = TimeSeriesSplit(n_splits=3)
    fold_acc = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Тензоры
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_va_t = torch.tensor(X_val, dtype=torch.float32)
        y_va_t = torch.tensor(y_val, dtype=torch.long)

        loader_tr = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                               batch_size=64, shuffle=True)
        loader_va = DataLoader(TensorDataset(X_va_t, y_va_t),
                               batch_size=64, shuffle=False)

        # Модель
        input_size = X_tr.shape[2]
        model = build_lstm_model(nl, u1, u2, dr, input_size,
                                 use_genetics=True)
        model.train()

        crit = nn.CrossEntropyLoss()
        opt  = Adam(model.parameters(), lr=1e-3)
        sched= ReduceLROnPlateau(opt, mode='min', patience=1,
                                 factor=0.5, min_lr=1e-5)

        # Быстрая тренировка (несколько эпох)
        for epoch in range(10):
            tot_loss = 0.0
            for xb, yb in loader_tr:
                opt.zero_grad()
                out = model(xb)
                loss= crit(out, yb)
                loss.backward()
                opt.step()
                tot_loss += loss.item()*xb.size(0)
            # валидация
            model.eval()
            with torch.no_grad():
                val_loss, corr, tot = 0.0, 0, 0
                for xb, yb in loader_va:
                    out = model(xb)
                    l   = crit(out, yb)
                    val_loss += l.item()*xb.size(0)
                    preds = out.argmax(dim=1)
                    corr += (preds==yb).sum().item()
                    tot  += yb.size(0)
            sched.step(val_loss/len(loader_va.dataset))
            acc = corr/tot
            model.train()

        fold_acc.append(acc)

    return (np.mean(fold_acc),)

def optimize_hyperparameters(X, y,
                             pop_size=POPULATION_SIZE,
                             ngen=GENERATIONS):
    toolbox.register("evaluate", evaluate_model, X=X, y=y)
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutGaussian,
                     mu=0, sigma=10, indpb=0.2)
    toolbox.register("select",   tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    for gen in range(ngen):
        offspring = list(map(toolbox.clone,
                             toolbox.select(pop, len(pop))))
        # скрещивание
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        # мутация + границы
        for m in offspring:
            if random.random() < 0.2:
                toolbox.mutate(m)
                del m.fitness.values
            check_bounds(m)
        # оценка
        invalid = [ind for ind in offspring
                   if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offspring
        hof.update(pop)
        logging.info(f"[GA] Gen {gen}: best={hof[0].fitness.values}")

    logging.info(f"[GA] Best individual: {hof[0]}")
    return hof[0]

# === ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ===
def train_model_for_symbol(df: pd.DataFrame,
                           symbol: str,
                           exchange: str,
                           use_genetics: bool = False):
    """
    Тренирует LSTM для заданного symbol+exchange и сохраняет модель.
    """
    df = feature_engineering(df)
    if len(df) < window_size + 1:
        logging.error(f"[Train] Недостаточно данных для {symbol}@{exchange}: {df.shape}")
        return None

    # Генерация X, y
    X, y = [], []
    for i in range(window_size, len(df)):
        seq = df.iloc[i-window_size:i].values
        pct = (df.iloc[i]["close"] - df.iloc[i-1]["close"]) / df.iloc[i-1]["close"]
        label = 1 if pct >= 0 else 0
        X.append(seq)
        y.append(label)
    X = np.array(X); y = np.array(y)

    # Подбор гиперпараметров
    if use_genetics:
        best = optimize_hyperparameters(X, y)
        nl, u1, dr = int(round(best[0])), int(round(best[1])), best[2]
        u2 = int(round(best[3])) if nl==2 and len(best)>3 else 64
        logging.info(f"[Train] GA-params for {symbol}@{exchange}: "
                     f"layers={nl}, u1={u1}, u2={u2}, dr={dr:.2f}")
    else:
        nl, u1, u2, dr = 2, 128, 64, 0.3
        logging.info(f"[Train] Default params for {symbol}@{exchange}: "
                     f"layers={nl}, u1={u1}, u2={u2}, dr={dr:.2f}")

    # Построение модели
    input_size = X.shape[2]
    model = build_lstm_model(nl, u1, u2, dr, input_size,
                             use_genetics=use_genetics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Тензоры и DataLoader
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    split = int(0.9 * len(X_t))
    train_ds = TensorDataset(X_t[:split], y_t[:split])
    val_ds   = TensorDataset(X_t[split:], y_t[split:])
    tr_ld    = DataLoader(train_ds, batch_size=64, shuffle=True)
    va_ld    = DataLoader(val_ds,   batch_size=64, shuffle=False)

    crit = nn.CrossEntropyLoss()
    opt  = Adam(model.parameters(), lr=1e-3)
    sched= ReduceLROnPlateau(opt, mode='min',
                             patience=3, factor=0.5, min_lr=1e-6)

    best_loss = float('inf')
    patience, max_pat = 0, 7

    # Основной цикл обучения
    for epoch in range(150):
        model.train()
        train_loss = 0.0
        for xb, yb in tr_ld:
            opt.zero_grad()
            out = model(xb)
            loss= crit(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()*xb.size(0)
        train_loss /= len(tr_ld.dataset)

        model.eval()
        val_loss, corr, tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in va_ld:
                out = model(xb)
                l   = crit(out, yb)
                val_loss += l.item()*xb.size(0)
                preds = out.argmax(dim=1)
                corr += (preds==yb).sum().item()
                tot  += yb.size(0)
        val_loss /= len(va_ld.dataset)
        val_acc   = corr / tot

        sched.step(val_loss)
        logging.info(
            f"[Train] {symbol}@{exchange} Epoch {epoch}: "
            f"TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}"
        )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_state= model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= max_pat:
                logging.info("[Train] Early stopping")
                break

    # Восстанавливаем лучшую модель
    model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "num_layers": nl,
            "units1": u1,
            "units2": u2,
            "dropout_rate": dr
        }
    }
    model_file = f"lstm_{symbol}_{exchange}_model.pth"
    torch.save(ckpt, model_file)
    logging.info(f"[Train] Saved checkpoint for {symbol}@{exchange} → {model_file}")
    return model
