# train.py
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score
from deap import base, creator, tools
from typing import Tuple, Optional

from app.config import POPULATION_SIZE, GENERATIONS, window_size
from model import build_lstm_model

logging.basicConfig(level=logging.INFO)


# === FEATURE ENGINEERING ===
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"[Features] Input shape: {df.shape}")
    from ta import add_all_ta_features

    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    df["log_return"]   = np.log(df.close / df.close.shift(1)).fillna(0)
    df["volatility"]   = df.log_return.rolling(10).std().fillna(0)
    df["momentum"]     = df.close - df.close.rolling(5).mean()
    df["price_range"]  = df.high - df.low
    df["volume_delta"] = df.volume.diff().fillna(0)
    df.dropna(inplace=True)

    logging.info(f"[Features] Output shape: {df.shape}")
    return df


# === DEAP GA SETUP ===
try:
    creator.FitnessMax
except Exception:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
try:
    creator.Individual
except Exception:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# 1) FUNCTIONS FOR INDIVIDUAL CREATION AND BOUND CHECKING
def create_individual():
    nl = toolbox.num_layers()
    if nl == 1:
        return creator.Individual([nl,
                                   toolbox.units1(),
                                   toolbox.dropout_rate()])
    else:
        return creator.Individual([nl,
                                   toolbox.units1(),
                                   toolbox.dropout_rate(),
                                   toolbox.units2()])

def check_bounds(ind):
    ind[0] = int(max(1, min(2, round(ind[0]))))
    ind[1] = int(max(32, min(256, round(ind[1]))))
    ind[2] = float(max(0.1, min(0.5, ind[2])))
    if ind[0] == 2:
        if len(ind) < 4:
            ind.append(random.randint(32, 256))
        else:
            ind[3] = int(max(32, min(256, round(ind[3]))))
    return ind

# 2) INITIALIZE TOOLBOX AND REGISTER OPERATORS
toolbox = base.Toolbox()

toolbox.register("num_layers",   random.choice, [1, 2])
toolbox.register("units1",       random.randint, 32, 256)
toolbox.register("units2",       random.randint, 32, 256)
toolbox.register("dropout_rate", random.uniform, 0.1, 0.5)
toolbox.register("individual",   create_individual)
toolbox.register("population",   tools.initRepeat, list, toolbox.individual)


# === MODEL EVALUATION FOR GA ===
def evaluate_model(individual, X, y) -> Tuple[float]:
    nl, u1, dr = individual[0], individual[1], individual[2]
    u2 = individual[3] if nl == 2 and len(individual) > 3 else 64

    tscv = TimeSeriesSplit(n_splits=3)
    fold_accs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        tr_ld = DataLoader(
            TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long)
            ),
            batch_size=64, shuffle=True
        )
        va_ld = DataLoader(
            TensorDataset(
                torch.tensor(X_va, dtype=torch.float32),
                torch.tensor(y_va, dtype=torch.long)
            ),
            batch_size=64, shuffle=False
        )

        model = build_lstm_model(
            num_layers=nl,
            units1=u1,
            units2=u2,
            dropout_rate=dr,
            input_size=X.shape[2],
            use_genetics=True
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                      patience=1, factor=0.5, min_lr=1e-5)

        for _ in range(5):
            model.train()
            for xb, yb in tr_ld:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb = xb.to(device)
                    preds.extend(model(xb).argmax(dim=1).cpu().numpy())
                    trues.extend(yb.numpy())
            acc = accuracy_score(trues, preds)
            scheduler.step(1.0 - acc)

        fold_accs.append(acc)

    return (float(np.mean(fold_accs)),)


def optimize_hyperparameters(X, y,
                             pop_size: int = POPULATION_SIZE,
                             ngen: int = GENERATIONS):
    toolbox.register("evaluate", evaluate_model, X=X, y=y)
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
    toolbox.register("select",   tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    for gen in range(ngen):
        offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, len(pop))]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for ind in offspring:
            if random.random() < 0.2:
                toolbox.mutate(ind)
                del ind.fitness.values
            check_bounds(ind)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        hof.update(pop)
        logging.info(f"[GA] Gen {gen}: best acc={hof[0].fitness.values[0]:.4f}")

    logging.info(f"[GA] Best individual: {hof[0]}")
    return hof[0]


# === TRAINING FUNCTION ===
def train_model_for_symbol(
    df: pd.DataFrame,
    symbol: str,
    exchange: str,
    use_genetics: bool = False
) -> Optional[Tuple[nn.Module, dict]]:
    df = feature_engineering(df)
    if len(df) < window_size + 1:
        logging.error(f"[Train] Not enough data: {df.shape}")
        return None

    X, y = [], []
    for i in range(window_size, len(df)):
        seq = df.iloc[i - window_size : i].values
        label = 1 if df.close.iloc[i] >= df.close.iloc[i - 1] else 0
        X.append(seq); y.append(label)
    X, y = np.array(X), np.array(y)

    if use_genetics:
        best = optimize_hyperparameters(X, y)
        nl, u1, dr = int(round(best[0])), int(round(best[1])), best[2]
        u2 = int(round(best[3])) if nl == 2 and len(best) > 3 else 64
        logging.info(f"[Train] GA params: layers={nl}, u1={u1}, u2={u2}, dr={dr:.2f}")
    else:
        nl, u1, u2, dr = 2, 128, 64, 0.3
        logging.info(f"[Train] Default params: layers={nl}, u1={u1}, u2={u2}, dr={dr:.2f}")

    cfg = {
        "num_layers":   nl,
        "units1":       u1,
        "units2":       u2,
        "dropout_rate": dr,
        "input_size":   X.shape[2],
        "use_genetics": use_genetics
    }

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_lstm_model(**cfg).to(device)

    tr_ld = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=64, shuffle=True
    )
    va_ld = DataLoader(
        TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                      torch.tensor(y_va, dtype=torch.long)),
        batch_size=64, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-6)

    best_acc, patience_cnt, best_state = 0.0, 0, None
    for epoch in range(1, 151):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(device)
                preds.extend(model(xb).argmax(dim=1).cpu().numpy())
                trues.extend(yb.numpy())
        acc = accuracy_score(trues, preds)
        scheduler.step(1.0 - acc)

        logging.info(f"[Train] Epoch {epoch}: ValAcc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_state, patience_cnt = acc, model.state_dict(), 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                logging.info("[Train] Early stopping")
                break

    if best_state:
        model.load_state_dict(best_state)
    logging.info(f"[Train] Best acc: {best_acc:.4f}")
    if best_acc < 0.75:
        logging.warning("[Train] Accuracy below 75%, consider more data or tuning")

    return model, cfg
