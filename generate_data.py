"""Однократная генерация уникального набора данных для учебного A/B-теста."""
import secrets
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parent / 'experiment_data.csv'
MY_SEED = secrets.randbelow(5_000_000)
np.random.seed(MY_SEED)
# Большой размер выборки улучшает мощность проверок редкой guardrail-метрики.
n_users = int(np.random.randint(9000, 10000, size=1)[0])
pre_activity = np.random.exponential(scale=10, size=n_users).astype(int)
# Исходный черновой вариант 40/60 (как в условии); ниже он перезаписывается.
variant = np.random.choice(['control', 'test'], size=n_users, p=[0.4, 0.6])
# Изменены разрешенные параметры рандомизации, чтобы дисбаланс
# pre_activity был хорошо различим при анализе (в оригинале 0.4 + 0.3*x).
prob_test = np.clip(0.36 + 0.55 * (pre_activity / pre_activity.max()), 0, 1)
variant2 = np.where(np.random.random(n_users) < prob_test, 'test', 'control')
variant = variant2
error_flag = np.zeros(n_users, dtype=int)
error_flag[variant == 'control'] = np.random.binomial(1, 0.01, sum(variant == 'control'))
error_flag[variant == 'test'] = np.random.binomial(1, 0.02, sum(variant == 'test'))
conversion_prob = 0.05 + 0.01 * (pre_activity / pre_activity.max())
converted = np.random.binomial(1, conversion_prob, n_users)
start_date = pd.Timestamp('2026-05-01')
dates = [start_date + pd.Timedelta(days=np.random.randint(0, 14)) for _ in range(n_users)]
df = pd.DataFrame({'user_id': range(n_users), 'variant': variant,
                   'converted': converted, 'error_flag': error_flag,
                   'pre_activity': pre_activity, 'date': dates})
df.to_csv(OUTPUT, index=False)
print(f'SEED={MY_SEED}; n_users={n_users}; data={OUTPUT}')
