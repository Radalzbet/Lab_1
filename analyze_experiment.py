"""Аудит A/B-эксперимента. Запуск: python analyze_experiment.py"""

# === Этап 2 ===
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, ttest_ind, mannwhitneyu, fisher_exact, chi2_contingency
from statsmodels.stats.proportion import confint_proportions_2indep
from sklearn.linear_model import LogisticRegression

sns.set_theme(style="whitegrid", palette="colorblind")
ALPHA = 0.05
DATA_PATH = Path("experiment_data.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("ab_experiment/experiment_data.csv")
assert DATA_PATH.exists(), "Поместите experiment_data.csv рядом с ноутбуком"
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
display(df.head())

# === Этап 4 ===
required = {"user_id", "variant", "converted", "error_flag", "pre_activity", "date"}
assert required.issubset(df.columns), f"Нет колонок: {required - set(df.columns)}"
print("Размер таблицы:", df.shape)
print("Пропуски по столбцам:\n", df.isna().sum().to_string())
print("Дубли user_id:", df.user_id.duplicated().sum())
print("Даты:", df.date.min().date(), "—", df.date.max().date())
assert not df.isna().any().any()
assert df.user_id.is_unique
assert set(df.variant.unique()) == {"control", "test"}
assert df.converted.isin([0, 1]).all() and df.error_flag.isin([0, 1]).all()
assert df.pre_activity.ge(0).all()

summary = df.groupby("variant", sort=True).agg(
    users=("user_id", "size"),
    conversions=("converted", "sum"),
    conversion_rate=("converted", "mean"),
    errors=("error_flag", "sum"),
    error_rate=("error_flag", "mean"),
    mean_pre_activity=("pre_activity", "mean"),
    median_pre_activity=("pre_activity", "median"),
)
summary["share"] = summary.users / len(df)
display(summary.style.format({"conversion_rate":"{:.3%}", "error_rate":"{:.3%}", "share":"{:.2%}", "mean_pre_activity":"{:.2f}"}))
c = df.loc[df.variant == "control"]
t = df.loc[df.variant == "test"]

# === Этап 6 ===
observed = np.array([len(c), len(t)])
expected = np.repeat(len(df) / 2, 2)
srm_stat, srm_p = chisquare(f_obs=observed, f_exp=expected)
srm_warning = srm_p < ALPHA
print(f"Факт control/test: {len(c)}/{len(t)} (доли {len(c)/len(df):.2%}/{len(t)/len(df):.2%})")
print(f"χ² = {srm_stat:.3f}, p-value = {srm_p:.3e}")
if srm_warning:
    print("⚠ SRM обнаружен")
else:
    print("SRM статистически не обнаружен")

# === Этап 8 ===
welch = ttest_ind(t.pre_activity, c.pre_activity, equal_var=False)
mann_whitney = mannwhitneyu(t.pre_activity, c.pre_activity, alternative="two-sided")
pooled_sd = np.sqrt((t.pre_activity.var(ddof=1) + c.pre_activity.var(ddof=1)) / 2)
smd_before = (t.pre_activity.mean() - c.pre_activity.mean()) / pooled_sd
cov_warning = (welch.pvalue < ALPHA) or (mann_whitney.pvalue < ALPHA)
print(f"Средняя активность: control={c.pre_activity.mean():.3f}, test={t.pre_activity.mean():.3f}")
print(f"t-test Уэлча p={welch.pvalue:.3e}; U-тест p={mann_whitney.pvalue:.3e}")
print(f"SMD = {smd_before:.3f}")
if cov_warning:
    print("⚠ Дисбаланс по pre_activity — возможен confounder")
else:
    print("Статистически значимого дисбаланса не выявлено")

# === Этап 10 ===
ct = np.array([[int(t.error_flag.sum()), len(t)-int(t.error_flag.sum())],
               [int(c.error_flag.sum()), len(c)-int(c.error_flag.sum())]])
odds_ratio, guardrail_p = fisher_exact(ct, alternative="two-sided")
rate_t, rate_c = t.error_flag.mean(), c.error_flag.mean()
err_diff = rate_t - rate_c
err_rr = rate_t / rate_c if rate_c > 0 else np.inf
ci_err = confint_proportions_2indep(int(t.error_flag.sum()), len(t),
                                    int(c.error_flag.sum()), len(c),
                                    compare="diff", method="newcomb")
guardrail_warning = (guardrail_p < ALPHA) and (rate_t > rate_c)
print(f"Ошибки: control={c.error_flag.sum()}/{len(c)} ({rate_c:.3%}); test={t.error_flag.sum()}/{len(t)} ({rate_t:.3%})")
print(f"Разница = {err_diff*100:+.3f} п.п.; RR = {err_rr:.3f}")
print(f"95% ДИ разницы = [{ci_err[0]*100:.3f}, {ci_err[1]*100:.3f}] п.п.; Fisher p={guardrail_p:.5f}")
if guardrail_warning:
    print("⚠ Guardrail: частота ошибок значимо выросла в тесте")
elif guardrail_p < ALPHA:
    print("Есть значимое изменение guardrail, без ухудшения в тесте")
else:
    print("Значимого различия guardrail не обнаружено")

# === Этап 12 ===
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
labels = ["control", "test"]
axs[0,0].bar(labels, observed, label="Факт", alpha=.85)
axs[0,0].plot(labels, expected, "k--o", label="Ожидание 50/50")
axs[0,0].set(title="Размеры групп и SRM", ylabel="Пользователи")
axs[0,0].legend()
sns.boxplot(data=df, x="variant", y="pre_activity", order=labels, ax=axs[0,1], showfliers=False)
axs[0,1].set(title="pre_activity по группам (без выбросов на графике)", xlabel="Группа")
axs[1,0].bar(labels, [rate_c * 100, rate_t * 100])
axs[1,0].set(title="Guardrail: ошибки", ylabel="Доля ошибок, %")
daily = df.groupby(["date", "variant"], as_index=False)["error_flag"].mean()
sns.lineplot(data=daily, x="date", y="error_flag", hue="variant", marker="o", ax=axs[1,1])
axs[1,1].set(title="Частота ошибок по датам", ylabel="Доля ошибок", xlabel="Дата")
axs[1,1].tick_params(axis="x", rotation=40)
plt.tight_layout()
plt.show()

# === Этап 14 ===
invalid = srm_warning or cov_warning or guardrail_warning
print("Статусы:", {"SRM": bool(srm_warning), "pre_activity": bool(cov_warning),
                      "guardrail deterioration": bool(guardrail_warning)})
if invalid:
    print("❌ ЭКСПЕРИМЕНТ НЕ ВАЛИДЕН. Остановить тест, исправить ошибки, перезапустить.")
else:
    print("✅ Критичных предупреждений нет: можно анализировать основную метрику.")

conversion_table = pd.crosstab(df.variant, df.converted).reindex(index=labels, columns=[0,1], fill_value=0)
chi_conv, p_conv, _, _ = chi2_contingency(conversion_table, correction=False)
print(f"Конверсия control={c.converted.mean():.3%}; test={t.converted.mean():.3%}")
print(f"Разница={(t.converted.mean()-c.converted.mean())*100:+.3f} п.п.; χ² p={p_conv:.5f}")
if invalid:
    print("ВНИМАНИЕ: p-value конверсии только диагностический, вывод об эффекте делать нельзя.")
else:
    print("Значимость конверсии:", "p < 0.05" if p_conv < ALPHA else "p ≥ 0.05")

# === Этап 16 ===
assigned = (df.variant == "test").astype(int).to_numpy()
X = df[["pre_activity"]].to_numpy()
model = LogisticRegression(max_iter=1000)
model.fit(X, assigned)
ps = np.clip(model.predict_proba(X)[:, 1], 0.05, 0.95)
p_t = assigned.mean()
weights = np.where(assigned == 1, p_t / ps, (1-p_t) / (1-ps))

def weighted_mean(values, w):
    return np.average(np.asarray(values), weights=w)

wt = weights[assigned == 1]
wc = weights[assigned == 0]
mean_t = weighted_mean(t.pre_activity, wt)
mean_c = weighted_mean(c.pre_activity, wc)
smd_after = (mean_t - mean_c) / pooled_sd
conv_t_adj = weighted_mean(t.converted, wt)
conv_c_adj = weighted_mean(c.converted, wc)
ess_t = wt.sum()**2 / (wt**2).sum()
ess_c = wc.sum()**2 / (wc**2).sum()

results = pd.DataFrame({
    "До IPW": [t.pre_activity.mean(), c.pre_activity.mean(), smd_before,
               t.converted.mean(), c.converted.mean(), t.converted.mean()-c.converted.mean()],
    "После IPW": [mean_t, mean_c, smd_after, conv_t_adj, conv_c_adj, conv_t_adj-conv_c_adj],
}, index=["Средняя активность test", "Средняя активность control", "SMD (активность)",
          "Конверсия test", "Конверсия control", "Разница конверсий"])
display(results.round(5))
print(f"Effective sample size: control={ess_c:.0f} из {len(c)}, test={ess_t:.0f} из {len(t)}")
print("IPW исправляет наблюдаемый дисбаланс, но не SRM и не сам факт ошибочного назначения.")
print("Окончательное решение: эксперимент НЕ ВАЛИДЕН, необходим перезапуск.")
