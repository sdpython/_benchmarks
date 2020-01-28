import matplotlib.pyplot as plt
import pandas
name = "../../scikit-learn/results/bench_polynomial_features.csv"
df = pandas.read_csv(name)
plt.close('all')

index = ['N', 'count', 'degree', 'dim', 'error_c', 'interaction_only']
values = ['mean']
columns = ['test']
piv = pandas.pivot_table(data=df, index=index, values=values, columns=columns)
print(piv)


nrows = len(set(df.degree))
fig, ax = plt.subplots(nrows, 4, figsize=(nrows * 4, 12))
pos = 0

for di, degree in enumerate(sorted(set(df.degree))):
    pos = 0
    for order in sorted(set(df.order)):
        for interaction_only in sorted(set(df.interaction_only)):
            a = ax[di, pos]
            if di == ax.shape[0] - 1:
                a.set_xlabel("N observations", fontsize=24)
            if pos == 0:
                a.set_ylabel("Time (s) degree={}".format(degree),
                             fontsize=24)

            for color, dim in zip('brgyc', sorted(set(df.dim))):
                subset = df[(df.degree == degree) & (df.dim == dim) &
                            (df.interaction_only == interaction_only) &
                            (df.order == order)]
                if subset.shape[0] == 0:
                    continue
                subset = subset.sort_values("N")
                label = "nf={} l=0.20.2".format(dim)
                subset.plot(x="N", y="time_0_20_2", label=label, ax=a,
                            logx=True, logy=True, c=color, style='--')
                label = "nf={} l=now".format(dim)
                subset.plot(x="N", y="time_current", label=label, ax=a,
                            logx=True, logy=True, c=color)

            a.legend(loc=0, fontsize=24)
            if di == 0:
                a.set_title("order={} interaction_only={}".format(
                    order, interaction_only), fontsize=24)
            pos += 1

plt.suptitle("Benchmark for PolynomialFeatures")
plt.show()
