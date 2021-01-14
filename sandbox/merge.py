from pandas import read_csv

df1 = read_csv("reduce_bench0.csv")
df2 = read_csv("reduce_bench.csv")

on = ['case', 'shape', 'op', 'axes', 'size']
merged = df1.merge(
    df2, left_on=on, right_on=on, suffixes=('0', 'new'))
merged['speedup'] = merged['ort0'] / merged['ortnew']

print(merged)
merged.to_excel("merged.xlsx", index=False)
