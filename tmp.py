import pandas as pd

root = "/data1/jty/data/math_lighteval_boxed"
train = pd.read_parquet(f"{root}/train.parquet")
test = pd.read_parquet(f"{root}/test.parquet")

for col in ["level", "type"]:
    print(f"\n========== {col} ==========")

    counts = pd.concat([
        train[col].value_counts().rename("Train N"),
        test[col].value_counts().rename("Test N"),
    ], axis=1).fillna(0).astype(int)

    counts["Train %"] = (
        100 * counts["Train N"] / len(train)
    ).round(2)

    counts["Test %"] = (
        100 * counts["Test N"] / len(test)
    ).round(2)

    counts["Diff (pp)"] = (
        counts["Test %"] - counts["Train %"]
    ).round(2)

    print(counts.sort_index().to_string())

print("\n========== Difficulty summary ==========")
for name, df in [("Train", train), ("Test", test)]:
    levels = df["level"].str.extract(r"(\d+)")[0].astype(int)
    print(
        name,
        "Mean level:", round(levels.mean(), 3),
        "Level 4–5 (%):",
        round(100 * (levels >= 4).mean(), 2),
    )

print("\n========== Test original indices ==========")
indices = test["extra_info"].map(lambda x: x["index"])
print("Min:", indices.min(), "Max:", indices.max())
print("Unique:", indices.nunique())
print("First 10:", indices.head(10).tolist())
print("Last 10:", indices.tail(10).tolist())