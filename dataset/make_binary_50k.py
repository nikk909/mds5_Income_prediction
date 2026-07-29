"""
将 BRFSS 糖尿病指标数据转为二分类，并按标签分层随机抽取 50,000 条。
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
SRC = HERE / "diabetes_012_health_indicators_BRFSS2015.csv"
OUT = HERE / "diabetes_binary_50k_stratified.csv"
N = 50_000
SEED = 42


def main() -> None:
    df = pd.read_csv(SRC)

    # 三分类 -> 二分类：0=无糖尿病，1=前期或糖尿病
    df = df.copy()
    df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)
    df = df.drop(columns=["Diabetes_012"])

    print("Full data:", df.shape)
    print("Full binary class ratio:")
    print(df["Diabetes_binary"].value_counts(normalize=True).sort_index().round(4))

    if len(df) < N:
        raise ValueError(f"Only {len(df)} rows available, cannot sample {N}.")

    sampled, _ = train_test_split(
        df,
        train_size=N,
        stratify=df["Diabetes_binary"],
        random_state=SEED,
    )
    sampled = sampled.reset_index(drop=True)

    print("\nSampled data:", sampled.shape)
    print("Sampled binary class counts:")
    print(sampled["Diabetes_binary"].value_counts().sort_index())
    print("Sampled binary class ratio:")
    print(sampled["Diabetes_binary"].value_counts(normalize=True).sort_index().round(4))

    sampled.to_csv(OUT, index=False)
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
