import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/IDS.csv")
df.dropna(inplace=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_client1_df, train_client2_df = train_test_split(train_df, test_size=0.5, random_state=42)

train_client1_df.to_csv("data/train_client_1.csv", index=False)
train_client2_df.to_csv("data/train_client_2.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
