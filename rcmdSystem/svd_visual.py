# %% [markdown]
# ## SVD 추천시스템
# 1. 서프라이즈를 사용한 ML-100K 추천 시스템
# 1. SVD를 사용하기
# 1. K값을 바꿔가면서 비교, 최적의 K를 찾기
# 1. K를 x축, 성능(RMSE)를 y축으로 해서 시각화

# %%
import pandas as pd
import surprise

# %%
rating_df = pd.read_csv('u.data', sep='\t', header=None, names = ['userID', 'movieID', 'rating', 'timestamp'])

# %%
pv_tb = pd.pivot_table(data = rating_df, values = 'rating', index = 'movieID', columns = 'userID').fillna(0)
pv_tb

# %%
from surprise import Dataset, SVD, Reader, accuracy
# from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

reader = Reader(line_format="user item rating", sep="\t") 
data = Dataset.load_from_df(rating_df[['userID', 'movieID', 'rating']], reader=reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
li = []
for fac in range(2, 100):
    svd = SVD(n_factors = fac)
    svd.fit(trainset)
    li.append(accuracy.rmse(svd.test(testset)))

# %%
import matplotlib.pyplot as plt
plt.plot(li)

# %%
min(li)

# %%
rating_df.shape

# %%
test = pd.read_csv('C:\Code\skkukdt_minzy\dataset\dest.csv', sep = '|', header = None)
test.columns = ['r', 'rmse']

# %%
test

# %%

plt.figure(figsize=(20, 5))
plt.plot(test['rmse'])
plt.show()

# %%
test['rmse'].min()

# %%
test['rmse'].argmin()

# %%



