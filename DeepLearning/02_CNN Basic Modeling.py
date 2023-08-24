# %% [markdown]
# # 02 CNN Basic Modeling

# %%
import tensorflow

# %%
tensorflow.__version__

# %%
from tensorflow.keras import datasets, layers, models #Tensorflow에 있는 Keras 함수들 호출하기

# %% [markdown]
# ## 1. 데이터 로드 및 탐색

# %%
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# %%
from matplotlib import pyplot as plt

plt.figure()
plt.imshow(train_images[10004])
plt.show()

# %%
train_images.shape, test_images.shape

# %%
train_images[0][0]

# %%
# conv layer에 넣어주기 위해서 28x28을 28x28x1 형태로 변경해야한다.
train_images = train_images.reshape((60000, 28,28,1))
test_images = test_images.reshape((10000, 28,28,1))

# %%
train_images[0]

# %%
train_images, test_images = train_images / 255.0, test_images / 255.0

# %% [markdown]
# ### 2. 모델 구축
# - Filtering 층과 Classification 층으로 구분
# - Filtering 층 : Dense 대신 Conv2D, MaxPooling2D 사용
#     - Conv2D : 
#         - filter개수
#         - filter_size
#         - stride : default = 1
#     - MaxPooling2D : 
#         - filter_size
#         - stride : default = filter_size
# 
# - model.summary()
#     - 모델 구조, weight 개수 세기 정답지
#     - Output Shape : 피처맵 
#     - param # : weight 개수

# %%
#모델 구축
model = models.Sequential()
## filtering layer
# conv 2d에 들어오는 데이터는 채널도 들어오기때문에 3차원으로 입력해줘야 한다.
# stride default = 1
# padding default 안준다. 주고싶으면 padding = 'same'
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# filter size 2*2, stride는 default가 filter size(2)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))

## classification layer
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# %%
model.summary()

# %%
#모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
#모델 학습
model.fit(train_images, train_labels, epochs=3,validation_split=0.2,verbose=1)

# %%
#모델 검증
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# %% [markdown]
# ## cifar10 dataset 실습

# %%
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images.reshape((50000, 32,32,3))
test_images = test_images.reshape((10000, 32,32,3))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

plt.figure(figsize=(20,20))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
plt.show()

# %%
model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu',input_shape=(32, 32, 3)))
model.add(layers.Conv2D(8, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10,verbose=1)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# %%
model.summary()


