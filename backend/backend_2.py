import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense

# backend check
K.backend()
print(K.backend())

#1 엡실론 설정
K.epsilon()
K.set_epsilon(1e-03)
K.epsilon()
print(K.epsilon())

#2 기본 float 타입 설정
K.floatx()
K.set_floatx('float32')
K.floatx()


#3 채널 순서 정하기
K.image_data_format()
K.set_image_data_format('channels_last')
# K.set_image_data_format('channels_first')
K.image_data_format()
print(K.image_data_format())

#4 Clear session
# 현재의 TF graph를 버리고 새로 만든다. 예전 모델, 레이어와의 충돌을 피한다.
# K.clear_session()
# UserWarning: `tf.keras.backend.set_learning_phase` is deprecated 
# and will be removed after 2020-10-11. 
# To update it, simply pass a True/False value to the `training` 
# argument of the `__call__` method of your layer or model.

#5 learning_phase
# train time과 test time에 있어 다른 behavior를 적용하는 keras function에 대해
# 0 = test, 1 = train을 가리키는 bool tensor를 인풋으로 제공한다.
# K.learning_phase()

# # Sets the learning phase to a fixed value.
# K.set_learning_phase(1)


#6 is_tensor, is_placeholder
# 타겟이 케라스 layer 혹은 Input에서 반환된 텐서가 맞는지 True, False 반환
keras_placeholder = K.placeholder(shape=(2, 4, 5))
K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.

keras_input = Input(shape=[10])
K.is_keras_tensor(keras_input) # An Input is a Keras tensor.

keras_layer_output = Dense(units=10)(keras_input)
K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.


#7 ndim, dtype: Returns the number of axes, dtype in a tensor
x = K.random_normal_variable(shape=(16, 16, 3), name='x', dtype='float32', mean=0, scale=1)
K.ndim(x)
K.dtype(x)
print(K.dtype(x))

#8 파라미터 수 세기
var9 = K.random_normal_variable(shape=(4,4,4), dtype='float32', mean=0, scale=1)
K.count_params(var9)


#9 다른 dtype으로 바꾸기
var10 = K.cast(var9, dtype='float64')
print(var10)


#10 update, update_add, update_sub, moving_average_update
old = K.random_normal_variable(shape=(2, 2), dtype='float32', mean=0, scale=1)
new = K.random_uniform_variable(shape=(2, 2), dtype='float32', low=0, high=1)
K.update(old, new)


#11 dot, transpose
K.dot(old, new)
K.transpose(old)


#12 gather: Retrieves the elements of indices in the reference tensor
var11 = K.variable(value=np.array([[1,2], [3,4]]))
what = K.gather(reference=var11, indices=0)
K.eval(what)


#13 max, min, sum, prod, mean, var, std -- 모두 같은 argument
result = K.max(var11, axis=0, keepdims=False)
K.eval(result)


#14 cumsum, cumprod, argmax, argmin -- 같은 argument
result = K.cumprod(var11, axis=0)
K.eval(result)

K.eval(K.argmax(result, axis=0))


#15 수학 계산: square, sqrt, log, exp round, sign equal, not_equal,
# greater, greater_equal, less, less_equal, maximum, minimum


# #16 Batch Normalization
# # output = (x - mean) / sqrt(var + epsilon) * gamma + beta
# x_normed = K.batch_normalization(x, mean=0, var=1, beta, gamma, axis=-1, epsilon=0.001)

# # Computes mean and std for batch then apply batch_normalization on batch.
# (y_normed, mean, variance) = K.normalize_batch_in_training(x, gamma, beta, reduction_axes=-1, epsilon=0.001)


# #17 concatenate, reshape, permute_dimensions
# K.concatenate([var3, var4], axis=-1)

var12 = K.variable(np.array(np.arange(0,24,1).reshape(2,3,4)))
print(K.eval(var12).shape)
K.permute_dimensions(var12, pattern=[2,1,0])


#18 resize image: 중간 argument는 양의 정수
img_tensor = K.random_normal_variable(shape=(1, 100, 60, 3), mean=0, scale=1)
K.resize_images(img_tensor, 10, 10, data_format='channels_last')


#19 flatten, expand_dims, squeeze
var13 = K.variable(value=np.ones((224, 224, 3)))
var14 = K.expand_dims(var13, axis=0)
var14
K.flatten(var13)

var14 = K.variable(value=np.ones((224, 224, 1)))
K.squeeze(var14, axis=-1)


#20 spatial_2d_padding
K.spatial_2d_padding(var14, padding=((3, 3), (3, 3)))


#21 기타
# stack: Stacks a list of rank R tensors into a rank R+1 tensor
K.stack(x, axis=0)

# one_hot: Computes the one-hot representation of an integer tensor
K.one_hot(indices, num_classes)

# slice
K.slice(x, start, size)

# get_value, batch_get_value(returns: a list of np arr)
K.get_value(var12)


#22 gradients
# loss=scalar tensor to minimize
# variables=list of varialbes or placholder
# 아래는 style transfer 예시
# 여기서 loss는 스칼라 값이었고, combination_image는 placeholder 였음
K.gradients(loss=loss, variables=combination_image)


#23 함수
K.relu(x='tensor or variable', alpha=0.0, max_value=None)
K.softmax(x, axis=-1)
K.categorical_crossentropy(target='output과 같은 shape의 텐서',
                           output='결과', from_logits=False, axis=-1)

K.dropout(x='tensor', level='fraction of the entries in the tensor that will be set to 0',
          noise_shape=None, seed=None)
K.l2_normalize(x='tensor', axis=None)
K.in_top_k(predictions='a tensor of (batch_size, classes)', targets='1D tensor of length batch size',
           k='# of top elements to consider')


#24 CNN
K.conv2d(x='tensor', kernel='kernel_tensor', strides=(1, 1), padding='valid', data_format='channels_last')
K.pool2d(x, pool_size=(2, 2), strides=(1, 1), padding='valid', data_format='channels_last', pool_mode='max')


#25 returns a tensor with ~ distributions
# random_normal, random_uniform, truncated_normal
var15 = K.random_normal(shape=(3, 3), mean=0.0, stddev=1.0, dtype='float32', seed=None)
K.get_value(var15)