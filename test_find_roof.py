import numpy as np
from scipy import optimize
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from pr3d.de import ConditionalGaussianMM
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import time 
"""
def F(x, a, b):
    return np.power(x, a+1.0) - b

N = 1000000

a = np.random.rand(N)
b = np.random.rand(N)


# It shows that scipy implementation of newton is vectorized:
print(optimize.newton(F, np.zeros(N), args=(a, b)))

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html

"""

# load the conditional trained model
conditional_delay_model = ConditionalGaussianMM(
    h5_addr = "gmm_conditional_model.h5",
    dtype = 'float64',
)

N = 100000
x = { 'queue_length1': np.zeros(N), 'queue_length2': np.zeros(N), 'queue_length3' : np.zeros(N) }
x_list = np.array([np.array([*items]) for items in zip(*x.values())])

samples = np.random.uniform(low=0.0,high=1.0,size=N)
samples = np.linspace(0.00001,0.99999,N)
#samples = np.ones(N)*0.1

def model_cdf_fn(x ,a, b):
    #return conditional_delay_model.cdf(x=a, y=x)-b
    pdf, logpdf, cdf = conditional_delay_model.prob_batch(x=a,y=x)
    return cdf - b

def model_pdf_fn(x, a, b=None):
    pdf, logpdf, cdf = conditional_delay_model.prob_batch(x=a,y=x)
    return pdf

"""
y = np.linspace(0.0,100.0,N)
result = model_cdf_fn(x=y,a=x_list,b=samples)
plt.plot(y,result,'.')
plt.show()

result = model_pdf_fn(x=y,a=x_list)
plt.plot(y,result,'.')
plt.show()

start = time.time()
result = optimize.newton(
    func = model_cdf_fn, 
    x0 = conditional_delay_model.mean(x=x), # we feed the mean of the mixture as the initial guess
    args=(x_list,samples),
    fprime = model_pdf_fn,
    disp = True,
)
#plt.plot(result,'.')
#plt.show()
end = time.time()
print(end - start)
"""
def model_cdf_fn_t(x):
    a = x_list
    b = samples
    #return conditional_delay_model.cdf(x=a, y=x)-b
    pdf, logpdf, cdf = conditional_delay_model.prob_batch(x=a,y=x)
    return tf.convert_to_tensor(cdf - b, dtype='float64')

#print(model_cdf_fn_t(np.ones(N)*10.00))
#print(tf.convert_to_tensor(conditional_delay_model.mean(x=x)))

start = time.time()
result = tfp.math.find_root_secant(
    objective_fn = model_cdf_fn_t,
    initial_position = tf.convert_to_tensor(conditional_delay_model.mean(x=x)),
    value_tolerance = np.ones(N)*1e-7, #tf.convert_to_tensor(np.ones(N)*1e-7, dtype='float64'),
    position_tolerance = np.ones(N)*1e-3, #tf.convert_to_tensor(np.ones(N)*1e-2, dtype='float64'),
)
end = time.time()
print(end - start)
plt.plot(result[0],'.')
plt.savefig('foo.png')

"""
start = time.time()
result = tfp.math.find_root_chandrupatla(
    objective_fn = model_cdf_fn_t,
    low = tf.convert_to_tensor(np.ones(N)*0.0001, dtype='float64'),
    high = tf.convert_to_tensor(np.ones(N)*1000, dtype='float64'),
    value_tolerance = tf.convert_to_tensor(np.ones(N)*1e-7, dtype='float64'),
)
end = time.time()
print(end - start)
#print(result[0])
#plt.plot(result[0],'.')
#plt.show()
"""