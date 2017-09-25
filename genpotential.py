# genpotential.py
# This snippet of python code generates random potentials of 3 different types,
# Step functions, piecewise linear functions, and random Fourier series.
# Each of these types gets more “jagged” as generation progresses.
# The ground state wavefunction of each potential is found using Tensorflow’s gradient
# descent method on the energy functional given by the Schroedinger equation.
# The potentials and solutions are partitioned into training and validation data
# and saved with the random seed appended to the filename

import csv
import numpy as np
import tensorflow as tf

def subexp(expon):
    return np.power(abs(np.log(np.random.uniform())),expon)

def generatepot(style,param): #0=step,1=linear,2=fourier; 0-1 "jaggedness" scale
    mu = 1. + bins*param #mean number of jump points for styles 0 + 1
    forxp = 2.5 - 2*param #fourier exponent for style 2
    scale = 5.0*(np.pi*np.pi*0.5) # energy scale
    if style < 2:
        dx = bins/mu
        xlist = [-dx/2]
        while xlist[-1] < bins:
            xlist.append(xlist[-1]+dx*subexp(1.))
        vlist = [scale*subexp(2.) for k in range(len(xlist))]
        k = 0
        poten = []
        for l in range(1,bins):
            while xlist[k+1] < l:
                k = k + 1
            if style == 0:
                poten.append(vlist[k])
            else:
                poten.append(vlist[k]+(vlist[k+1]-vlist[k])*(l-xlist[k])/(xlist[k+1]-xlist[k]))
    else:
        sincoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k,forxp) for k in range(1,bins//2)]
        coscoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k,forxp) for k in range(1,bins//2)]
        zercoef = scale*subexp(2.)
        poten = np.maximum(np.add(np.add(np.matmul(sincoef,sinval),np.matmul(coscoef,cosval)),zercoef),0).tolist()
    return poten

seed = 0
np.random.seed(seed)
bins = 128 #dx = 1/bins; actual number of columns saved = bins-1, because 1st and last are 0
npots = 200 #ends up being 3*this*(validnth-1)/validnth
validnth = 5 #every nth sample func is saved as validation
sinval = np.sin([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)])
cosval = np.cos([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)])
sqrt2 = np.sqrt(2)

defgrdstate = tf.constant([sqrt2*np.sin(i*np.pi/bins) for i in range(1,bins)])
psi = tf.Variable(defgrdstate)
zerotens = tf.zeros([1])
psil = tf.concat([psi[1:],zerotens],0)
psir = tf.concat([zerotens,psi[:-1]],0)
renorm = tf.assign(psi,tf.divide(psi,tf.sqrt(tf.reduce_mean(tf.square(psi)))))
optimzi = tf.train.GradientDescentOptimizer(0.0625/bins)
reinit = tf.assign(psi,defgrdstate)
init = tf.global_variables_initializer()

potentials = []
validpots = []
wavefuncs = []
validfuncs = []

sess = tf.Session()
sess.run(init)
for i in range(npots):
    if i%10 == 0:
        print str((100.*i)/npots) + '% complete'
    for j in range(3):
        vofx = generatepot(j,(1.*i)/npots)
        energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                            tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
        training = optimzi.minimize(energy)
        sess.run(reinit)
        for t in range(20000):
            sess.run(training)
            sess.run(renorm)
        if i%validnth == 0:
            validpots.append(vofx)
            validfuncs.append(sess.run(psi).tolist())
        else:
            potentials.append(vofx)
            wavefuncs.append(sess.run(psi).tolist())

with open('test_pots'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(potentials)
with open('valid_pots'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(validpots)
with open('test_out'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(wavefuncs)
with open('valid_out'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(validfuncs)
print 'Output complete'
