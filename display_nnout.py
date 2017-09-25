 display_nnout.py
# Makes plots of an individual potential (scaled to unit max), the gradient descent (“correct”) ground state,
# and the neural network predicted ground state
# should be added to notebook containing schroedinger_nn.py
import matplotlib.pyplot as mp
potenid = 5397
mp.plot(sess.run(L3,feed_dict={X: [trainx[potenid]]})[0])
mp.plot([trainx[potenid][i]/max(trainx[potenid]) for i in range(bins - 1)])
mp.plot(trainy[potenid])
mp.show()
