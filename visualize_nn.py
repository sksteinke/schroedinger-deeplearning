# visualize_nn.py
# outputs bitmaps of the weights and biases from schroedinger_nn.py.
# Sorts them using a Gaussian kernel to increase spatial correlation between weights and nodes.
# Doubles the size of the bitmap before outputting.
# Append to schroedinger_nn.py notebook

def doubler(aray):
    dbled = np.zeros([2*i for i in aray.shape])
    if len(aray.shape) == 1:
        for i in range(aray.shape[0]):
            dbled[2*i] = aray[i]
            dbled[2*i+1] = aray[i]
    elif len(aray.shape) == 2:
        for i in range(aray.shape[0]):
            for j in range(aray.shape[1]):
                dbled[2*i][2*j] = aray[i][j]
                dbled[2*i+1][2*j] = aray[i][j]
                dbled[2*i][2*j+1] = aray[i][j]
                dbled[2*i+1][2*j+1] = aray[i][j]
    return dbled

from PIL import Image
we1 = sess.run(W1)
bi1 = sess.run(B1)
we2 = sess.run(W2)
bi2 = sess.run(B2)
we3 = sess.run(W3)
bi3 = sess.run(B3)

gauswid = 1.
weiscale = []
for i in range(bins-1):
    line = np.exp([-np.square(float(i-j)/gauswid)/2. for j in range(bins-1)])
    line = np.divide(line,sum(line))
    weiscale.append(line.tolist())

weconv1 = np.matmul(we1,weiscale)
weconv2 = np.matmul(we2,weiscale)

sign = 1
mask = np.zeros(bins-1)
for i in range(bins-1):
    ind = (bins-2)/2+int(np.floor((i+1)/2))*sign
    sign = -sign
    mxin = np.argmax(np.add(weconv1[ind],mask))
    swapper = np.identity(bins-1)
    swapper[ind][ind] = 0
    swapper[mxin][mxin] = 0
    swapper[ind][mxin] = 1
    swapper[mxin][ind] = 1
    we1 = np.matmul(we1,swapper)
    weconv1 = np.matmul(weconv1,swapper)
    bi1 = np.matmul(bi1,swapper)
    we2 = np.matmul(swapper,we2)
    mask[ind] = -1.E12

sign = 1
mask = np.zeros(bins-1)
for i in range(bins-1):
    ind = (bins-2)/2+int(np.floor((i+1)/2))*sign
    sign = -sign
    mxin = np.argmax(np.add(weconv2[ind],mask))
    swapper = np.identity(bins-1)
    swapper[ind][ind] = 0
    swapper[mxin][mxin] = 0
    swapper[ind][mxin] = 1
    swapper[mxin][ind] = 1
    we2 = np.matmul(we2,swapper)
    weconv2 = np.matmul(weconv2,swapper)
    bi2 = np.matmul(bi2,swapper)
    we3 = np.matmul(swapper,we3)
    mask[ind] = -1.E12


max1 = max(max(we1.tolist()))
min1 = min(min(we1.tolist()))
wedb1 = doubler(we1)
weight1 = np.divide(np.subtract(wedb1,min1),max1-min1)
wim1 = Image.fromarray((weight1*255).astype(np.uint8),'L')
wim1.save('W1.bmp')
max1 = max(bi1.tolist())
min1 = min(bi1.tolist())
bidb1 = doubler(bi1)
bia1 = np.divide(np.subtract(bidb1,min1),max1-min1)
bias1 = np.array([bia1.tolist() for i in range(32)])
bim1 = Image.fromarray((bias1*255).astype(np.uint8),'L')
bim1.save('B1.bmp')

max2 = max(max(we2.tolist()))
min2 = min(min(we2.tolist()))
wedb2 = doubler(we2)
weight2 = np.divide(np.subtract(wedb2,min2),max2-min2)
wim2 = Image.fromarray((weight2*255).astype(np.uint8),'L')
wim2.save('W2.bmp')
max2 = max(bi2.tolist())
min2 = min(bi2.tolist())
bidb2 = doubler(bi2)
bia2 = np.divide(np.subtract(bidb2,min2),max2-min2)
bias2 = np.array([bia2.tolist() for i in range(32)])
bim2 = Image.fromarray((bias2*255).astype(np.uint8),'L')
bim2.save('B2.bmp')

max3 = max(max(we3.tolist()))
min3 = min(min(we3.tolist()))
wedb3 = doubler(we3)
weight3 = np.divide(np.subtract(wedb3,min3),max3-min3)
wim3 = Image.fromarray((weight3*255).astype(np.uint8),'L')
wim3.save('W3.bmp')
max3 = max(bi3.tolist())
min3 = min(bi3.tolist())
bidb3 = doubler(bi3)
bia3 = np.divide(np.subtract(bidb3,min3),max3-min3)
bias3 = np.array([bia3.tolist() for i in range(32)])
bim3 = Image.fromarray((bias3*255).astype(np.uint8),'L')
bim3.save('B3.bmp')