# save_nn.py
# small tool to save the neural network state. append to schroedinger_nn.py notebook.
import csv
with open('W1.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W1).tolist())
with open('W2.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W2).tolist())
with open('W3.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W3).tolist())
with open('B1.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B1).tolist()])
with open('B2.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B2).tolist()])
with open('B3.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B3).tolist()])
