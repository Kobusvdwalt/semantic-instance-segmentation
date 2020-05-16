import pickle
import matplotlib.pyplot as plt
history_2 = pickle.load(open("./trainHistoryDict_2", "rb" ))
history_10 = pickle.load(open("./trainHistoryDict", "rb" ))


plt.plot(history_2['val_recall'])
plt.plot(history_2['val_precision'])
plt.plot(history_10['val_recall'])
plt.plot(history_10['val_precision'])
plt.xlabel('Epoch')

plt.legend(['val_recall', 'val_precision'], loc='upper left')

plt.show()



#plt.title('Precision over time')
#plt.ylabel('precision')