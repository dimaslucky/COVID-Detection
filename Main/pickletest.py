os.chdir('/home/m13518003/Tugas Akhir/Pickles')

with open('Raw_Audio.pkl', 'rb') as file:  # Python 3: open(..., 'rb')
    pickled1, pickled2 = pickle.load(file)

print(pickled1.shape)
print(pickled2.shape)