import idx2numpy


class DataLoader:
    def __init__(self, batch_size=32):
        self.data = idx2numpy.convert_from_file('train-images.idx3-ubyte')
        self.data = self.data.reshape(-1, 28 ** 2).astype('float32') / 255.  # reshape and normalize
        self.labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
        self.batch_size = batch_size

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        self.idx = 0
        self.max = len(self)
        return self

    def __next__(self):
        if self.idx < self.max:  # keep iterating
            start, stop = self.idx, min(self.max, self.idx + self.batch_size)
            self.idx += self.batch_size
            return self.data[start:stop, :].reshape(-1, 28 ** 2), self.labels[self.labels[start:stop].reshape(-1, 1)]
        else:  # end the iteration
            raise StopIteration
