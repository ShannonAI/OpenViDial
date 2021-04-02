class Vocab(object):

    def __init__(self, file):
        self.word2id = dict()
        word_cnt = 0
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                self.word2id[line] = word_cnt
                word_cnt += 1
            f.close()
        self.id2word = dict()
        for key, value in self.word2id.items():
            self.id2word[value] = key
    
    def __getitem__(self, word):
        return self.word2id[word]
    
    def __len__(self):
        return len(self.word2id)
    
    def __contains__(self, word):
        return word in self.word2id

    def id2word(self, id):
        return self.id2word[id]
    