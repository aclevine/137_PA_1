# -*- mode: Python; coding: utf-8 -*-
import os
from csv import *
from string import *

# Some special unicode characters
currencies = u'$¢£¤¥ÐŁƒɃ৳฿ლ₡₤₥₦₨₩₪₫€₭₮₱₲₴₽元円'
punctuation += u'‐-‑⁃֊᠆‧·¡!¿?⸘‽“”‘’‛‟.,‚„′″´˝^°¸˛¨`˙˚ªº…&_¯­–‑—§⁊¶†‡@‰‱¦ˉˆ˘ˇ‼︎⁇⁈⁉︎❛❜❝❞❢❣❡'

# LOAD RESOURCES
def load_brown_clusters(brown_input_path):
    brown_dict = {}
    with open(brown_input_path) as fo:
        for line in fo:
            word, binary, cluster_id = line.split()
            brown_dict[word] = cluster_id
    return brown_dict

def load_gazetteer(gaze_input_path):
    """ load entity type, entity pairs keyed on tokens"""
    gazetteer = {}
    with open(gaze_input_path) as fo:
        for line in fo:
            entity_type, entity = line.split(' ', 1)
            for position, word in enumerate(entity.split()):
                # save full chunk, token position in chunk, entity type
                entry = (position, entity.split(), entity_type)
                gazetteer[word] = gazetteer.get(word, []) + [entry]
    return gazetteer

def load_w2v_clusters(w2v_input_path):
    with open(w2v_input_path, 'rb') as file:
        lines = filter(None, file.read().split('\n'))
    return dict(line.split() for line in lines)

brown_dict = load_brown_clusters('./resources/brown_clusters.txt')
gazetteer = load_gazetteer('./resources/named_entity_lists/eng.list') # named entity dict
w2v_dict = load_w2v_clusters('./resources/words2vec-clusters.txt')

class ACEDialect(Dialect):
    """A CSV dialect for reading ACE BIO/POS data."""
    delimiter = '\t'
    doublequote = False
    lineterminator = '\n'
    quoting = QUOTE_NONE
    skipinitialspace = True

register_dialect('ace', ACEDialect)

#===============================================================================

class Token(object):
    """A token is a tuple-like object with four attributes:
        index : an integer representing the index of the token in the sentence
        text  : a string representation of the token
        pos   : a string representing the POS tag of the token
        bio   : a string representing the BIO tag"""
    def __init__(self, index, text, pos, bio):
        self.index = index
        self.text = text
        self.pos = pos
        self.bio = bio
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._args())
    
    def _args(self):
        """Returns an ordered list of arguments for string representation."""
        args = self.index, self.text, self.pos, self.bio
        return ', '.join(map(repr, args))

class Sentence(tuple):
    """A tuple wrapper class for working with sequences of Tokens."""
    def __init__(self, tokens=tuple()):
        if not all(isinstance(item, Token) for item in tokens):
            raise TypeError, 'expected a sequence of Tokens'
    
    def __add__(self, other):
        if not isinstance(other, (Sentence, Token)):
            raise TypeError, 'can only concatenate Sentence or Token'
        if isinstance(other, Token):
            return self + Sentence((other,))
        return Sentence(super(Sentence, self).__add__(other))
    
    def __iadd__(self, other):
        if not isinstance(other, (Sentence, Token)):
            raise TypeError, 'can only concatenate Sentence or Token'
        if isinstance(other, Token):
            return self + Sentence((other,))
        return self + other
    
    def __repr__(self):
        return '<{} with {} tokens>'.format(self.__class__.__name__, len(self))
    
    def __str__(self):
        return ' '.join(t.text.encode('utf-8') for t in self)
    
    def featurize(self, features, window=range(1)):
        #Preciously: def featurize(self, features, window=range(1)):
        #Chen: I added NEDict as a parameter for the method resourceConfirm();
        if not isinstance(window, list):
            raise ValueError, 'window must be a list of integers'
        if not all(isinstance(item, int) for item in window):
            raise ValueError, 'window must be a list of integers'
        bounds = set(range(len(self)))
        for index, item in enumerate(self): # sequence of Token objs
            values = [item.bio]
            context = [index + offset for offset in window]
            for f in features:
                for i in context:
                    if i in bounds:
                        value = f(self, i)
                        if isinstance(value, basestring):
                            # ':' and '=' are special delimiters
                            for char in set('=:'):
                                value.replace(char, '.')
                            if not value:
                                continue
                        label = u'{}[{}]={}'.format(f.__name__, i-index, '{}')
                        values.append(label.format(value).replace(':', '.'))
            yield values


# FEATURES
def text(token):
    """The token itself (as a unicode object)."""
    if isinstance(token, Token):
        token = text(token.text)
    if isinstance(token, str):
        token = text(token.decode('utf-8'))
    if isinstance(token, unicode):
        return token

def prev_n(sent, i, n=3):    
    """ n tokens before target in sentence"""
    end = i
    start = max(0, end-n)
    return '_'.join([t.text for t in sent[start:end]])

def next_n(sent, i, n=3):
    """ n tokens after target in sentence"""
    start = min(i+1, len(sent))
    end = min(start+n, len(sent))
    return '_'.join([t.text for t in sent[start:end]])

def prev_bigram(sent, i):
    """target and prev token in sentence"""
    start = max(0, i-1)
    end = min(i+1, len(sent))
    return '_'.join([t.text for t in sent[start:end]])

def next_bigram(sent, i):
    """target and next token in sentence"""
    return prev_bigram(sent, i+1)

def trigram(sent, i):
    """target and 2 surrounding tokens in sentence"""
    start = max(0, i-1)
    end = min(i+2, len(sent))
    return '_'.join([t.text for t in sent[start:end]])

def trigram_pos(sent, i):
    """POS tags for previous, target, and next token"""
    start = max(0, i-1)
    end = min(i+2, len(sent))
    return '_'.join([t.pos for t in sent[start:end]])

def prev_trigram(sent, i):
    """target and 2 preceding tokens in sentence"""
    return trigram(sent, i-1)

def next_trigram(sent, i):
    """target and 2 following tokens in sentence"""
    return trigram(sent, i+1)

def prev_bigram_pos(sent, i):
    """previous pos and target pos"""
    start = max(0, i-1)
    end = min(i+1, len(sent))
    return '_'.join([t.pos for t in sent[start:end]])

def next_bigram_pos(sent, i):
    """previous pos and target pos"""
    start = max(0, i+1)
    end = min(i+1, len(sent))
    return '_'.join([t.pos for t in sent[start:end]])

def entity_type(sent, i):
    """check token against gazetteer"""
    token = sent[i]
    entity_list = gazetteer.get(text(token), '')
    if entity_list:
        for position, entity, entity_type in entity_list:
            start = i - position
            end = start + len(entity)
            tokens = [text(t) for t in sent[start:end]]
            if tokens == entity:
                return entity_type
    return 'None'

def brown_cluster_id(sent, i):
    token = sent[i]
    return brown_dict.get(token.text, -1)

def w2vcluster(sent, i):
    """Returns the word2vec cluster ID of the token."""
    token = text(sent[i])
    return w2v_dict.get(text(token).lower(), -1)

def unigram(sent, i):
    return text(sent[i])

def nopunct(sent, i):
    """The token itself stripped of leading/trailing punctuation."""
    token = sent[i]
    return text(token).strip((punctuation))

def pos(sent, i):
    """The part-of-speech tag."""
    token = sent[i]
    if isinstance(token, Token):
        return token.pos

def length(sent, i):
    """The number of characters in the token."""
    token = sent[i]
    return len(text(token))

def cap(sent, i):
    """True if every character in the token is capitalized, False otherwise."""
    token = sent[i]
    return unicode.isupper(text(token))

def title(sent, i):
    """True if the token is titlecase, otherwise False."""
    token = sent[i]
    return unicode.istitle(text(token))

def alnum(sent, i):
    """True if the token is composed of strictly alphanumeric characters,
    False otherwise."""
    token = sent[i]
    return unicode.isalnum(text(token))

def alpha(sent, i):
    """True if the token is composed of strictly alphabetic characters,
    False otherwise."""
    token = sent[i]
    return unicode.isalpha(text(token))

def num(sent, i):
    """True if the token is composed of strictly numerical characters,
    False otherwise.""" 
    token = sent[i]
    return unicode.isnumeric(text(token))

def first(sent, i):
    """True if the token is the first token in the sentence, otherwise False."""
    token = sent[i]
    if isinstance(token, Token):
        return token.index == 0

def tail(sent, i):
    """The last four characters (or fewer) of the token."""
    token = sent[i]
    return text(token)[-4:]

def shape(sent, i):
    """Return a general shape of the token.
    
    E.g.:
        'Abc'  -> 'Xxx'
        'ABC'  -> 'XXX'
        'Àbc'  -> 'Uxx'
        '2015' -> 'dddd'
        'ab12' -> 'xxdd'
        'àçè'  -> 'uuu'
        'ÀÇÈ'  -> 'UUU'
        '€10'  -> '$dd'
        ' a '  -> 'x'
        '-a-'  -> '.x.'
    """
    token = sent[i]
    chars = []
    for c in text(token):
        if c in ascii_uppercase:
            chars += 'X'
        elif c in ascii_lowercase:
            chars += 'x'
        elif c.isspace():
            chars += ''
        elif c.isdigit():
            chars += 'd'
        elif c in currencies:
            chars += '$'
        elif c in punctuation:
            chars += '.'
        elif c.isupper():
            chars += 'U'
        else:
            chars += 'u'
    return u''.join(chars)

def simple_shape(sent, i):
    token = sent[i]
    return ''.join(('X' if char.isupper() else 'x') for char in text(token))

#===============================================================================
def write_crf_data(in_path, out_path, features):
    with open(in_path, 'rb') as in_file:
        with open(out_path, 'wb') as out_file:
            csv_reader = reader(in_file, 'ace')
            csv_writer = writer(out_file, 'ace')
            sentence = Sentence()
            for row in csv_reader:
                if not row:
                    for line in sentence.featurize(features, range(-1,2)):
                    #Previous: for line in sentence.featurize(features, range(-2,3)):
                    #Chen: I added a NEDict for the featurize function I changed.
                        csv_writer.writerow(line)
                    sentence = Sentence()
                else:
                    sentence += Token(*row)

def write_sent_data(in_path, out_path):
    with open(in_path, 'rb') as in_file:
        with open(out_path, 'wb') as out_file:
            csv_reader = reader(in_file, 'ace')
            csv_writer = writer(out_file, 'ace')
            sentence = Sentence()
            for row in csv_reader:
                if not row:
                    csv_writer.writerow(str(sentence))
                    sentence = Sentence()
                else:
                    sentence += Token(*row)

if __name__ == '__main__':
    train_path = os.path.join('resources', 'project1-train-dev', 'train.gold')
    dev_path = os.path.join('resources', 'project1-train-dev', 'dev.gold')
    crf_train = 'train.crfsuite.txt'
    crf_test = 'dev.crfsuite.txt'
    features = pos, cap, title, alnum, num, alpha, nopunct, first, length, \
               shape, simple_shape, \
               entity_type, brown_cluster_id, w2vcluster, \
               prev_bigram_pos, prev_bigram, \
               trigram_pos, trigram, next_trigram, prev_trigram
               #next_bigram_pos, next_bigram, \
    write_crf_data(train_path, crf_train, features)
    write_crf_data(dev_path, crf_test, features)
    
    #write_sent_data('./resources/project1-train-dev/dev.gold, 'sent.txt')
