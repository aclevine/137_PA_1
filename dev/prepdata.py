# -*- mode: Python; coding: utf-8 -*-
import os
from csv import *
from string import *

# LOAD RESOURCES
currencies = u'$Â¢Â£Â¤Â¥ÃÅÆ’Éƒà§³à¸¿áƒšâ‚¡â‚¤â‚¥â‚¦â‚¨â‚©â‚ªâ‚«â‚¬â‚­â‚®â‚±â‚²â‚´â‚½å…ƒå††'
punctuation += u'â€-â€‘âƒÖŠá †â€§Â·Â¡!Â¿?â¸˜â€½â€œâ€â€˜â€™â€›â€Ÿ.,â€šâ€žâ€²â€³Â´Ë^Â°Â¸Ë›Â¨`Ë™ËšÂªÂºâ€¦&_Â¯Â­â€“â€‘â€”Â§âŠÂ¶â€ â€¡@â€°â€±Â¦Ë‰Ë†Ë˜Ë‡â€¼ï¸Žâ‡âˆâ‰ï¸Žâ›âœââžâ¢â£â¡'

def load_brown_clusters(brown_input_path):
    brown_dict = {}
    with open(brown_input_path) as fo:
        for line in fo:
            word, binary, cluster_id = line.split()
            brown_dict[word] = cluster_id
    return brown_dict

brown_dict = load_brown_clusters('./resources/brown_clusters.txt')

def load_gazetteer(gaze_input_path):
    gazetteer = {}
    with open(gaze_input_path) as fo:
        for line in fo:
            entity_type, entity = line.split(' ', 1)
            gazetteer[entity] = entity_type
    return gazetteer

gazetteer = load_gazetteer('./resources/named_entity_lists/eng.list')

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
    """docstring for Token"""
    def __init__(self, index, text, pos, bio):
        self.index = index
        self.text = text
        self.pos = pos
        self.bio = bio
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._args())
    
    def _args(self):
        """Returns an ordered list of t for string representation."""
        args = self.index, self.text, self.pos, self.bio
        return ', '.join(map(repr, args))

class Sentence(tuple):
    """docstring for Sentence"""
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
    
    def featurize(self, features, NEdict, window=range(1)):
        #Preciously: def featurize(self, features, window=range(1)):
        #Chen: I added NEDict as a parameter for the method resourceConfirm();
        if not isinstance(window, list):
            raise ValueError, 'window must be a list of integers'
        if not all(isinstance(item, int) for item in window):
            raise ValueError, 'window must be a list of integers'
        bounds = range(len(self))
        for index, item in enumerate(self):
            values = [item.bio]
            context = [index + offset for offset in window]
            f_dict = dict((f.__name__, f) for f in features)
            for feature, function in f_dict.iteritems():
                for i in context:
                    if i in bounds:
                        value = function(self[i])
                        if isinstance(value, basestring):
                            for char in '=:':
                                value.replace(char, '.')
                        label = u'{}[{}]={}'.format(feature, i-index, u'{}')
                        # ':' and '=' are special delimiters
                        values.append(label.format(value).replace(':', '.'))
            ##### Chen: The code bellow adds the feature RC to see if self[index].text appears in the resources.
            label2 = u'{}[{}]={}'.format('RC','0', self.resourceConfirm(index, NEdict))
            values.append(label2)
            ##### End
            yield values


    def resourceConfirm(self, index, NEdict):
        #This function takes in a Sentence object and an index and the NEDict dictinary as input, returns a string that has the form: B-PER or I-LOC.
        #For example, if there is a sentence 'I am Kurt Heimpel' in the dev data and 'Kurt Heimpel' appears in the resources, then resourceConfirm(self,0,NEdict) and resourceConfirm(self,1,NEdict) would return 'None', while resourceConfirm(self,2,NEdict) would return 'B-PER' and resourceConfirm(self,3,NEdict) would return 'I-PER'.
        global NElen
        global INType
        flag = True
        if NElen > 0:
            NElen = NElen-1
            return u'RCI-{}'.format(INType)
        else:
            if self[index].text in NEdict.keys():
                for name in NEdict[self[index].text]:
                    for k in range(1,len(name)):
                        if index + k < len(self) and name[k] != self[index + k]:
                            flag = False
                    if flag:
                        
                        NElen = len(name) - 1
                        INType = u'{}'.format(name[0])
                        flag = True
                        return u'RCB-{}'.format(name[0])
            return u'None'
            else:
                return u'None'
    
#     def ngramize(self, n):
#         """Returns a tuple of n-grams in the sentence."""
#         slices = (slice(i, i+n) for i in range(max(len(self)-n+1, 0)))
#         return tuple(self[s] for s in slices)

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
    
def prev_trigram(sent, i):
    """target and 2 preceding tokens in sentence"""
    return trigram(sent, i-1)

def next_trigram(sent, i):
    """target and 2 following tokens in sentence"""
    return trigram(sent, i+1)

def entity_type(sent, i):
    """check token against gazetteer, backoff as needed"""
    #trigrams
    trigram = trigram(sent,i).replace('_', ' ')
    if trigram in gazetteer:
        return gazetteer[trigram]
    prev_trigram = prev_trigram(sent,i).replace('_', ' ')
    if prev_trigram in gazetteer:
        return gazetteer[prev_trigram]
    next_trigram = next_trigram(sent,i).replace('_', ' ')
    if next_trigram in gazetteer:
        return gazetteer[next_trigram]    
    #bigrams
    prev_bigram = prev_bigram(sent,i).replace('_', ' ')
    if prev_bigram in gazetteer:
        return gazetteer[prev_bigram]
    next_bigram = next_bigram(sent,i).replace('_', ' ')
    if next_bigram in gazetteer:
        return gazetteer[next_bigram]    
    #unigram
    token = sent[i]
    return gazetteer.get(token, 'None')

    
def brown_cluster_id(sent, i):
    token = sent[i]
    return brown_dict.get(token.text, -1)
    
def unigram(sent, i):
    return text(sent[i])

def nopunct(sent, i):
    """The token itself stripped of leading/trailing punctuation."""
    token = sent[i]
    return text(token).strip(punctuation)

def pos(sent, i):
    """The part-of-speech tag."""
    token = sent[i]
    if isinstance(token, Token):
        return token.pos

def orth(sent, i):
    """A binary representation of the unicode.
    E.g.:
        u'...' -> '000'
        u'abc' -> '000'
        u'abC' -> '001'
        u'AbÃ§' -> '100'
        u'AbC' -> '101'
        u'Ã€BC' -> '111'
    """
    token = sent[i]
    return ''.join(map(str,map(int,(map(unicode.isupper, text(token))))))

def cap(sent, i):
    """True if every character in the token is capitalized, otherwise False."""
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
    
    E.g., 'Abc' -> '
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


#===============================================================================
def write_crf_data(in_path, out_path, features):
    with open(in_path, 'rb') as in_file:
        with open(out_path, 'wb') as out_file:
            csv_reader = reader(in_file, 'ace')
            csv_writer = writer(out_file, 'ace')
            sentence = Sentence()
            for row in csv_reader:
                if not row:
                    for line in sentence.featurize(features, NEDict, range(-2,3)):
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
                    csv_writer.writerow([' '.join([token.text for token in sentence])])
                    sentence = Sentence()
                else:
                    sentence += Token(*row)

if __name__ == '__main__':
    NElen = 0;
    INType =''
    NEDict = getNEDict()
    #NEDict is a dictionary that has the form: (for example)
    #NEDict[''Kurt''] = [['PER', 'Heimpel'], ['PER', 'Pariser'], ['PER', 'Schumacher'], ['PER', 'Sontheimer'], ['PER', 'Stadler'], ['PER', 'Strentz']]

# It stores the following info in the resources
#PER Kurt Heimpel
#PER Kurt Pariser
#PER Kurt Schumacher
#PER Kurt Sontheimer
#PER Kurt Stadler
#PER Kurt Strentz

    train_path = os.path.join('resources', 'project1-train-dev', 'train.gold')
    dev_path = os.path.join('resources', 'project1-train-dev', 'dev.gold')
    crf_train = 'train.crfsuite.txt'
    crf_test = 'dev.crfsuite.txt'
    features = unigram, nopunct, pos, cap, title, alnum, num, first, tail, shape, trigram
    write_crf_data(train_path, crf_train, features)
    write_crf_data(dev_path, crf_test, features)
    
    #write_sent_data('./resources/project1-train-dev/dev.gold, 'sent.txt')
    
