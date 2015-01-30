# -*- mode: Python; coding: utf-8 -*-
import os
from csv import *
from string import *

class ACEDialect(Dialect):
    """A CSV dialect for reading ACE BIO/POS data."""
    delimiter = '\t'
    doublequote = False
    lineterminator = '\n'
    quoting = QUOTE_NONE
    skipinitialspace = True

register_dialect('ace', ACEDialect)

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
        if not isinstance(window, list):
            raise ValueError, 'window must be a list of integers'
        if not all(isinstance(item, int) for item in window):
            raise ValueError, 'window must be a list of integers'
        bounds = range(len(self))
        for index, item in enumerate(self):
            # print(index)
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
            label2 = u'{}[{}]={}'.format('RC','0', self.resourceConfirm(index, NEdict))
            values.append(label2)
            yield values
    def resourceConfirm(self, index, NEdict):
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

    def ngramize(self, n):
        """Returns a tuple of n-grams in the sentence."""
        slices = (slice(i, i+n) for i in range(max(len(self)-n+1, 0)))
        return tuple(self[s] for s in slices)

def text(token):
    """The token itself (as a unicode object)."""
    if isinstance(token, Token):
        token = text(token.text)
    if isinstance(token, str):
        token = text(token.decode('utf-8'))
    if isinstance(token, unicode):
        return token

def nopunct(token):
    """The token itself stripped of leading/trailing punctuation."""
    return text(token).strip(punctuation)

def pos(token):
    """The part-of-speech tag."""
    if isinstance(token, Token):
        return token.pos

def orth(token):
    """A binary representation of the unicode.

    E.g.:
        u'...' -> '000'
        u'abc' -> '000'
        u'abC' -> '001'
        u'Abç' -> '100'
        u'AbC' -> '101'
        u'ÀBC' -> '111'
    """
    return ''.join(map(str,map(int,(map(unicode.isupper, text(token))))))

def cap(token):
    """True if every character in the token is capitalized, otherwise False."""
    return unicode.isupper(text(token))

def title(token):
    """True if the token is titlecase, otherwise False."""
    return unicode.istitle(text(token))

def alnum(token):
    """True if the token is composed of strictly alphanumeric characters,
    False otherwise."""
    return unicode.isalnum(text(token))

def alpha(token):
    """True if the token is composed of strictly alphabetic characters,
    False otherwise."""
    return unicode.isalpha(text(token))

def num(token):
    """True if the token is composed of strictly numerical characters,
    False otherwise."""
    return unicode.isnumeric(text(token))

def first(token):
    """True if the token is the first token in the sentence, otherwise False."""
    if isinstance(token, Token):
        return token.index == 0

def tail(token):
    """The last four characters (or fewer) of the token."""
    return text(token)[-4:]

currencies = u'$¢£¤¥ÐŁƒɃ৳฿ლ₡₤₥₦₨₩₪₫€₭₮₱₲₴₽元円'
punctuation += u'‐-‑⁃֊᠆‧·¡!¿?⸘‽“”‘’‛‟.,‚„′″´˝^°¸˛¨`˙˚ªº…&_¯­–‑—§⁊¶†‡@‰‱¦ˉˆ˘ˇ‼︎⁇⁈⁉︎❛❜❝❞❢❣❡'
def shape(token):
    """Return a general shape of the token.

    E.g., 'Abc' -> '
    """
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

def write_crf_data(in_path, out_path):
    with open(in_path, 'rb') as in_file:
        with open(out_path, 'wb') as out_file:
            csv_reader = reader(in_file, 'ace')
            csv_writer = writer(out_file, 'ace')
            sentence = Sentence()
            for row in csv_reader:
                if not row:

                    for line in sentence.featurize(features, NEDict, range(-2,3)):
                        csv_writer.writerow(line)
                    sentence = Sentence()
                else:
                    sentence += Token(*row)

def getNEDict(filename = 'named_entity_lists'):
    NEDict = {}
    folder = os.listdir(filename)
    files = []
    for i in folder:
        files.append(i)
    for i in files:
        print(i)
        f = open(filename + '/' + i, 'r')
        for line in f.readlines():
            items = line.split()
            following = []
            following.append(items[0])
            for k in range(len(items)):
                if k >= 2 and k < len(items):
                    following.append(items[k])
            if not items[1] in NEDict.keys():
                followings = []
                followings.append(following)
                NEDict[items[1]] = followings
            else:
                NEDict[items[1]].append(following)
                # print(items[1], NEDict[items[1]])
    return NEDict




if __name__ == '__main__':
    NElen = 0;
    INType =''
    NEDict = getNEDict()
    print('done')

    train_path = os.path.join('project1-train-dev', 'train.gold')
    dev_path = os.path.join('project1-train-dev', 'dev.gold')
    crf_train = 'train.crfsuite.txt'
    crf_test = 'dev.crfsuite.txt'
    features = text, nopunct, pos, cap, title, alnum, num, first, tail, shape
    write_crf_data(train_path, crf_train)
    write_crf_data(dev_path, crf_test)
