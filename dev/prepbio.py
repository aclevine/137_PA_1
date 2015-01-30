# -*- mode: Python; coding: utf-8 -*-
"""
A script for preparing BIO (beginning inside outside) sequential data from ACE 
named-entity stand-off annotations.

This script is dependent on the following software:
     
     Tokenization : Marc Verhagen's English lexer/tokenizer
     POS tagging : NLTK (Natural Language Toolkit)
     SGML and XML parsing : BeautifulSoup
"""

__author__ = "Zachary Yocum"
__email__ = "zyocum@brandeis.edu"

import csv, os
from re import match

from bs4 import BeautifulSoup as BS
from nltk import pos_tag
from tokenizer import Tokenizer

class Extent(object):
    """A class for working with textual extents.
    
    An Extent's start represents the integer offset of the first 
    character in the extent. An Extent's end represents the integer offset 
    of the last character in the extent.  An Extent's content is the character 
    sequence that spans form the start of the extent to its end."""
    def __init__(self, start, end, content):
        if start > end:
            raise ValueError, 'start must be less than or equal to end'
        super(Extent, self).__init__()
        self.start = start
        self.end = end
        self.content = content
        self.span = slice(start, end)

    
    def __lt__(self, other):
        return self.span < other.span
    
    def __le__(self, other):
        return self.span <= other.span
    
    def __eq__(self, other):
        return self.span == other.span
    
    def __ne__(self, other):
        return self.span != other.span
    
    def __gt__(self, other):
        return self.span > other.span
    
    def __ge__(self, other):
        return self.span >= other.span
    
    def __cmp__(self, other):
        return cmp(self.span, other.span)
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._args())
    
    def _args(self):
        args = filter(None, (self.start, self.end, self.content))
        return ', '.join(map(repr, args))

class Token(Extent):
    """A word token."""
    def __init__(self, start, end, content, pos=None):
        super(Token, self).__init__(start, end, content)
        self.pos = pos
    
    def _args(self):
        """Returns an ordered list of arguments for string representation."""
        args = filter(None, (self.start, self.end, self.content, self.pos))
        return ', '.join(map(repr, args))

class NamedEntity(Extent):
    """An ACE named entity extent tag."""
    def __init__(self, start, end, content, entity_type):
        super(NamedEntity, self).__init__(start, end, content)
        self.entity_type = entity_type
    
    def _args(self):
        """Returns an ordered list of arguments for string representation."""
        args = (
            self.start,
            self.end,
            self.content,
            self.entity_type
        )
        return ', '.join(map(repr, args))

class SGMLDocument(BS):
    """A class for working with SGML documents."""
    def __init__(self, file):
        super(SGMLDocument, self).__init__(file)
        self.doc_id = self.docno.get_text().strip()
        self.tokenizer = Tokenizer(self.get_text())
        self.tokenizer.tokenize_text()
        self.lexes = self.tokenizer.lexes
        self.pos_tags = pos_tag([text for start, end, text in self.lexes])
        self.tokens = self._pos_tagged_tokens()
    
    def _pos_tagged_tokens(self):
        """Generates (token, part-of-speech) pairs from the document."""
        for lex, tagged_token in zip(self.lexes, self.pos_tags):
            start, end, text = lex
            token, pos = tagged_token
            yield Token(start, end, text, pos)

class ACEAnnotation(BS):
    """A class for working with ACE standoff annotations."""
    def __init__(self, file):
        super(ACEAnnotation, self).__init__(file, 'xml')
        self.doc_id = self.document.attrs['DOCID']
        self.named_entities = sorted(
            self._named_entities(),
            key=lambda ne : ne.start
        )
    
    def _named_entities(self):
        """Generates NamedEntities from the annotation."""
        for entity in self.findAll('entity'):
            for entity_mention in entity.findAll('entity_mention'):
                if entity_mention.attrs['TYPE'] == 'NAM':
                    char_seq = entity_mention.head.charseq
                    text = char_seq.get_text()
                    attrs = char_seq.attrs
                    start, end = map(int, (attrs['START'], attrs['END']))
                    start -= 1 # ACE start offsets are off-by-one
                    entity_type = entity.attrs['TYPE']
                    yield NamedEntity(start, end, text, entity_type)

class Corpus(object):
    """A class for working with collections of documents."""
    def __init__(self, directory, pattern, doc_type=BS, recursive=True):
        super(Corpus, self).__init__()
        self.directory = directory
        self.pattern = pattern
        self.recursive = recursive
        self.doc_type = doc_type
        self.index = dict(
            (document.doc_id, document) for document in self.documents()
        )
    
    def __contains__(self, doc_id):
        return doc_id in self.index
    
    def __getitem__(self, doc_id):
        return self.index[doc_id]
    
    def __len__(self):
        return len(self.index)
    
    def __repr__(self):
        return '<{}:{} with {} documents>'.format(
            self.__class__.__name__,
            self.directory,
            len(self)
        )
    
    def documents(self):
        """Generates all documents in the corpus."""
        results = self.find_files(self.directory, self.pattern, self.recursive)
        for path in results:
            with open(path, 'rb') as file:
                yield self.doc_type(file)
    
    def find_files(self, directory='.', pattern='.*', recursive=True):
        """Generates paths to all files in the specified directory matching 
        the given regex pattern.
        
        By default, all files in the current working directory are searched for 
        recursively."""
        if recursive:
            return (os.path.join(directory, file_name)
                for directory, subdirectories, file_names in os.walk(directory)
                for file_name in file_names if match(pattern, file_name))
        else:
            return (os.path.join(directory, file_name)
                for file_name in os.listdir(directory)
                if match(pattern, file_name))

class ACECorpus(Corpus):
    """A class for working with collections of ACEAnnotations."""
    def __init__(self, directory):
        super(ACECorpus, self).__init__(
            directory,
            pattern='.*\.xml',
            doc_type=ACEAnnotation
        )

class SGMCorpus(Corpus):
    """A class for working with collections of SGMLDocuments."""
    def __init__(self, directory):
        super(SGMCorpus, self).__init__(
            directory,
            pattern='.*\.sgm',
            doc_type=SGMLDocument
        )

def make_csvs(corpus_dir, csvs_dir):
    """Writes out CSV files for each document in the provided ACE data.  The 
    CSVs contain records with 'content', 'pos', and 'bio' fields for each 
    token in the corresponding document."""
    ace_corpus = ACECorpus(corpus_dir)
    sgm_corpus = SGMCorpus(corpus_dir)
    for doc_id in sgm_corpus.index:
        document = sgm_corpus[doc_id]
        annotation = ace_corpus[doc_id]
        named_entities = annotation.named_entities
        csv_path = os.path.join(csvs_dir, '{}.csv'.format(doc_id))
        with open(csv_path, 'wb') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['content', 'pos', 'bio'],
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            named_entity = named_entities.pop(0)
            for token in document.tokens:
                bio = 'O' # Default BIO tag
                # Check if it is appropriate to move on to the next named-entity
                if token.start > named_entity.end:
                    if named_entities:
                        named_entity = named_entities.pop(0)
                entity_type = named_entity.entity_type
                # Check if the token should be tagged as B or I
                if all(( 
                    token.start >= named_entity.start,
                    token.end <= named_entity.end
                )):
                    bio = 'I-{}'.format(entity_type)
                    if token.start == named_entity.start:
                        bio = 'B-{}'.format(entity_type)
                # Write the token, POS tag, and BIO tag to CSV
                writer.writerow(
                    dict(
                        content=token.content,
                        pos=token.pos,
                        bio=bio
                    )
                )

if __name__ == '__main__':
    from sys import argv
    usage = __doc__ + """Usage : python bioprep.py arg0 arg1
     arg0 : directory containing the .sgm and .xml input data files
     arg1 : directory where CSV output data files will be written
"""
    try:
        _, corpus_dir, csvs_dir = argv
    except ValueError:
        print usage
    else:
        make_csvs(corpus_dir, csvs_dir)