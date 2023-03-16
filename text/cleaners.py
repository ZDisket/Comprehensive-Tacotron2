""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from .numbers import normalize_numbers
from phonemizer import phonemize
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
spanish_phonemizer = phonemizer.backend.EspeakBackend(language='es-419', preserve_punctuation=True,  with_stress=True)


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')



from g2p_en import G2p as grapheme_to_phn

_punctuation = "!'(),.:;?"
_special = "- "
_arpa_exempt = _punctuation + _special

arpa_g2p = grapheme_to_phn()

def to_arpa(in_str):
    phn_arr = arpa_g2p(in_str)

    arpa_str = "{"
    in_chain = True

    # Iterative array-traverse approach to build ARPA string. Phonemes must be in curly braces, but not punctuation
    for token in phn_arr:
        
        if token in _arpa_exempt and in_chain:
            arpa_str += "}"
            in_chain = False
            

        if token not in _arpa_exempt and not in_chain:
            arpa_str += "{"
            in_chain = True

   
        arpa_str += " " + token

    if in_chain:
        arpa_str += "}"
        
    #removing multiple whitespace
    arpa_str = " ".join(arpa_str.split())
    arpa_str = arpa_str.replace("{ ","{")
    # string results in double-dir spaced punctuation "{M AY1} , {M AY1}", we want it only single right spaced: "{M AY1}, {M AY1}"
    for p in _punctuation:
        arpa_str = arpa_str.replace(f" {p}",f"{p}")
    
    return arpa_str



# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def vbasic_cleaners(text):
    '''Only collapses whitespaces, for already processed text'''
    text = collapse_whitespace(text)
    return text

def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners2(text):
  '''Pipeline for IPA English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)
  phonemes = phonemes[0]
  phonemes = collapse_whitespace(phonemes)
  return phonemes


def spanish_cleaners(text):
  '''Pipeline for IPA Spanish text, punctuation + stress'''
  text = lowercase(text)
  phonemes = spanish_phonemizer.phonemize([text], strip=True, njobs=1)
  phonemes = phonemes[0].replace("(es-419)","").replace("(en)","")
  phonemes = collapse_whitespace(phonemes)
  return phonemes