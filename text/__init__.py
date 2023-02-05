""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols_en, symbols_en_1, symbols_cmn, symbols_ce

class SymbolsManager:
  def __init__(self, symbols):
    self.symbols = symbols
    # Mappings from symbol to numeric ID and vice versa:
    self._symbol_to_id = {s: i for i, s in enumerate(symbols)}
    self._id_to_symbol = {i: s for i, s in enumerate(symbols)}

    # Special symbol ids
    self.SPACE_ID = symbols.index(" ")

def create_symbols_manager(language):
  if language == 'cmn':
    symbols_manager = SymbolsManager(symbols_cmn)
  elif language == 'ce':
    symbols_manager = SymbolsManager(symbols_ce)
  elif language == 'en':
    symbols_manager = SymbolsManager(symbols_en_1)
  elif language == 'default':
    symbols_manager = SymbolsManager(symbols_en)
  else:
    symbols_manager = SymbolsManager(symbols_en)

  return symbols_manager

def text_to_sequence(text, cleaner_names, symbol_to_id):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    symbol_id = symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text, symbol_to_id):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence


def sequence_to_text(sequence, id_to_symbol):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
