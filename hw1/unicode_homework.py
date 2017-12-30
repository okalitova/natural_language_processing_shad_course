import unicodedata

mapping = {}
unicode_chars_number = 1114112
for char in range(0, unicode_chars_number):
    try:
        if (unicodedata.category(chr(char)) == "Nd"):
            mapping[chr(char)] = unicodedata.decimal(chr(char))
    except ValueError:
        pass

def decode_digits(string):
    digits = []
    for char in string:
        digits.append(mapping[char])
    return ''.join(map(str, digits))

def get_arabic_letters():
    arabic_letters = []
    unicode_chars_number = 1114112
    for char in range(0, unicode_chars_number):
        try:
            if unicodedata.name(chr(char)).find("ARABIC LETTER") != -1:
                arabic_letters.append(chr(char))
        except ValueError:
            pass
    return arabic_letters

def get_clean_arabic_letters():
    clean_arabic_letters = []
    for arrabic_letter in get_arabic_letters():
        if unicodedata.name(arrabic_letter).find("WITH") == -1:
            clean_arabic_letters.append(arrabic_letter)
    return clean_arabic_letters

def delete_diacritic(sent):
    return ''.join(c for c in unicodedata.normalize('NFKD', sent)
                  if unicodedata.category(c) != 'Mn')

