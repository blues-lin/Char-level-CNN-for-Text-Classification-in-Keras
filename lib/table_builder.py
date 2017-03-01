""" Build char table into file."""
import sys


class LookupTableBuilder:
    """Build look-up table into txt file with \n. """

    def __init__(self, fileName):
        self._allChar = set()
        self._fileName = fileName

    def addChar(self, c):
        self._allChar.add(c)

    def saveChar(self):
        nChar = len(self._allChar)
        self._f = open(self._fileName, "w", encoding='utf-8')
        print("Saving {0} chars into file: {1}".format(nChar, self._fileName))
        for c in self._allChar:
            try:
                self._f.write(c)
            except:
                print(c)
                print("error:", sys.exc_info()[0])
        self._f.close()
