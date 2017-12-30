from itertools import product
from collections import defaultdict

class CYKParser:
    def __init__(self, rules):
        """
            `rules` - your grammar
        """
        ### YOUR CODE HERE
        self._rules = rules
        ### END OF YOUR CODE
    
    ### SOME ADDITION FUNCTIONS IF YOU NEED IT
    ###
    ### END YOUR ADDITIONAL FUNCTIONS
    
    def dfs(self, key, i, j):
        if i == j:
            return [key + ' -> ' + self._path[i][j][key]]
        B, C, i, k, j = self._path[i][j][key]
#         print(B, C, i, k, j)
#         print(self.dfs(B, i, k -1) + [key + '->' + B + ',' + C] + self.dfs(C, k, j))
        return [key + ' -> ' + B + ' ' + C] + self.dfs(B, i, k -1) + self.dfs(C, k, j)

    def parse(self, in_tokens):
        """
            `in_tokens` - input sentence
            return True in case of sentence can be parsed, False otherwise
        """
        c = [[[] for i in range(len(in_tokens))] for j in range(len(in_tokens)) ]
        self._path = [[defaultdict(int) for i in range(len(in_tokens))] for j in range(len(in_tokens)) ]
        ### YOUR CODE HERE
        for j in range(len(in_tokens)):
            for key, values in self._rules.items():
                if in_tokens[j] in values:
                    c[j][j].append(key)
                    self._path[j][j][key] = in_tokens[j]
            for i in range(j - 1, -1, -1):
                for k in range(i + 1, j + 1):
                    for key, values in self._rules.items():
                        for value in values:
                            BC = value.split(" ")
                            if len(BC) != 2:
                                continue
                            [B, C] = BC
                            if B in c[i][k - 1] and C in c[k][j]:
                                c[i][j].append(key)
                                self._path[i][j][key] = (B, C, i, k, j)
        if 'S' in c[0][len(in_tokens) - 1]:
             return [self.dfs('S', 0, len(in_tokens) - 1)]
        return []


l = 'Left-Arc'
r = 'Right-Arc'
s = 'Shift'
R = 'Reduce'

EX_1 = [s, s, l, r, r, R, R, s, l, l, r, r, r, s, l, r, r, r, R, R, r, r, R, R, R, R, R, R]

EX_2 = [s, l, r, r, s, l, r, s, s, l, s, l, l, r, R, r, r, R, R, R, R, R]

EX_3 = None