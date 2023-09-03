
origin = ['A','G','C','C','T']
print('origin:')
print(origin)
origin.reverse()
print('sequence_reverse:')
print(origin)
def sequence_reverse(origin):
    for i in range(len(origin)):
        if origin[i] == 'A':
            origin[i] = 'T'
        elif origin[i] == 'T':
            origin[i] = 'A'
        elif origin[i] == 'C':
            origin[i] = 'G'
        elif origin[i] == 'G':
            origin[i] = 'C'
        elif origin[i] == 'T':
            origin[i] = 'A'

sequence_reverse(origin)
print("reversed")
print(origin)
