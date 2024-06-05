import sys

dictionary = {}

for line in sys.stdin.readlines():
    print(line)
    words = line.lower().split()
    for word in words:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1
print(dictionary)
