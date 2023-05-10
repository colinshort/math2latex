class Element:
    def __init__(self, val):
        self.val = val
    def __str__(self) -> str:
        return str(self.val)
a = []
for i in range(1, 6):
    a.append(Element(i))
for el in a:
    print(el)
print("\n")

i = 0
while i < len(a):
    curr = a[i]
    j = i + 1
    while j < len(a) - 1:
        next = a[j]
        if(abs(curr.val-next.val) <= 1):
            curr.val = curr.val + next.val
            a.remove(next)
            break
        j += 1
    i += 1

for el in a:
    print(el)