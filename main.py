class student:
    std = "10"
    def __init__(self):
        self.class_ = "somethign"


obj1 = student()
obj2 = student()

obj2.std = "20"
print(obj2.std)