import StackModule

stack = StackModule.Stack()
stack.push(10)
stack.push(20)
print(stack.to_string())  # Output: [10, 20]
print(stack.pop())        # Output: 20
print(stack.is_empty())   # Output: False
print(stack.top())        # Output: 10
