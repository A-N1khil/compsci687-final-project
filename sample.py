class Toledo:
    def greet(self):
        return "Hello from Toledo!"

    def greet_with_name(self, name):
        return f"Hello, {name}, from Toledo!"

    def farewell(self):
        return "Goodbye from Toledo!"

toledo = Toledo()
toledo.greet()
print(toledo.greet())
print(toledo.greet_with_name("Alice"))
print(toledo.farewell())