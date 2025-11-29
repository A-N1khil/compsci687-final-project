class FortressOfSolitude:
    def __init__(self):
        self.location = "Arctic"
        self.is_secret = True

    def enter(self):
        return "Welcome to the Fortress of Solitude."

    def get_location(self):
        return self.location

    def reveal_secret(self):
        return base_secret()

fos = FortressOfSolitude()
print(fos.enter())
print(fos.get_location())
