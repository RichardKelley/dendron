from collections.abc import MutableMapping

class Blackboard(MutableMapping):

    def __init__(self, data=()):
        self.mapping = {}
        self.update(data)

    def __getitem__(self, key):
        return self.mapping[key]

    def get(self, key):
        return self.mapping[key]

    def __delitem__(self, key):
        v = self.mapping[key]
        del self.mapping[key]
        self.pop(v, None)

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def set(self, key, value):
        self.mapping[key] = value

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"Blackboard({self.mapping})"

    
