import easydict
from typing import Dict

class Record(easydict.EasyDict):
    def __init__(self, a=None):
        super(Record, self).__init__(a)
    
    def add(self, a: Dict):
        for k, v in a.items():
            if self.get(k) is None:
                self[k] = []
            self[k].append(v)
    
