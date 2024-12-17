class TimeClass(object):

    _time = 0

    @property
    def time(self):
        return type(self)._time

    @time.setter
    def time(self,val):
        type(self)._time = val

    def incrementTime(self):
        type(self)._time += 1