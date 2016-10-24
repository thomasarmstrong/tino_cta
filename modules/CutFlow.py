from astropy import units as u


class UndefinedCutException(Exception):
    pass


class CutFlow():

    def __init__(self, name="CutFlow"):
        self.name = name
        self.cuts = {}
        self.cut_names = []
        self.col_width = 20

    def count(self, name):
        if name not in self.cuts:
            self.cuts[name] = [None, 1]
            self.cut_names.append(name)
        else:
            self.cuts[name][1] += 1

    def set_cut(self, function, name):
        self.cuts[name] = [function, 0]
        if name not in self.cut_names:
            self.cut_names.append(name)

    def cut(self, name, *args, **kwargs):
        if name not in self.cuts:
            raise UndefinedCutException(
                "unknown cut {} -- only know: {}"
                .format(name, [a for a in self.cuts.keys()]))

        if self.cuts[name][0](*args, **kwargs):
            self.cuts[name][1] += 1
            return True
        else:
            return False

    def __getitem__(self, name):
        '''
        returns the function stored as @name to call with
        classifier["@name"](cut_parameter) '''
        if name not in self.cuts:
            raise UndefinedCutException(
                "unknown cut {} -- only know: {}"
                .format(name, [a for a in self.cuts.keys()]))
        return self.cuts[name][0]

    def __call__(self, base_cut="noCuts"):
        if base_cut not in self.cuts:
            raise UndefinedCutException(
                "unknown cut {} -- only know: {}"
                .format(base_cut, [a for a in self.cuts.keys()]))
        base_value = self.cuts[base_cut][1]

        print(self.name)
        print("Cut Name" +
              " "*(self.col_width-5-len("Cut Name")) +
              "passed Events" +
              " "*(self.col_width-8) +
              "efficiency")

        for id, cut in enumerate(self.cut_names):
            value = self.cuts[cut][1]
            col_buffer = 0
            if value < 10: col_buffer = 1
            if value > 99: col_buffer = -1
            print("{}{}{}{}{}".format(
                  cut,
                  " "*(self.col_width-len(cut)+col_buffer),
                  value,
                  " "*(self.col_width),
                  value/base_value*100*u.percent)
                  )

