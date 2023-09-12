class Patient(object):

    def __init__(self, *args):
        if len(args) == 4:
            self.all_vessel_num = args[0]
            self.calc_vessel_num = args[1]
            self.max_dia = args[2]
            self.dia_sum = args[3]
            self.count = 1
        else:
            self.all_vessel_num = args[0]
            self.calc_vessel_num = args[1]
            self.max_dia = args[2]
            self.dia_sum = args[3]
            self.count = 0

    def setCount(self, count):
        self.count = count

    def update(self, all_vessel_num, calc_vessel_num, max_dia, dia_sum):
        self.all_vessel_num += all_vessel_num
        self.calc_vessel_num += calc_vessel_num
        self.max_dia = max(self.max_dia, max_dia)
        self.dia_sum += dia_sum
        self.count += 1
