class gharaffaConstantCyclePolicy:

    def __init__(self, all_red_frequency=1):
        self.all_red_frequency = all_red_frequency
        self.PHASES = {
        0: "G E",
        1: "G N",
        2: "G W",
        3: "G S",
        4: "ALL RED",
        5: "G EW",
        6: "G NS",
        7: "G E BY W",
        8: "G N BY S",
        9: "G S BY N",
        10: "G W BY E"
        }
        self.cycle = [1, 9, 2, 7, 6, 10, 0, 8, 3, 5]
        #self.cycle = [5, 5, 5, 5, 6, 6] # closer to default 0 policy in sumo, follows the cycle with all red
        self.all_red_phase = 4
        self.next_phase = 0
        self.cycle_count = 0

    def select_action(self, state, eval=False):
        if self.next_phase == -1:
            action = self.all_red_phase
            self.next_phase = 0
        else:
            action = self.cycle[self.next_phase]
            self.next_phase = (1 + self.next_phase) % len(self.cycle)
            if self.next_phase == 0:
                self.cycle_count += 1
                if self.cycle_count % self.all_red_frequency == 0:
                    self.next_phase = -1

        return action

    def reset(self):
        self.next_phase = 0
        self.cycle_count = 0

    def print_stat(self):
        pass
