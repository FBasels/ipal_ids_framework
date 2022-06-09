import json
import ipal_iids.settings as settings
from ids.ids import MetaIDS


class SimpleProcessState(MetaIDS):
    _name = "SimpleProcessState"
    _description = "Tracking of min-max changes of values in timeframe"
    _requires = ["train.state", "live.state"]
    _proc_state_default_settings = {
        "n": 20750,  # timeframe
        "q": 0.1,  # learning rate
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._proc_state_default_settings)

        self.q_min = {}
        self.q_max = {}
        self.slid_win = []

    def update_min_max(self, win: [], s_min: {}, s_max: {}) -> ({}, {}):
        # create missing keys in s_min/s_max
        topics = win[0]["state"].keys()
        for t in topics:
            if t not in s_min.keys():
                s_min[t] = 0
                s_max[t] = 0
        # update s_min/s_max
        for curr_state in win:
            for t in topics:
                dif = curr_state["state"][t] - win[0]["state"][t]
                if dif < s_min[t]:
                    s_min[t] = dif
                elif dif > s_max[t]:
                    s_max[t] = dif
        return s_min, s_max

    def train(self, ipal=None, state=None):
        win = []
        s_min = {}
        s_max = {}

        with self._open_file(state) as f:
            for line in f.readlines():
                curr = json.loads(line)
                if not win or win[0]["timestamp"] + self.settings["n"] >= curr["timestamp"]:
                    win.append(curr)
                else:
                    # do until the time of the current state is in the time window
                    while win and win[0]["timestamp"] + self.settings["n"] < curr["timestamp"]:
                        s_min, s_max = self.update_min_max(win, s_min, s_max)
                        win.pop(0)
                    win.append(curr)

            s_min, s_max = self.update_min_max(win, s_min, s_max)
            # calculate q_min and q_max
            for k in s_min.keys():
                self.q_min[k] = s_min[k] + self.settings["q"] * (s_max[k] - s_min[k])
                self.q_max[k] = s_max[k] + self.settings["q"] * (s_max[k] - s_min[k])
            settings.logger.info("-" * 10 + "TRAINING RESULTS" + "-" * 10 +
                                 "\nq_min: {}".format(self.q_min) +
                                 "\nq_max: {}".format(self.q_max) +
                                 "\n-" * 36)
            print("-" * 10 + "TRAINING RESULTS" + "-" * 10)
            print("q_min: {}".format(self.q_min))
            print("q_max: {}".format(self.q_max))
            print("-" * 36)
        print("Hey yo! Do not disturb, I am trying to train myself.")

    def new_state_msg(self, msg):
        # remove all msg from window which are older than n seconds
        while self.slid_win and self.slid_win[0]["timestamp"] + self.settings["n"] < msg["timestamp"]:
            self.slid_win.pop(0)

        self.slid_win.append(msg)
        alert = False
        res = {}
        for topic in self.slid_win[0].keys():
            dif = msg["state"][topic] - self.slid_win[0]["state"][topic]
            if dif < self.q_min:
                alert = True
                res[topic] = dif
            elif dif > self.q_max:
                alert = True
                res[topic] = dif
        return alert, res
