import json
import ipal_iids.settings as settings
from ids.ids import MetaIDS


class ProbabilisticSuffixTree(MetaIDS):
    _name = "ProbabilisticSuffixTree"
    _description = "Probabilistic suffix tree based on last n messages"
    _requires = ["train.ipal", "live.ipal"]
    _proc_state_default_settings = {
        "n": 3,  # depth of the tree
        "threshold": 0.0208
    }

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._proc_state_default_settings)

        self.tree = {"count": 0}
        self.slid_win = []

    # creates node name in tree(=key for dictionary) for ipal message
    def get_key(self, ipal: json) -> str:
        src = ipal["src"].split(":")[0]
        dest = ipal["dest"].split(":")[0]
        return "({}, {}, {}, {}, {})".format(src, dest, ipal["type"], ipal["activity"], list(ipal["data"].keys()))

    # recurse function calculating likelihood of each node in tree
    def calc_likelihood(self, tree: {}, ref: int):
        if len(tree.keys()) == 2:
            tree["likelihood"] = tree["count"] / ref
        else:
            for k in tree.keys():
                if k != "count" and k != "msg_ref":
                    self.calc_likelihood(tree[k], ref)
            tree["likelihood"] = tree["count"] / ref

    def train(self, ipal=None, state=None):
        win = []
        msg_ref = 1     # used for visualization

        # build tree
        with self._open_file(ipal) as f:
            for line in f.readlines():
                curr = json.loads(line)

                if len(win) < self.settings["n"] - 1:   # collect first n-1 messages
                    win.append(curr)
                elif len(win) == self.settings["n"] - 1:
                    win.append(curr)
                    sub_tree = self.tree
                    sub_tree["count"] += 1
                    for msg in win:
                        key = str(self.get_key(msg)).replace('\'', '')
                        if key in sub_tree.keys():
                            sub_tree = sub_tree[key]
                            sub_tree["count"] += 1
                        else:
                            sub_tree[key] = {"count": 1, "msg_ref": msg_ref}
                            msg_ref += 1
                            sub_tree = sub_tree[key]
                    win.pop(0)
                else:
                    settings.logger.critical("Buffer window size exceeded, something went wrong.\n"
                                             "Removing first element and continuing...")
                    win.pop(0)
        # update labels of tree
        self.calc_likelihood(self.tree, self.tree["count"])

    def new_ipal_msg(self, msg):
        # remove oldest ipal message from buffer window
        if len(self.slid_win) == self.settings["n"]:
            self.slid_win.pop(0)

        key = str(self.get_key(msg)).replace('\'', '')
        self.slid_win.append(key)
        pos = self.tree
        for m in self.slid_win:
            if m not in pos.keys():
                return True, {m: "likelihood: 0"}
            else:
                pos = pos[m]
                if pos["likelihood"] < self.settings["threshold"]:
                    return True, {m: "likelihood: {}".format(pos["likelihood"])}
        return False, {}

    def save_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        model = {
            "_name": self._name,
            "settings": self.settings,
            "tree": self.tree,
        }

        with self._open_file(self._resolve_model_file_path(), mode="wt") as f:
            f.write(json.dumps(model, indent=4) + "\n")
        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None:
            return False

        try:  # Open model file
            with self._open_file(self._resolve_model_file_path(), mode="rt") as f:
                model = json.load(f)
        except FileNotFoundError:
            settings.logger.info(
                "Model file {} not found.".format(str(self._resolve_model_file_path()))
            )
            return False

        # Load model
        assert self._name == model["_name"]
        self.settings = model["settings"]
        self.tree = model["tree"]
        return True
