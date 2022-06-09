from .autoregression.Autoregression import Autoregression
from .classifier.BLSTM import BLSTM
from .classifier.DecisionTree import DecisionTree
from .classifier.ExtraTrees import ExtraTrees
from .classifier.IsolationForest import IsolationForest
from .classifier.NaiveBayes import NaiveBayes
from .classifier.RandomForest import RandomForest
from .classifier.SVM import SVM
from .interarrivaltime.Mean import InterArrivalTimeMean
from .interarrivaltime.Range import InterArrivalTimeRange
from .oracles.OptimalIDS import OptimalIDS
from .oracles.DummyIDS import DummyIDS
from .simple_process_state.simple_process_state import SimpleProcessState

idss = [
    Autoregression,
    BLSTM,
    DecisionTree,
    ExtraTrees,
    InterArrivalTimeMean,
    InterArrivalTimeRange,
    IsolationForest,
    OptimalIDS,
    DummyIDS,
    RandomForest,
    SVM,
    SimpleProcessState,
]


def get_all_iidss():
    return {ids._name: ids for ids in idss}
