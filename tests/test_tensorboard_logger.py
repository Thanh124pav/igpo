import importlib.util
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "tensorboard_logger",
    Path(__file__).parents[1] / "treetune" / "common" / "tensorboard_logger.py",
)
_tensorboard_logger = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tensorboard_logger)
CompositeLogger = _tensorboard_logger.CompositeLogger
TensorBoardLogger = _tensorboard_logger.TensorBoardLogger


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, key, value, step):
        self.scalars.append((key, value, step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


class FakeLogger:
    def __init__(self):
        self.calls = []
        self.saved = []
        self.metrics = []
        self.summary = {}

    def log(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def save(self, *args, **kwargs):
        self.saved.append((args, kwargs))

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))


def test_tensorboard_logger_writes_only_scalar_metrics(tmp_path):
    writer = FakeWriter()
    logger = TensorBoardLogger(tmp_path / "tb", writer=writer)

    logger.log(
        {
            "train/global_step": 7,
            "loss": 1.25,
            "flag": True,
            "nested": {"skip": 1},
            "text": "skip",
        }
    )

    assert writer.scalars == [
        ("train/global_step", 7.0, 7),
        ("loss", 1.25, 7),
        ("flag", 1.0, 7),
    ]
    assert writer.flush_count == 1


def test_tensorboard_logger_uses_explicit_or_incremental_steps(tmp_path):
    writer = FakeWriter()
    logger = TensorBoardLogger(tmp_path / "tb", writer=writer)

    logger.log({"metric": 3.0}, step=4)
    logger.log({"metric": 5.0})

    assert writer.scalars == [("metric", 3.0, 4), ("metric", 5.0, 5)]


def test_composite_logger_fans_out_and_mirrors_summary(tmp_path):
    fake = FakeLogger()
    writer = FakeWriter()
    tensorboard = TensorBoardLogger(tmp_path / "tb", writer=writer)
    composite = CompositeLogger([fake, tensorboard])

    composite.define_metric("metric")
    composite.log({"metric": 2.0}, step=3)
    composite.save("config.json", policy="now")
    composite.summary["answer_rate"] = 0.5

    assert fake.metrics == [(("metric",), {})]
    assert fake.calls == [(({"metric": 2.0},), {"step": 3})]
    assert fake.saved == [(("config.json",), {"policy": "now"})]
    assert writer.scalars == [("metric", 2.0, 3)]
    assert fake.summary["answer_rate"] == 0.5
    assert tensorboard.summary["answer_rate"] == 0.5
