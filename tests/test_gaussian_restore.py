import unittest


try:
    import torch

    from scene.gaussian_model import GaussianModel

    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DEPENDENCIES_AVAILABLE = False


class _OptimizerStub:
    def __init__(self):
        self.param_groups = [{"name": f"rgb_{index}"} for index in range(6)]
        self.loaded = False

    def load_state_dict(self, state):
        if len(state["param_groups"]) != len(self.param_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        self.loaded = True


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "3DGS dependencies are optional locally")
class GaussianRestoreTests(unittest.TestCase):
    def test_in_place_joint_restore_recreates_semantic_optimizer_group(self):
        model = GaussianModel(3)
        model._semantic_features = object()
        optimizer = _OptimizerStub()
        model.training_setup = lambda _args: setattr(model, "optimizer", optimizer)
        semantic_features = torch.zeros((2, 4), dtype=torch.float32)
        observed = {}

        def setup_joint(dimensions, learning_rate, features=None, importance_score=None):
            observed["semantic_was_cleared"] = model._semantic_features is None
            if model._semantic_features is not None:
                return
            model._semantic_features = features
            model.optimizer.param_groups.append({"name": "semantic"})

        model.setup_joint_semantics = setup_joint
        tensors = [torch.zeros(1) for _ in range(10)]
        checkpoint_optimizer = {
            "state": {},
            "param_groups": [{} for _ in range(7)],
        }
        model_args = (
            3,
            *tensors[:9],
            checkpoint_optimizer,
            1.0,
            {
                "semantic_features": semantic_features,
                "semantic_lr": 0.01,
                "importance_score": torch.ones(2),
            },
        )

        model.restore(model_args, object())

        self.assertTrue(observed["semantic_was_cleared"])
        self.assertIs(model._semantic_features, semantic_features)
        self.assertTrue(optimizer.loaded)


if __name__ == "__main__":
    unittest.main()
