import unittest

import torch
import torch.nn as nn

from orivane import PatchBase, PatchType, inject, patch

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 3)])

    def forward(self, inputs):
        return self.layers[0](inputs)


class PatchTests(unittest.TestCase):
    def test_patch_modules_are_optimized_and_unpatch_restores_state(self):
        model = TinyModel()
        model.layers[0].bias.requires_grad_(False)
        original_requires_grad = [
            parameter.requires_grad for parameter in model.parameters()
        ]
        inputs = torch.randn(4, 3)

        @patch(model=model)
        class AdapterPatch(PatchBase):
            def __init__(self):
                super().__init__()
                self.adapter = nn.Linear(3, 3)

            @inject(method="forward", at=PatchType.AFTER)
            def add_adapter(self, output):
                return output + self.adapter(output)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.layers.parameters()
            )
        )
        patch_parameters = list(AdapterPatch.trainable_parameters())
        self.assertTrue(patch_parameters)
        self.assertTrue(
            set(map(id, patch_parameters))
            <= {id(parameter) for parameter in model.parameters()}
        )
        self.assertTrue(
            any("_enceladus_patch_0.adapter" in key for key in model.state_dict())
        )

        optimizer = torch.optim.SGD(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=0.1,
        )
        self.assertEqual(
            {id(parameter) for group in optimizer.param_groups for parameter in group["params"]},
            {id(parameter) for parameter in patch_parameters},
        )
        model(inputs).sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in patch_parameters))

        model.eval()
        self.assertFalse(AdapterPatch.training)
        model.train()
        self.assertTrue(AdapterPatch.training)

        AdapterPatch.unpatch()
        AdapterPatch.unpatch()
        self.assertEqual(
            [parameter.requires_grad for parameter in model.parameters()],
            original_requires_grad,
        )
        self.assertFalse(any("_enceladus_patch_" in key for key in model.state_dict()))
        self.assertFalse(AdapterPatch.__enceladus_patched__)

    def test_dotted_path_targets_nested_module_and_unpatch_restores_it(self):
        model = TinyModel()
        inputs = torch.randn(2, 3)
        expected = model.layers[0](inputs) * 2
        original_function = model.layers[0].forward.__func__

        @patch(model=model, train_only_added=False)
        class NestedPatch(PatchBase):
            def __init__(self):
                super().__init__()

            @inject(method="layers.0.forward", at=PatchType.AFTER)
            def double_output(self, output):
                return output * 2

        torch.testing.assert_close(model.layers[0](inputs), expected)
        NestedPatch.unpatch()
        self.assertIs(model.layers[0].forward.__func__, original_function)
        torch.testing.assert_close(model(inputs), model.layers[0](inputs))

    def test_before_callback_can_replace_arguments(self):
        model = nn.Linear(3, 2)
        inputs = torch.randn(2, 3)
        expected = model(inputs + 1)

        @patch(model=model, train_only_added=False)
        class InputPatch(PatchBase):
            def __init__(self):
                super().__init__()

            @inject(method="forward", at=PatchType.BEFORE)
            def shift_input(self, inputs):
                return (inputs + 1,), {}

        torch.testing.assert_close(model(inputs), expected)
        InputPatch.unpatch()


if __name__ == "__main__":
    unittest.main()
