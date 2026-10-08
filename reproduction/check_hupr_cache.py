"""Check mapped inputs, labels and shape conversion against the release on CPU."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from hupr_runtime import MappedHuPR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--radar-maps', type=Path, required=True)
    parser.add_argument('--sequence', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('released_hupr', root/'Datasets/HuPR.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    assert (args.radar_maps / f'single_{args.sequence}.pkl').exists()
    results = []
    for representation in ('complex', 'cartesian', 'polar'):
        config = dict(dataset_path=str(args.cache), file_indexes=[args.sequence],
                      preload=False, format=representation, tqdm_disable=True)
        mapped = MappedHuPR(config)
        # Use the release's actual non-preloaded __getitem__ without asking it
        # to deserialize every file simply to construct the index.
        released = original.HuPR_Dataset.__new__(original.HuPR_Dataset)
        released.config = {**config, 'dataset_path': str(args.radar_maps.parent)}
        released.preload = False
        released.image_path = []
        released.data_paths = dict(
            file_path=[str(args.radar_maps/f'single_{args.sequence}.pkl')]*600,
            index=list(range(600)))
        converter = original.HuPR_data_shape_converter(
            dict(format=representation, model_input_shape='BCHW'))
        for frame in (0, 300, 599):
            actual, expected = mapped[frame], released[frame]
            assert actual[0].dtype == expected[0].dtype
            assert torch.equal(actual[0], expected[0]), (representation, frame)
            # The released training/evaluation function explicitly calls
            # label.float(). NumPy cache labels can be float64 before that.
            for got, want in zip(actual[1:], expected[1:]):
                assert torch.equal(got.float(), want.float()), (representation, frame)
            assert torch.equal(converter.shape_convert(actual[0][None]),
                               converter.shape_convert(expected[0][None]))
        results.append(dict(format=representation, frames=[0, 300, 599],
                            exact_inputs_and_float32_targets=True,
                            cached_label_dtype=str(actual[1].dtype),
                            released_label_dtype=str(expected[1].dtype)))
    print(json.dumps(dict(sequence=args.sequence, checks=results), indent=2))


if __name__ == '__main__':
    main()
