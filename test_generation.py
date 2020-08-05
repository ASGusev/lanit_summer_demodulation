import json
import argparse
from typing import Dict, List

import numpy as np

from demodulation.data_preprocessing import generate_samples_per_snr, SnrSampleSet
import modulation


SnrSampleSetJson = Dict[int, List[Dict[str, np.ndarray]]]


def convert_samples_to_json(snr_samples: SnrSampleSet) -> SnrSampleSetJson:
    return {
        snr: [
            {
                'sample': [int(i) for i in sample],
                'wave': [float(i) for i in wave]
            }
            for sample, wave in samples
        ]
        for snr, samples in snr_samples.items()
    }


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--n_samples', type=int, default=5, help='Number of samples per SNR value')
    arg_parser.add_argument('--sample_len', type=int, default=50, help='Length of each sample')
    arg_parser.add_argument('--snr', type=int, nargs='+', default=[-2, -4, -6, -8, -10, -12],
                            help='Sound-noise ratios to generate samples for')
    arg_parser.add_argument('--shift', type=float, default=None,
                            help='Value that is added to all channels frequencies, as share of the channel width')
    args = arg_parser.parse_args()

    modulator = modulation.Modulator(frequency_shift=args.shift)
    test_samples = generate_samples_per_snr(args.n_samples, args.sample_len, modulator, args.snr)
    test_samples = convert_samples_to_json(test_samples)
    out_file_name = f'test_data_{",".join(map(str, args.snr))}_{args.n_samples}_{args.sample_len}.json'
    with open(out_file_name, 'wt') as out_file:
        json.dump(test_samples, out_file)
