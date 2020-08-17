import json
import argparse
from pathlib import Path

from demodulation.data_preprocessing import generate_samples_per_snr, convert_samples_to_json
import modulation


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--n_samples', type=int, default=5, help='Number of samples per SNR value')
    arg_parser.add_argument('--sample_len', type=int, default=50, help='Length of each sample')
    arg_parser.add_argument('--snr', type=int, nargs='+', default=[-2, -4, -6, -8, -10, -12],
                            help='Sound-noise ratios to generate samples for')
    arg_parser.add_argument('--shift', type=float, default=None,
                            help='Value that is added to all channels frequencies, as share of the channel width')
    arg_parser.add_argument('--prefix', default='test', help='Prefix of the generated file name')
    arg_parser.add_argument('--dir', default='data', help='Directory to place the generated file')
    args = arg_parser.parse_args()

    modulator = modulation.Modulator(frequency_shift=args.shift)
    test_samples = generate_samples_per_snr(args.n_samples, args.sample_len, modulator, args.snr)
    test_samples = convert_samples_to_json(test_samples)
    shift_part = f'sh{int(args.shift * 100)}_' if args.shift else ""
    ratios_part = ",".join(map(str, args.snr))
    out_file_name = f'{args.prefix}_{ratios_part}_{shift_part}{args.n_samples}_{args.sample_len}.json'
    out_dir_path = Path(args.dir)
    out_file_path = out_dir_path / out_file_name
    with out_file_path.open('wt') as out_file:
        json.dump(test_samples, out_file)
