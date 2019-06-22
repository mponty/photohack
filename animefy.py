import argparse
import subprocess
import tempfile
from pathlib import Path


def main(input_dir: Path, output_dir: Path, stylegan_dir: Path, twingan_dir: Path, twingan_model_dir: Path):
    with tempfile.TemporaryDirectory(prefix='animefy-') as work_dir:
        work_dir = Path(work_dir)
        assert work_dir.exists() and work_dir.is_dir()

        aligned_images_dir = work_dir / 'aligned'

        subprocess.check_call([
            'python',
            'align_images.py',
            '--output_size', '256',
            str(input_dir),
            str(aligned_images_dir),
        ], cwd=stylegan_dir)

        subprocess.check_call([
            'python',
            'inference/image_translation_infer.py',
            '--model_path', str(twingan_model_dir),
            '--image_hw', '256',
            '--input_tensor_name', 'sources_ph',
            '--output_tensor_name', 'custom_generated_t_style_source:0',
            '--input_image_path', str(aligned_images_dir),
            '--output_image_path', str(output_dir),
        ], cwd=twingan_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put anime into your photos')
    parser.add_argument('input_dir', help='Directory with photos', type=Path)
    parser.add_argument('output_dir', help='Where to put glorious anime', type=Path)
    parser.add_argument('stylegan_dir', type=Path)
    parser.add_argument('twingan_dir', type=Path)
    parser.add_argument('twingan_model_dir', type=Path)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.stylegan_dir, args.twingan_dir, args.twingan_model_dir)
