import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio
import numpy as np
import requests
from skimage.transform import resize


def main(input_dir: Path, output_dir: Path, stylegan_dir: Path, twingan_dir: Path, twingan_model_dir: Path):
    with tempfile.TemporaryDirectory(prefix='animefy-') as work_dir:
        work_dir = Path(work_dir)
        assert work_dir.exists() and work_dir.is_dir()

        aligned_images_dir = work_dir / 'aligned'

        subprocess.check_call([
            'python3',
            'align_images.py',
            '--output_size', '256',
            str(input_dir),
            str(aligned_images_dir),
        ], cwd=str(stylegan_dir))

        anime_face_dir = work_dir / 'anime_face'

        subprocess.check_call([
            'python3',
            'inference/image_translation_infer.py',
            '--model_path', str(twingan_model_dir),
            '--image_hw', '256',
            '--input_tensor_name', 'sources_ph',
            '--output_tensor_name', 'custom_generated_t_style_source:0',
            '--input_image_path', str(aligned_images_dir),
            '--output_image_path', str(anime_face_dir),
        ], cwd=str(twingan_dir))

        downscaled_dir = work_dir / 'downscaled'
        downscaled_dir.mkdir(exist_ok=True, parents=True)

        for anime_path in anime_face_dir.iterdir():
            anime = imageio.imread(anime_path)
            anime_downscaled = (resize(anime, (128, 128, 3)) * 255).astype(np.uint8)
            imageio.imwrite(downscaled_dir / anime_path.name, anime_downscaled)

        headers = {
            'api-key': '<USE YOUR KEY HERE>',
        }

        upscaled_dir = work_dir / 'upscaled'
        upscaled_dir.mkdir(exist_ok=True, parents=True)

        for anime_downscaled_path in downscaled_dir.iterdir():
            with anime_downscaled_path.open('rb') as fp:
                files = {
                    'image': (anime_downscaled_path.name, fp),
                }

                with requests.post('https://api.deepai.org/api/waifu2x', headers=headers, files=files) as response:
                    j = response.json()
                    output_url = j['output_url']

                    with requests.get(output_url) as r:
                        with (upscaled_dir / anime_downscaled_path.name).open('wb') as fpp:
                            fpp.write(r.content)

        faces_files = sorted(aligned_images_dir.iterdir())
        anime_files = sorted(upscaled_dir.iterdir())

        for (face_path, anime_path) in zip(faces_files, anime_files):
            face = imageio.imread(face_path)
            anime = imageio.imread(anime_path)

            if face.shape[-1] == 4:
                face = face[:, :, :3]
            if anime.shape[-1] == 4:
                anime = anime[:, :, :3]

            assert face_path.name == anime_path.name
            assert face.shape == (256, 256, 3), face.shape
            assert anime.shape == (256, 256, 3), anime.shape

            output_path = output_dir / face_path.name
            output = np.hstack([face, anime])
            imageio.imwrite(output_path, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put anime into your photos')
    parser.add_argument('input_dir', help='Directory with photos', type=Path)
    parser.add_argument('output_dir', help='Where to put glorious anime', type=Path)
    parser.add_argument('stylegan_dir', type=Path)
    parser.add_argument('twingan_dir', type=Path)
    parser.add_argument('twingan_model_dir', type=Path)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.stylegan_dir, args.twingan_dir, args.twingan_model_dir)
