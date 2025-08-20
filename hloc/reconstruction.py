import shutil
import argparse
import multiprocessing
from typing import Any, Dict, Optional
from pathlib import Path

import pycolmap

from . import logger
from .triangulation import (
    OutputCapture,
    import_matches,
    import_features,
    parse_option_args,
    estimation_and_geometric_verification,
)


def create_empty_db(database_path: Path) -> None:
    if database_path.exists():
        logger.warning('The database already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = pycolmap.Database()
    db.open(str(database_path))
    db.close()


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: list[str] | None = None,
    options: dict[str, Any] | None = None,
):
    logger.info('Importing images into the database...')
    if options is None:
        options = pycolmap.ImageReaderOptions()

    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')

    with pycolmap.ostream():
        pycolmap.import_images(
            str(database_path),
            str(image_dir),
            camera_mode,
            image_list or [],
            options,
        )


def get_image_ids(database_path: Path) -> dict[str, int]:
    db = pycolmap.Database()
    db.open(str(database_path))
    images = {}
    for image in db.read_all_images():
        images[image.name] = image.image_id
    db.close()

    return images


def run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    models_path = sfm_dir / 'models'
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info('Running 3D reconstruction...')
    if options is None:
        options = {}
    options = {'num_threads': min(multiprocessing.cpu_count(), 16), **options}
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                str(database_path),
                str(image_dir),
                str(models_path),
                options=options,
            )

    if len(reconstructions) == 0:
        logger.error('Could not reconstruct any model!')
        return None
    logger.info(f'Reconstructed {len(reconstructions)} model(s).')

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(f'Largest model is #{largest_index} with {largest_num_images} images.')

    for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: float = 0.0,
    image_list: list[str] | None = None,
    image_options: dict[str, Any] | None = None,
    mapper_options: dict[str, Any] | None = None,
    skip_reconstruction: bool = False,
) -> pycolmap.Reconstruction | None:
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, image_list, image_options)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)

    if skip_reconstruction:
        logger.info('Skipping reconstruction as requested.')
        return None

    recon = run_reconstruction(sfm_dir, database, image_dir, verbose, mapper_options)
    if recon is not None:
        logger.info(f'Reconstruction statistics:\n{recon.summary()}' + f'\n\tnum_input_images = {len(image_ids)}')

    return recon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument(
        '--camera_mode',
        type=str,
        default='AUTO',
        choices=list(pycolmap.CameraMode.__members__.keys()),
    )
    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument(
        '--image_options',
        nargs='+',
        default=[],
        help='List of key=value from {}'.format(pycolmap.ImageReaderOptions().todict()),
    )
    parser.add_argument(
        '--mapper_options',
        nargs='+',
        default=[],
        help='List of key=value from {}'.format(pycolmap.IncrementalMapperOptions().todict()),
    )
    args = parser.parse_args().__dict__

    image_options = parse_option_args(args.pop('image_options'), pycolmap.ImageReaderOptions())
    mapper_options = parse_option_args(args.pop('mapper_options'), pycolmap.IncrementalMapperOptions())

    main(**args, image_options=image_options, mapper_options=mapper_options)
