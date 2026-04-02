import argparse
from enum import StrEnum
from typing import Any
from pathlib import Path

import numpy as np
import poselib
import pycolmap
from tqdm import tqdm
from joblib import Parallel, delayed

from . import logger
from .utils.io import get_matches, get_keypoints
from .utils.parsers import parse_pairs, parse_retrieval
from .utils.geometry import compute_epipolar_errors


class PoselibMethod(StrEnum):
    RELATIVE_POSE = "relative_pose"
    SHARED_FOCAL = "shared_focal"
    FUNDAMENTAL = "fundamental"


class OutputCapture:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            pycolmap.logging.alsologtostderr = False

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            pycolmap.logging.alsologtostderr = True


def create_db_from_model(
    reconstruction: pycolmap.Reconstruction, database_path: str | Path
) -> dict[str, int]:
    database_path = Path(database_path)
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()

    with pycolmap.Database.open(str(database_path)) as db:
        for camera_id, camera in reconstruction.cameras.items():
            db.write_camera(camera, use_camera_id=True)
        for rig_id, rig in reconstruction.rigs.items():
            db.write_rig(rig, use_rig_id=True)
        for frame_id, frame in reconstruction.frames.items():
            db.write_frame(frame, use_frame_id=True)
        for image_id, image in reconstruction.images.items():
            db.write_image(image, use_image_id=True)

    return {image.name: image_id for image_id, image in reconstruction.images.items()}


def import_features(
    image_ids: dict[str, int],
    db: pycolmap.Database,
    features_path: Path,
):
    logger.info(f"Import features from {features_path}")
    for image_name, image_id in tqdm(image_ids.items(), desc="Importing features"):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin
        db.write_keypoints(image_id, keypoints)


def import_matches(
    image_ids: dict[str, int],
    db: pycolmap.Database,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: float = 0.0,
    skip_geometric_verification: bool = False,
):
    logger.info(f"Import matches from {matches_path}")
    pairs = parse_pairs(pairs_path)

    matched: set[tuple[int, int]] = set()
    for name0, name1 in tqdm(pairs, desc="Importing matches"):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue

        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]

        db.write_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.write_two_view_geometry(
                id0, id1, pycolmap.TwoViewGeometry(inlier_matches=matches)
            )


def _camera_model_name(camera) -> str:
    if hasattr(camera, "model_name"):
        name = camera.model_name
        return name() if callable(name) else str(name)

    if hasattr(camera, "model"):
        return str(camera.model).split(".")[-1]

    raise AttributeError("Could not infer camera model name from pycolmap camera.")


def _camera_to_poselib(camera) -> dict[str, Any]:
    return {
        "model": _camera_model_name(camera),
        "width": int(camera.width),
        "height": int(camera.height),
        "params": [float(x) for x in camera.params],
    }


class TVGResult:
    def __init__(
        self,
        id0: int,
        id1: int,
        inlier_ratio: float | None,
        tvg: pycolmap.TwoViewGeometry | None = None,
    ):
        self.id0 = id0
        self.id1 = id1
        self.inlier_ratio = inlier_ratio

        if tvg is not None:
            self.tvg = tvg
        else:
            tvg = pycolmap.TwoViewGeometry()
            tvg.inlier_matches = np.empty((0, 2), dtype=np.int32)
            self.tvg = tvg


def _estimate_one_pair(
    name0: str,
    name1: str,
    id0: int,
    id1: int,
    images: dict[str, Any],
    cameras: dict[int, Any],
    features_path: Path,
    matches_path: Path,
    method: Any,
    ransac_options: dict[str, Any],
    bundle_options: dict[str, Any],
    min_inlier_ratio: float = 0.0,
):
    kps0 = get_keypoints(features_path, name0).astype(np.float64)
    kps0 += 0.5
    kps1 = get_keypoints(features_path, name1).astype(np.float64)
    kps1 += 0.5

    matches, _ = get_matches(matches_path, name0, name1)

    if matches.shape[0] == 0:
        return TVGResult(
            id0=id0,
            id1=id1,
            inlier_ratio=None,
            tvg=None,
        )

    pts0 = kps0[matches[:, 0]]
    pts1 = kps1[matches[:, 1]]

    tvg = pycolmap.TwoViewGeometry()
    try:
        if method == PoselibMethod.RELATIVE_POSE:
            cam0 = _camera_to_poselib(cameras[images[name0].camera_id])
            cam1 = _camera_to_poselib(cameras[images[name1].camera_id])
            pose, info = poselib.estimate_relative_pose(
                pts0, pts1, cam0, cam1, ransac_options, bundle_options
            )

            cam2_from_cam1 = pycolmap.Rigid3d(rotation=pose.R, translation=pose.t)

            tvg.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
            tvg.cam2_from_cam1 = cam2_from_cam1
            tvg.E = pycolmap.essential_matrix_from_pose(cam2_from_cam1)

        elif method == PoselibMethod.SHARED_FOCAL:
            cam0 = cameras[images[name0].camera_id]
            cam1 = cameras[images[name1].camera_id]
            assert isinstance(cam0, pycolmap.Camera)
            assert isinstance(cam1, pycolmap.Camera)
            assert cam0.width == cam1.width and cam0.height == cam1.height

            pp = [0.5 * cam0.width, 0.5 * cam0.height]
            _, info = poselib.estimate_shared_focal_relative_pose(
                pts0, pts1, pp, ransac_options, bundle_options
            )

        elif method == PoselibMethod.FUNDAMENTAL:
            F, info = poselib.estimate_fundamental(
                pts0, pts1, ransac_options, bundle_options
            )

            tvg.config = pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED
            tvg.F = F

        else:
            raise ValueError(f"Unknown method: {method}")

        inliers = np.asarray(info["inliers"], dtype=bool).reshape(-1)
        inlier_matches = matches[inliers]
        inlier_ratio = float(len(inlier_matches)) / float(len(matches))

        if inlier_ratio < min_inlier_ratio:
            inlier_ratio = None
            inlier_matches = np.empty((0, 2), dtype=np.int32)

        tvg.inlier_matches = inlier_matches

        return TVGResult(
            id0=id0,
            id1=id1,
            inlier_ratio=inlier_ratio,
            tvg=tvg,
        )

    except Exception:
        pass

    return TVGResult(
        id0=id0,
        id1=id1,
        inlier_ratio=None,
        tvg=None,
    )


def estimation_and_geometric_verification_poselib(
    database_path: Path,
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    verbose: bool = False,
    method: PoselibMethod = PoselibMethod.RELATIVE_POSE,
    ransac_options: dict[str, Any] | None = None,
    bundle_options: dict[str, Any] | None = None,
):
    logger.info("Performing geometric verification (poselib)")

    if ransac_options is None:
        ransac_options = {
            "max_epipolar_error": 4.0,
            "max_iterations": 20000,
        }
    if bundle_options is None:
        bundle_options = {}

    pairs = parse_retrieval(pairs_path)

    with pycolmap.Database.open(str(database_path)) as db:
        images = {img.name: img for img in db.read_all_images()}
        cameras = {cam.camera_id: cam for cam in db.read_all_cameras()}

    # Unique pairs
    tasks = {}
    for name0 in pairs:
        for name1 in pairs[name0]:
            id0 = images[name0].image_id
            id1 = images[name1].image_id

            key = (min(id0, id1), max(id0, id1))
            if key not in tasks:
                tasks[key] = (name0, name1)

    results = Parallel(n_jobs=-1, backend="threading", verbose=0)(
        delayed(_estimate_one_pair)(
            name0,
            name1,
            id0,
            id1,
            images,
            cameras,
            features_path,
            matches_path,
            method,
            ransac_options,
            bundle_options,
            min_inlier_ratio=0.1,
        )
        for (id0, id1), (name0, name1) in tqdm(
            tasks.items(), desc="Estimating two-view geometries with PoseLib"
        )
    )

    inlier_ratios = []
    with pycolmap.Database.open(str(database_path)) as db:
        for res in results:
            assert isinstance(res, TVGResult)

            db.write_two_view_geometry(
                res.id0,
                res.id1,
                res.tvg,
            )
            if res.inlier_ratio is not None:
                inlier_ratios.append(res.inlier_ratio)

    if len(inlier_ratios) > 0:
        v_avg = 100.0 * float(np.mean(inlier_ratios))
        v_med = 100.0 * float(np.median(inlier_ratios))
        v_min = 100.0 * float(np.min(inlier_ratios))
        v_max = 100.0 * float(np.max(inlier_ratios))
        logger.info(
            "Poselib inlier ratio mean/med/min/max: "
            f"{v_avg:.2f} / {v_med:.2f} / {v_min:.2f} / {v_max:.2f} %",
        )


def estimation_and_geometric_verification(
    database_path: Path, pairs_path: Path, verbose: bool = False
):
    logger.info("Performing geometric verification of the matches...")
    options = pycolmap.TwoViewGeometryOptions(
        {
            "ransac": pycolmap.RANSACOptions(
                {
                    "max_num_trials": 20000,
                    "min_inlier_ratio": 0.1,
                }
            )
        }
    )

    if pairs_path.suffix == ".json":
        pairs_txt = pairs_path.with_suffix(".txt")
        if not pairs_txt.exists():
            pairs = parse_pairs(pairs_path)
            with open(pairs_txt, mode="w") as f:
                for name0, name1 in pairs:
                    f.write(f"{name0} {name1}\n")

        pairs_path = pairs_txt

    with OutputCapture(verbose):
        pycolmap.verify_matches(
            str(database_path),
            str(pairs_path),
            options=options,
        )


def geometric_verification(
    image_ids: dict[str, int],
    reference: pycolmap.Reconstruction,
    db: pycolmap.Database,
    features_path: Path,
    pairs_path: Path,
    matches_path: Path,
    max_error: float = 4.0,
):
    logger.info("Performing geometric verification of the matches...")

    pairs = parse_retrieval(pairs_path)
    inlier_ratios = []
    matched: set[tuple[int, int]] = set()
    for name0 in tqdm(pairs):
        id0 = image_ids[name0]
        image0 = reference.images[id0]
        cam0 = reference.cameras[image0.camera_id]
        kps0, noise0 = get_keypoints(features_path, name0, return_uncertainty=True)
        noise0 = 1.0 if noise0 is None else noise0
        if len(kps0) > 0:
            kps0 = np.stack(cam0.cam_from_img(kps0))
        else:
            kps0 = np.zeros((0, 2))

        for name1 in pairs[name0]:
            id1 = image_ids[name1]
            image1 = reference.images[id1]
            cam1 = reference.cameras[image1.camera_id]
            kps1, noise1 = get_keypoints(features_path, name1, return_uncertainty=True)
            noise1 = 1.0 if noise1 is None else noise1
            if len(kps1) > 0:
                kps1 = np.stack(cam1.cam_from_img(kps1))
            else:
                kps1 = np.zeros((0, 2))

            matches = get_matches(matches_path, name0, name1)[0]

            if (id0, id1) in matched or (id1, id0) in matched:
                continue

            matched = matched.union({(id0, id1), (id1, id0)})

            if matches.shape[0] == 0:
                db.write_two_view_geometry(id0, id1, pycolmap.TwoViewGeometry())
                continue

            cam1_from_cam0 = image1.cam_from_world() * image0.cam_from_world().inverse()
            errors0, errors1 = compute_epipolar_errors(
                cam1_from_cam0, kps0[matches[:, 0]], kps1[matches[:, 1]]
            )
            valid_matches = np.logical_and(
                errors0 <= cam0.cam_from_img_threshold(noise0 * max_error),
                errors1 <= cam1.cam_from_img_threshold(noise1 * max_error),
            )

            # TODO: We could also add E to the database, but we need
            # to reverse the transformations if id0 > id1 in utils/database.py.
            two_view_geo = pycolmap.TwoViewGeometry()
            two_view_geo.inlier_matches = matches[valid_matches, :]
            db.write_two_view_geometry(id0, id1, two_view_geo)
            inlier_ratios.append(np.mean(valid_matches))

    v_avg = 100.0 * float(np.mean(inlier_ratios))
    v_med = 100.0 * float(np.median(inlier_ratios))
    v_min = 100.0 * float(np.min(inlier_ratios))
    v_max = 100.0 * float(np.max(inlier_ratios))
    logger.info(
        "Inlier ratio mean/med/min/max: "
        f"{v_avg:.2f}/{v_med:.2f}/{v_min:.2f}/{v_max:.2f} %.",
    )


def run_triangulation(
    model_path: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    verbose: bool = False,
    options: dict[str, Any] | None = None,
) -> pycolmap.Reconstruction:
    model_path.mkdir(parents=True, exist_ok=True)
    logger.info("Running 3D triangulation...")
    if options is None:
        options = {}

    with OutputCapture(verbose):
        reconstruction = pycolmap.triangulate_points(
            reference_model, database_path, image_dir, model_path, options=options
        )

    return reconstruction


def main(
    sfm_dir: Path,
    reference_model: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    skip_geometric_verification: bool = False,
    estimate_two_view_geometries: bool = False,
    min_match_score: float | None = None,
    verbose: bool = False,
    mapper_options: dict[str, Any] | None = None,
) -> pycolmap.Reconstruction:
    assert reference_model.exists(), reference_model
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    reference = pycolmap.Reconstruction(str(reference_model))

    image_ids = create_db_from_model(reference, database)
    with pycolmap.Database.open(str(database)) as db:
        import_features(image_ids, db, features)
        import_matches(
            image_ids,
            db,
            pairs,
            matches,
            min_match_score,
            skip_geometric_verification,
        )
    if not skip_geometric_verification:
        if estimate_two_view_geometries:
            estimation_and_geometric_verification(database, pairs, verbose)
        else:
            with pycolmap.Database.open(str(database)) as db:
                geometric_verification(
                    image_ids, reference, db, features, pairs, matches
                )
    reconstruction = run_triangulation(
        sfm_dir, database, image_dir, reference, verbose, mapper_options
    )
    logger.info(
        "Finished the triangulation with statistics:\n%s", reconstruction.summary()
    )
    return reconstruction


def parse_option_args(args: list[str], default_options) -> dict[str, Any]:
    options = {}
    for arg in args:
        idx = arg.find("=")
        if idx == -1:
            raise ValueError("Options format: key1=value1 key2=value2 etc.")

        key, value = arg[:idx], arg[idx + 1 :]
        if not hasattr(default_options, key):
            raise ValueError(
                f'Unknown option "{key}", allowed options and default values for {default_options.summary()}'
            )

        value = eval(value)
        target_type = type(getattr(default_options, key))
        if not isinstance(value, target_type):
            raise ValueError(
                f'Incorrect type for option "{key}": {type(value)} vs {target_type}'
            )

        options[key] = value

    return options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfm_dir", type=Path, required=True)
    parser.add_argument("--reference_sfm_model", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)

    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)

    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--min_match_score", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args().__dict__

    mapper_options = parse_option_args(
        args.pop("mapper_options"), pycolmap.IncrementalMapperOptions()
    )

    main(**args, mapper_options=mapper_options)
