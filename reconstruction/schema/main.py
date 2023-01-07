from __future__ import annotations

import os
import time
import warnings
from curses import keyname
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader

Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

import datajoint as dj
from mei import mixins
from mei.import_helpers import import_object
from nnfabrik.builder import get_model, resolve_fn
from nnfabrik.main import Dataset
from nnfabrik.utility.dj_helpers import CustomSchema, cleanup_numpy_scalar, make_hash
from reconstructing_robustness.data.datasets import ImageNet, ImageNetC, dataset_names
from reconstructing_robustness.dataset import get_transforms
from reconstructing_robustness.dj_tables.nnfabrik import Model, TrainedModel
from reconstructing_robustness.model import model_fn
from reconstructing_robustness.utils.constants import FETCH_DIR
from reconstructing_robustness.utils.reconstruction_utils import (
    get_class_by_img_path,
    rescale_mei_in_zspace,
)

from ..modules.reducers import ConstrainedOutputModel

# from bias_transfer.tables.trained_model import *
# from bias_transfer.tables.trained_transfer_model import *


schema = CustomSchema(dj.config.get("reconstruction_schema", "nnfabrik_core"))
resolve_target_fn = partial(resolve_fn, default_base="targets")
resolve_scaler_fn = partial(resolve_fn, default_base="scalers")


@schema
class ReconSeed(mixins.MEISeedMixin, dj.Lookup):
    """Seed table for MEI method."""


@schema
class ReconstructionImages(dj.Manual):
    definition = """
      # images for reconstruction
      img_id:    int                  # unique id
      img_path: varchar(128)          # image path
      dataset: varchar(64)            # dataset namec
      corruption: varchar(64)         # corruption subcategory
      severity: int                   # corruption severtiy [0-5]
      img_hash: varchar(64)           # image hash
      ---
      image: longblob                 # actual image as numpy array
      norm: float                     # norm of the image
      img_comment: varchar(64) 
      """

    def add_entry(
        self,
        img_id: int,
        img_path: str,
        dataset,
        corruption: str,
        severity: int,
        img_comment: str,
        skip_duplicates=False,
    ):
        key = {}

        key["img_id"] = img_id
        key["img_comment"] = img_comment

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} does not exist.")
        else:
            key["img_path"] = img_path

        if not dataset.name in dataset_names:
            raise ValueError(f"dataset must be one of {dataset_names}.")
        else:
            key["dataset"] = dataset.name

        c_subcategories = ImageNetC.subcategories
        # add empty string to represent clean ImageNet validation set
        c_subcategories += [""]
        if not corruption in c_subcategories:
            raise ValueError(f"corruption must be one of {c_subcategories}.")
        elif corruption == "" and dataset.name == ImageNetC.name:
            raise ValueError(f"corruption does not match dataset {ImageNetC.name}.")
        elif corruption != "" and dataset.name == ImageNet.name:
            raise ValueError(f"corruption does not match {ImageNet.name}.")
        else:
            key["corruption"] = corruption

        # severtiy zero represents clean ImageNet validation set
        valid_severities = [i for i in range(6)]
        if not severity in valid_severities:
            raise ValueError(f"severity must be one of {valid_severities}.")
        elif severity == 0 and dataset.name == ImageNetC.name:
            raise ValueError(f"severity 0 does not match dataset {ImageNetC.name}.")
        elif severity > 0 and dataset.name == ImageNet.name:
            raise ValueError(f"severity > 0 does not match dataset {ImageNet.name}.")
        else:
            key["severity"] = severity

        key["img_hash"] = make_hash(
            [img_id, img_path, dataset.name, corruption, severity]
        )

        # add image
        image = Image.open(img_path).convert("RGB")
        t = get_transforms(dataset)  # transform image according to dataset
        image = t(image).numpy()
        image = image[None, :]
        key["image"] = image

        # add nomrm
        key["norm"] = np.linalg.norm(image)

        # insert key
        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key


@schema
class ReconMethod(mixins.MEIMethodMixin, dj.Lookup):
    seed_table = ReconSeed
    optional_names = (
        "initial",
        "optimizer",
        "transform",
        "regularization",
        "precondition",
        "postprocessing",
    )

    def generate_mei(
        self, dataloaders: Dataloaders, model: Module, key: Key, seed: int
    ) -> Dict[str, Any]:
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        self.insert_key_in_ops(method_config=method_config, key=key)
        method_fn = self.import_func(method_fn)
        mei, score, output = method_fn(dataloaders, model, method_config, seed)
        return dict(key, mei=mei, score=score, output=output)

    def insert_key_in_ops(self, method_config, key):
        for k, v in method_config.items():
            if k in self.optional_names:
                if "kwargs" in v:
                    if "key" in v["kwargs"]:
                        v["kwargs"]["key"] = key


@schema
class ReconMethodParameters(dj.Computed):
    definition = """
    -> ReconMethod
    ---
    lr: float   # list all the parameters here that are of interest
    norm_fraction: float    # fraction of full image norm
    n_iter: float
    initial: varchar(64)
    optimizer: varchar(64)
    precondition = "": varchar(64)
    postprocessing = "": varchar(64)
    """

    def make(self, key):

        # get the config
        method_config = (ReconMethod & key).fetch1("method_config")

        # get all the attributes
        key["norm_fraction"] = (
            method_config.get("postprocessing").get("kwargs").get("norm_fraction")
        )
        key["initial"] = method_config.get("initial").get("path")
        key["optimizer"] = method_config.get("optimizer").get("path")
        key["lr"] = method_config.get("optimizer").get("kwargs").get("lr")
        key["n_iter"] = method_config.get("stopper").get("kwargs").get("num_iterations")
        if "precondition" in method_config:
            key["precondition"] = method_config.get("precondition").get("path")
        if "postprocessing" in method_config:
            key["postprocessing"] = method_config.get("postprocessing").get("path")

        self.insert1(key, ignore_extra_fields=True)


@schema
class ReconTargetFunction(dj.Manual):
    definition = """
    target_fn:       varchar(64)
    target_hash:     varchar(64)
    ---
    target_config:   longblob
    target_comment:  varchar(128)
    """

    resolve_fn = resolve_target_fn

    @property
    def fn_config(self):
        target_fn, target_config = self.fetch1("target_fn", "target_config")
        target_config = cleanup_numpy_scalar(target_config)
        return target_fn, target_config

    def add_entry(
        self, target_fn, target_config, target_comment="", skip_duplicates=False
    ):
        """
        Add a new entry to the TargetFunction table.

        Args:
            target_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `targets` subpackage.
            target_config (dict) - Python dictionary containing keyword arguments for the target_fn
            dataset_comment - Optional comment for the entry.
            target_comment - If True, no error is thrown when a duplicate entry (i.e. entry with same target_fn and target_config) is found.

        Returns:
            key - key in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """

        try:
            resolve_target_fn(target_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + "\nTable entry rejected")
            return

        target_hash = make_hash(target_config)
        key = dict(
            target_fn=target_fn,
            target_hash=target_hash,
            target_config=target_config,
            target_comment=target_comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key

    def get_target_fn(self, key=None, **kwargs):
        if key is None:
            key = self.fetch("KEY")
        target_fn, target_config = (self & key).fn_config
        return partial(self.resolve_fn(target_fn), **target_config, **kwargs)


@schema
class ReconTargetUnit(dj.Manual):
    definition = """
    -> Model
    unit_fn:                        varchar(128)
    unit_hash:                      varchar(128)
    ---
    unit_config:                   longblob       # list of unit_ids 
    unit_comment:                   varchar(128)
    """
    dataset_table = Dataset

    def add_entry(
        self,
        model_fn,
        model_hash,
        unit_fn,
        unit_config,
        unit_comment="",
        skip_duplicates=False,
    ):
        """
        Add a new entry to the TargetFunction table.

        Args:
            target_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `targets` subpackage.
            target_config (dict) - Python dictionary containing keyword arguments for the target_fn
            dataset_comment - Optional comment for the entry.
            target_comment - If True, no error is thrown when a duplicate entry (i.e. entry with same target_fn and target_config) is found.

        Returns:
            key - key in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """

        unit_hash = make_hash(unit_config)
        key = dict(
            model_fn=model_fn,
            model_hash=model_hash,
            unit_hash=unit_hash,
            unit_fn=unit_fn,
            unit_config=unit_config,
            unit_comment=unit_comment,
        )
        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key

    def get_wrapper(self, key, model):
        unit_fn, unit_config = (self & key).fetch1("unit_fn", "unit_config")
        object_kwargs = dict(model=model, **unit_config)
        wrapper = import_object(path=unit_fn, object_kwargs=object_kwargs)
        return wrapper


@schema
class ReconTargetUnitParameters(dj.Computed):
    definition = """
    -> ReconTargetUnit
    ---
    return_layer: varchar(64)
    """

    def make(self, key):
        """
        Add entries

        Parameters
        ----------
        key : _type_
            _description_
        """

        unit_config = (ReconTargetUnit & key).fetch1("unit_config")

        ((layer_name, new_layer_name),) = unit_config["return_layers"].items()

        key["return_layer"] = layer_name

        self.insert1(key)


@schema
class ReconObjective(dj.Computed):
    target_fn_table = ReconTargetFunction
    target_unit_table = ReconTargetUnit
    constrained_output_model = ConstrainedOutputModel

    @property
    def definition(self):
        definition = """
        -> self.target_fn_table 
        -> self.target_unit_table
        ---
        objective_comment:  varchar(128)
        """
        return definition

    def make(self, key):
        comments = []
        comments.append((self.target_fn_table & key).fetch1("target_comment"))
        comments.append((self.target_unit_table & key).fetch1("unit_comment"))

        key["objective_comment"] = ", ".join(comments)
        self.insert1(key)

    def get_output_selected_model(
        self,
        model: Module,
        target_fn: Callable,
    ) -> constrained_output_model:

        return self.constrained_output_model(
            model=model,
            target_fn=target_fn,
        )


@schema
class Reconstruction(mixins.MEITemplateMixin, dj.Computed):
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    -> self.image_table
    -> self.seed_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    score               : float         # some score depending on the used method function
    output              : attach@minio  # object returned by the method function
    """

    trained_model_table = TrainedModel
    selector_table = ReconObjective
    target_fn_table = ReconTargetFunction
    target_unit_table = ReconTargetUnit
    method_table = ReconMethod
    image_table = ReconstructionImages
    seed_table = ReconSeed
    storage = "minio"
    database = ""  # hack to supress DJ error

    class Responses(dj.Part):
        @property
        def definition(self):
            definition = """
            # Contains the models state dict, stored externally.
            -> master
            ---
            original_responses:                 attach@{storage}
            reconstructed_responses:            attach@{storage}
            """.format(
                storage=self._master.storage
            )
            return definition

    def get_model_responses(self, model, image, device="cuda"):
        with torch.no_grad():
            responses = model(
                image.to(device),
            )
        return responses

    def _insert_responses(self, response_entity: Dict[str, Any]) -> None:
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        with self.get_temp_dir() as temp_dir:
            for name in ("original_responses", "reconstructed_responses"):
                self._save_to_disk(response_entity, temp_dir, name)
            self.Responses.insert1(response_entity, ignore_extra_fields=True)

    def make(self, key):

        start = time.time()

        dataloaders, model = self.model_loader.load(key=key)
        model.eval().cuda()
        seed = (self.seed_table() & key).fetch1("mei_seed")
        image = torch.from_numpy((self.image_table & key).fetch1("image").copy())
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        wrapper = self.target_unit_table().get_wrapper(key, model)
        responses = self.get_model_responses(wrapper, image)
        target_fn = (self.target_fn_table & key).get_target_fn(responses=responses)
        output_selected_model = self.selector_table().get_output_selected_model(
            model=wrapper, target_fn=target_fn
        )
        # ping DB to prevent LostConnectionError
        self.connection.ping()

        print("Starting to generate MEI...")  # docker logs
        mei_entity = self.method_table().generate_mei(
            dataloaders, output_selected_model, key, seed
        )
        print("score = ", mei_entity["score"])  # docker logs
        reconstructed_image = mei_entity["mei"]
        print("Getting model activations in response to MEI...")  # docker logs
        reconstructed_responses = self.get_model_responses(
            model=wrapper,
            image=reconstructed_image,
        )
        response_entity = dict(
            original_responses=responses,
            reconstructed_responses=reconstructed_responses,
        )
        print("Inserting MEI...")  # docker logs
        self._insert_mei(mei_entity)
        mei_entity.update(response_entity)
        self._insert_responses(mei_entity)

        end = time.time()
        elapsed = time.gmtime(end - start)
        print(f"reconstruction took {elapsed.tm_min} min {elapsed.tm_sec} sec")


@schema
class ReconScaler(dj.Manual):
    definition = """
    scaler_fn:                        varchar(128)
    scaler_hash:                      varchar(64)
    ---
    scaler_config:                    longblob
    scaler_comment:                   varchar(128)
    """

    resolve_fn = resolve_scaler_fn

    @property
    def fn_config(self):
        scaler_fn, scaler_config = self.fetch1("scaler_fn", "scaler_config")
        scaler_config = cleanup_numpy_scalar(scaler_config)
        return scaler_fn, scaler_config

    def add_entry(
        self,
        scaler_fn: str,
        scaler_config: dict,
        scaler_comment: str = "",
        skip_duplicates=False,
    ):

        scaler_hash = make_hash((scaler_fn, scaler_config))
        key = dict(
            scaler_fn=scaler_fn,
            scaler_config=scaler_config,
            scaler_hash=scaler_hash,
            scaler_comment=scaler_comment,
        )
        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        return key

    def get_scaler_fn(self, key=None, **kwargs):
        key = self.fetch("KEY") if key is None else key
        scaler_fn, scaler_config = (self & key).fn_config
        return partial(self.resolve_fn(scaler_fn), **scaler_config, **kwargs)


@schema
class ReconstructionClassification(dj.Computed):
    definition = """
        # Contains classification results of re-scaled reconstructions
        -> self.recon_table
        -> self.model_table.proj(evaluator_model_fn="model_fn", evaluator_model_hash="model_hash")
        -> self.scaler_fn_table.proj()
        ---
        norm_scaled: float # norm of the re-scaled reconstruction in z-score space
        classification: int # the predicted class
        correct: tinyint # 0/1 if it's the correct label
        logits: longblob
        """

    model_table = Model
    recon_table = Reconstruction
    image_table = ReconstructionImages
    method_params_table = ReconMethodParameters
    scaler_fn_table = ReconScaler

    def make(self, key):

        start = time.time()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mei_path, norm_fraction, norm_orig = (
            self.recon_table * self.image_table * self.method_params_table & key
        ).fetch1("mei", "norm_fraction", "norm", download_path=FETCH_DIR)
        true_class_idx = get_class_by_img_path(key["img_path"], return_idx_only=True)
        # load (normlized) mei as torch tensor
        assert os.path.exists(mei_path)
        print("mei path: ", mei_path)  # docker logs
        mei = torch.load(mei_path, map_location=torch.device(device))
        # if norm_fraction is not full norm, scale reconstruction
        scaler_fn = self.scaler_fn_table().get_scaler_fn(key=key, norm_orig=norm_orig)
        scaler_comment = (self.scaler_fn_table & key).fetch1("scaler_comment")
        if norm_fraction < 1.0:
            print("Scaling reconstruction using: " + scaler_comment)  # docker logs
            mei = scaler_fn(mei)
        # get model and evaluate
        print("Getting evaluator model...")  # docker logs
        eval_model_restr = dict(model_hash=key["evaluator_model_hash"])
        model_fn, model_config = (self.model_table & eval_model_restr).fn_config
        model = get_model(
            model_fn=model_fn,
            model_config=model_config,
            dataloaders={},
            seed=key["seed"],
        )
        # get model prediction
        model, mei = model.to(device), mei.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(mei)
            predicted_class = logits.argmax()
        # add table entry
        key["norm_scaled"] = torch.linalg.norm(mei).item()
        key["classification"] = predicted_class.item()
        key["correct"] = 1 if predicted_class.item() == true_class_idx else 0
        key["logits"] = logits.detach().cpu().numpy()
        self.insert1(key)

        end = time.time()
        elapsed = time.gmtime(end - start)
        print(f"Classification took {elapsed.tm_min} min {elapsed.tm_sec} sec")


@schema
class ReconClassification(dj.Computed):
    definition = """
        ->Reconstruction
        ->Model.proj(evaluator_model_fn='model_fn', evaluator_model_hash='model_hash')
        ---
        classification: int # the predicted class
        correct: tinyint # 0/1 if it's the correct label
        """

    def make(self, key):
        # load reconstruction, load model, get classification

        start = time.time()

        # get true class of original image
        img_path = (Reconstruction().image_table() & key).fetch1("img_path")
        true_class_info = get_class_by_img_path(img_path)
        true_class = true_class_info[1]
        assert isinstance(true_class, int)
        # get (normlized) mei as torch tensor
        mei = (Reconstruction() & key).fetch1("mei", download_path=FETCH_DIR)
        mei = torch.load(mei)
        # get model and evaluate
        eval_model_restr = dict(model_hash=key["evaluator_model_hash"])
        model_config, model_comment = (
            Reconstruction().trained_model_table().model_table() & eval_model_restr
        ).fetch1("model_config", "model_comment")
        seed = (Reconstruction().trained_model_table() & eval_model_restr).fetch1(
            "seed"
        )
        model = model_fn({}, seed, **model_config)
        print(f"Model: {model_comment}")
        # get model prediction
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        mei = mei.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(mei)
            predicted_class = logits.argmax(dim=1, keepdim=True)
        # add table entry
        key["classification"] = predicted_class.item()
        key["correct"] = 1 if predicted_class.item() == true_class else 0
        self.insert1(key)

        end = time.time()
        elapsed = time.gmtime(end - start)
        print(f"Classification took {elapsed.tm_min} min {elapsed.tm_sec} sec")


@schema
class ReconClassificationRescaled(dj.Computed):
    definition = """
        # Contains classification results of re-scaled reconstructions/MEIs
        ->Reconstruction
        ->Model.proj(evaluator_model_fn='model_fn', evaluator_model_hash='model_hash')
        ---
        norm_rescaled: float # norm of the re-scaled MEI in z-space (normalized)
        classification: int # the predicted class
        correct: tinyint # 0/1 if it's the correct label
        """

    def make(self, key):
        # load reconstruction, load model, get classification

        start = time.time()

        # get true class of original image
        img_path, norm_fraction = (
            Reconstruction() * ReconMethodParameters() & key
        ).fetch1("img_path", "norm_fraction")
        true_class_info = get_class_by_img_path(img_path)
        true_class = true_class_info[1]
        assert isinstance(true_class, int)
        # get (normlized) mei as torch tensor
        mei = (Reconstruction() & key).fetch1("mei", download_path=FETCH_DIR)
        mei = torch.load(mei)
        # if norm_fraction is not full norm, re-scale mei
        if norm_fraction < 1.0:
            mei = rescale_mei_in_zspace(mei)
        # get model and evaluate
        eval_model_restr = dict(model_hash=key["evaluator_model_hash"])
        model_config, model_comment = (
            Reconstruction().trained_model_table().model_table() & eval_model_restr
        ).fetch1("model_config", "model_comment")
        seed = (Reconstruction().trained_model_table() & eval_model_restr).fetch1(
            "seed"
        )
        model = model_fn({}, seed, **model_config)
        print(f"Model: {model_comment}")
        # get model prediction
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        mei = mei.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(mei)
            predicted_class = logits.argmax(dim=1, keepdim=True)
        # add table entry
        key["norm_rescaled"] = np.linalg.norm(mei.detach().cpu())
        key["classification"] = predicted_class.item()
        key["correct"] = 1 if predicted_class.item() == true_class else 0
        self.insert1(key)

        end = time.time()
        elapsed = time.gmtime(end - start)
        print(f"Classification took {elapsed.tm_min} min {elapsed.tm_sec} sec")


# @schema
# class ReconstructionTransfer(mixins.MEITemplateMixin, dj.Computed):
#    definition = """
#    # contains maximally exciting images (MEIs)
#    -> self.method_table
#    -> self.trained_model_table
#    -> self.selector_table
#    -> self.image_table
#    -> self.seed_table
#    ---
#    mei                 : attach@minio  # the MEI as a tensor
#    score               : float         # some score depending on the used method function
#    output              : attach@minio  # object returned by the method function
#    """
#
#    trained_model_table = TrainedTransferModel
#    selector_table = ReconObjective
#    target_fn_table = ReconTargetFunction
#    target_unit_table = ReconTargetUnit
#    method_table = ReconMethod
#    image_table = ReconstructionImages
#    seed_table = ReconSeed
#    storage = "minio"
#    database = ""  # hack to supress DJ error
#
#    class Responses(dj.Part):
#        @property
#        def definition(self):
#            definition = """
#            # Contains the models state dict, stored externally.
#            -> master
#            ---
#            original_responses:                 attach@{storage}
#            reconstructed_responses:            attach@{storage}
#            """.format(
#                storage=self._master.storage
#            )
#            return definition
#
#    def get_model_responses(self, model, image, device="cuda"):
#        with torch.no_grad():
#            responses = model(
#                image.to(device),
#            )
#        return responses
#
#    def _insert_responses(self, response_entity: Dict[str, Any]) -> None:
#        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
#        with self.get_temp_dir() as temp_dir:
#            for name in ("original_responses", "reconstructed_responses"):
#                self._save_to_disk(response_entity, temp_dir, name)
#            self.Responses.insert1(response_entity, ignore_extra_fields=True)
#
#    def make(self, key):
#        dataloaders, model = self.model_loader.load(key=key)
#        model.eval().cuda()
#        seed = (self.seed_table() & key).fetch1("mei_seed")
#        image = torch.from_numpy((self.image_table & key).fetch1("image"))
#        if image.shape[1] ==1:
#            image = image.repeat(1,3,1,1)
#        wrapper = self.target_unit_table().get_wrapper(key, model)
#        responses = self.get_model_responses(wrapper, image)
#        target_fn = (self.target_fn_table & key).get_target_fn(responses=responses)
#        output_selected_model = self.selector_table().get_output_selected_model(
#            model=wrapper, target_fn=target_fn
#        )
#
#        mei_entity = self.method_table().generate_mei(
#            dataloaders, output_selected_model, key, seed
#        )
#
#        reconstructed_image = mei_entity["mei"]
#        reconstructed_responses = self.get_model_responses(
#            model=wrapper,
#            image=reconstructed_image,
#        )
#        response_entity = dict(
#            original_responses=responses,
#            reconstructed_responses=reconstructed_responses,
#        )
#
#        self._insert_mei(mei_entity)
#        mei_entity.update(response_entity)
#        self._insert_responses(mei_entity)
