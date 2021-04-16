from __future__ import annotations
import warnings
from functools import partial
from typing import Dict, Any, Callable, List
Key = Dict[str, Any]
Dataloaders = Dict[str, DataLoader]

import torch
from torch.nn import Module

import datajoint as dj
from nnfabrik.main import Dataset
from nnfabrik.builder import resolve_fn
from nnfabrik.utility.dj_helpers import CustomSchema, make_hash, cleanup_numpy_scalar

from bias_transfer.tables.trained_model import *
from bias_transfer.tables.trained_transfer_model import *

from mei import mixins
from mei.import_helpers import import_object

from ..modules.reducers import ConstrainedOutputModel


schema = CustomSchema(dj.config.get("schema_name", "nnfabrik_core"))
resolve_target_fn = partial(resolve_fn, default_base="targets")


@schema
class ReconSeed(mixins.MEISeedMixin, dj.Lookup):
    """Seed table for MEI method."""


@schema
class ReconstructionImages(dj.Manual):
    definition = """
      # images for reconstruction
      img_id:    int                  # unique id
      img_class: varchar(64)          # image type descriptor
      ---
      image: longblob                 # actual image as numpy array
      """


@schema
class ReconMethod(mixins.MEIMethodMixin, dj.Lookup):
    seed_table = ReconSeed


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
    return_layer:                   longblob       # list of unit_ids 
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
        dataloaders, model = self.model_loader.load(key=key)
        model().eval().cuda()
        seed = (self.seed_table() & key).fetch1("mei_seed")
        image = torch.from_numpy((self.image_table & key).fetch1("image"))
        wrapper = self.target_unit_table().get_wrapper(key, model)
        responses = self.get_model_responses(wrapper, image)
        target_fn = (self.target_fn_table & key).get_target_fn(responses=responses)
        output_selected_model = self.selector_table().get_output_selected_model(
            model=model, target_fn=target_fn
        )

        mei_entity = self.method_table().generate_mei(
            dataloaders, output_selected_model, key, seed
        )

        reconstructed_image = mei_entity["mei"]
        reconstructed_responses = self.get_model_responses(
            model=wrapper,
            image=reconstructed_image,
        )
        response_entity = dict(
            original_responses=responses,
            reconstructed_responses=reconstructed_responses,
        )

        self._insert_mei(mei_entity)
        mei_entity.update(response_entity)
        self._insert_responses(mei_entity)
