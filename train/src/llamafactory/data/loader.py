# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)

# Review package adaptation (2026-09-24): configurable external data roots.
# --- SkinGPT-R1-Omni Path Constants ---
IMAGE_ROOT_DIR = os.environ.get("SKINGPT_IMAGE_ROOT", "./external/images")
FEATURE_ROOT_DIR = os.environ.get("SKINGPT_FEATURE_ROOT", "./external/teacher_features")


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    print(f"🔄 [SkinGPT-Loader] Starting to load dataset: {dataset_attr}...")
    
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    print(f"✅ [SkinGPT-Loader] Successfully loaded raw dataset. Size: {len(dataset)}")

    # --- SkinGPT-R1 Survival Patch ---
    # align_dataset (converter.py) usually strips unknown columns. 
    # Since we can't edit converter.py here, we backup skin_labels and restore them.
    # Note: teacher_feat is added LATER in _get_preprocessed_dataset, so it's safe.
    raw_skin_labels = None
    column_names = list(next(iter(dataset)).keys())
    if "skin_labels" in column_names and hasattr(dataset, "remove_columns"): # Ensure it's not IterableDataset in strict mode
        print(f"🛡️ [SkinGPT-Loader] Backing up 'skin_labels' before alignment...")
        raw_skin_labels = dataset["skin_labels"]
    
    # Execute standard alignment
    dataset = align_dataset(dataset, dataset_attr, data_args, training_args)

    # Restore skin_labels if they were lost
    if raw_skin_labels is not None:
        aligned_cols = list(next(iter(dataset)).keys())
        if "skin_labels" not in aligned_cols:
             print(f"🚑 [SkinGPT-Loader] Restoring 'skin_labels' to aligned dataset...")
             try:
                 dataset = dataset.add_column("skin_labels", raw_skin_labels)
                 print(f"✅ [SkinGPT-Loader] Restoration successful.")
             except Exception as e:
                 print(f"⚠️ [SkinGPT-Loader] Failed to restore skin_labels: {e}")
    # --- End Survival Patch ---

    return dataset


def _get_merged_dataset(
    dataset_names: Optional[list[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )

    # --- SkinGPT-R1-Omni Injection: Path Stitching & Teacher Feature Loading ---
    def inject_paths_and_features(example, idx):
        
        image_list = example.get("_images", [])
        
        
        if not image_list:
            example["teacher_feat"] = [0.0] * 1024
            return example

        abs_image_paths = []
        teacher_feat_vector = None
        has_npy = False

        
        for i, rel_path in enumerate(image_list):
            
            # rel_path e.g. "IIYI/26549_2.png"
            abs_path = os.path.join(IMAGE_ROOT_DIR, rel_path)
            abs_image_paths.append(abs_path)

            
            if i == 0:
                
                file_name_no_ext = os.path.splitext(rel_path)[0]
                feat_path = os.path.join(FEATURE_ROOT_DIR, file_name_no_ext + ".npy")
                
                if os.path.exists(feat_path):
                    try:
                        feat = np.load(feat_path)
                        # Ensure we get a flat list
                        if isinstance(feat, np.ndarray):
                            teacher_feat_vector = feat.flatten().tolist()
                            has_npy = True
                    except Exception as e:
                        if idx < 5: 
                            logger.warning(f"⚠️ Failed to load npy: {feat_path}, error: {e}")
                
        
        example["_images"] = abs_image_paths
        
        
        if teacher_feat_vector is not None:
            example["teacher_feat"] = teacher_feat_vector
        else:
            example["teacher_feat"] = [0.0] * 1024

        if idx % 100000 == 0:
            status = "✅ Found NPY" if has_npy else "❌ No NPY"
            print(f"   [Process {os.getpid()}] Item {idx}: {image_list[0]} -> {status}")

        return example
    
    print(f"🚀 [SkinGPT-Loader] Injecting absolute paths into '_images' for {len(dataset)} items...")
    dataset = dataset.map(
        inject_paths_and_features, 
        with_indices=True,
        num_proc=data_args.preprocessing_num_workers
    )
    print("✅ [SkinGPT-Loader] Injection complete.")

    # --- End of Injection ---

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    
    cols_to_keep = {"images", "teacher_feat", "videos", "audios", "skin_labels"}
    columns_to_remove = [c for c in column_names if c not in cols_to_keep]
    print(f"🛠️ [SkinGPT-Loader] Removing columns (except reserved): {columns_to_remove}")

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=columns_to_remove,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("="*40)
            print("🧐 [SkinGPT-Loader] Inspecting first processed batch:")
            ex = next(iter(dataset))
            for k, v in ex.items():
                if k in ["teacher_feat", "input_ids"]:
                    print(f"   -> key: {k:<15} | type: {type(v)} | len: {len(v)}")
                elif k == "skin_labels":
                    print(f"   -> key: {k:<15} | value: {v}")
                else:
                    print(f"   -> key: {k:<15}")
            print("="*40)
        except StopIteration:
            pass

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    print("🚀 [SkinGPT-Loader] Entering get_dataset()...")
    
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        
        
        print(f"🎉 [SkinGPT-Loader] Final Dataset Split:")
        if "train_dataset" in dataset_dict:
            print(f"   -> Train: {len(dataset_dict['train_dataset'])} samples")
        if "eval_dataset" in dataset_dict:
            print(f"   -> Eval : {len(dataset_dict['eval_dataset'])} samples")

        if data_args.tokenized_path is not None:
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")

        return get_dataset_module(dataset_dict)