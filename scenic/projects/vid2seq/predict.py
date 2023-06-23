r"""Main script for getting Dense Video Captioning predictions."""

import os
from typing import Any, Callable

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
from jax import random
import ml_collections
from scenic import app
from scenic.projects.vid2seq import models
from scenic.projects.vid2seq import trainer

from scenic.projects.vid2seq.configs.eval_configuration import get_config


def get_model_cls(model_name: str) -> Callable[..., Any]:
  """Returns model class given its name."""
  if model_name == 'vid2seq':
    return models.DenseVideoCaptioningModel
  raise ValueError(f'Unrecognized model: {model_name}.')


def make_predictions():

    rng_key = random.PRNGKey(0)

    config = get_config(runlocal="True")

    workdir = "/home/ec2-user/Documents/experiments"

    JRE_BIN_JAVA = "/usr/bin/java"

    jave_jre = JRE_BIN_JAVA

    os.environ['JRE_BIN_JAVA'] = java_jre

    flags.DEFINE_string('jre_path', '',
                        'Path to JRE.')

    # ensure arguments match
    config.model.decoder.num_bins = config.dataset_configs.num_bins
    config.model.decoder.tmp_only = config.dataset_configs.tmp_only
    config.model.decoder.order = config.dataset_configs.order

    model_cls = get_model_cls(config.model_name)

    data_rng, rng = jax.random.split(rng_key)
    dataset_dict = get_datasets(
      config,
      data_rng=data_rng)

    predictions = trainer.predict_only(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset_dict=dataset_dict,
        workdir=workdir,
        )
    
    print(predictions)


if __name__ == "__main__":
    make_predictions()
