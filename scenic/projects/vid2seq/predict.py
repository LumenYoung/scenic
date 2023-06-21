r"""Main script for getting Dense Video Captioning predictions."""

import os
from typing import Any, Callable

from absl import flags
from clu import metric_writers
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.projects.vid2seq import models
from scenic.projects.vid2seq import trainer

from scenic.projects.vid2seq.config.eval_configuration import get_config


def make_predictions():

    rng_key = random.PRNGKey(0)

    config = get_config(run_local="True")

    workdir = "/home/ec2-user/Documents/experiments"

    jave_jre = JRE_BIN_JAVA

    os.environ['JRE_BIN_JAVA'] = java_jre

    JRE_BIN_JAVA = "/usr/bin/java"

    flags.DEFINE_string('jre_path', '',
                        'Path to JRE.')

    # ensure arguments match
    config.model.decoder.num_bins = config.dataset_configs.num_bins
    config.model.decoder.tmp_only = config.dataset_configs.tmp_only
    config.model.decoder.order = config.dataset_configs.order

    model_cls = get_model_cls(config.model_name)
    data_rng, rng = jax.random.split(rng)
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
