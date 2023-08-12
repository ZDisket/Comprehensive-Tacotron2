import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import Tacotron2Loss
from dataset import Dataset

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, infer_one_sample #, read_lexicon
from dataset import TextDataset
from text import text_to_sequence, sequence_to_text

from model import Tacotron2
import numpy as np
import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

def get_tac2(configs, device):
  (preprocess_config, model_config, train_config) = configs

  model = Tacotron2(preprocess_config, model_config, train_config).to(device)

  model.eval()
  model.requires_grad_ = False
  return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="eval or train",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="path out for postnets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    dataset = args.dataset
    out_dir = args.out_path
    preprocess_config, model_config, train_config = get_configs_of(dataset)
    configs = (preprocess_config, model_config, train_config)


    tac2_model_name = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    model = get_tac2(configs, device)

    ckpt = torch.load(tac2_model_name)
    model.load_state_dict(ckpt["model"])

    if not os.path.isdir(out_dir):
      os.mkdir(out_dir)
    
    preprocess_config, model_config, train_config = configs

    model.mask_padding = False

    # Get dataset
    dataset = Dataset(
        args.source, preprocess_config, model_config, train_config, sort=True, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    normalize = preprocess_config["preprocessing"]["mel"]["normalize"]
    mel_stats = None


    # Evaluation
    for batchs in loader:
        for batch in tqdm(batchs):
            batch = to_device(batch, device, mel_stats if normalize else None)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))
                
                postnets = output[1]
                mel_lens = batch[7]
                ids = batch[0]
                
                postnets = postnets.cpu().numpy()
                mel_lens = mel_lens.cpu().numpy()
                for i in range(0,postnets.shape[0]):
                    postnet_unpadded = np.transpose(postnets[i,:mel_lens[i]]) # transpose because vocoder takes in inverse dim
                    out_fn = f"{ids[i]}.npy"
                    out_full_fn = os.path.join(out_dir,out_fn)
                    np.save(out_full_fn,postnet_unpadded)

