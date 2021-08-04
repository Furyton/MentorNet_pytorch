"""Generates training data for learning/updating MentorNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import torch
import pickle
import itertools
import numpy as np


def read_from_csv(input_csv_file):
  """Reads Data from an input CSV file.

  Args:
    input_csv_file: the path of the CSV file.

  Returns:
    a numpy array with different data at each index:
  """
  data = {}
  with open(input_csv_file, 'r') as csv_file_in:
    reader = csv.reader(csv_file_in)
    for row in reader:
      for (_, cell) in enumerate(row):
        rdata = cell.strip().split(' ')
        rid = rdata[0]
        rdata = [float(t) for t in rdata[1:]]
        data[rid] = rdata
    csv_file_in.close()
  return data


def generate_data_driven(input_csv_filename,
                         outdir,
                         percentile_range='40,50,60,70,80,90'):
  """Generates a data-driven trainable dataset, given a CSV.

  Refer to README.md for details on how to format the CSV.

  Args:
    input_csv_filename: the path of the CSV file. The csv file format
      0: epoch_percentage
      1: noisy label
      2: clean label
      3: loss
    outdir: directory to save the training data.
    percentile_range: the percentiles used to compute the moving average.
  """
  raw = read_from_csv(input_csv_filename)

  raw = np.array(raw.values())
  dataset_name = os.path.splitext(os.path.basename(input_csv_filename))[0]

  percentile_range = percentile_range.split(',')
  percentile_range = [int(x) for x in percentile_range]

  for percentile in percentile_range:
    percentile = int(percentile)
    p_perncentile = np.percentile(raw[:, 3], percentile)

    v_star = np.float32(raw[:, 1] == raw[:, 2])

    l = raw[:, 3]
    diff = raw[:, 3] - p_perncentile
    # label not used in the current version.
    y = np.array([0] * len(v_star))
    epoch_percentage = raw[:, 0]

    data = np.vstack((l, diff, y, epoch_percentage, v_star))
    data = np.transpose(data)

    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)
    data = data[perm,]

    tr_size = int(data.shape[0] * 0.8)

    tr = data[0:tr_size]
    ts = data[(tr_size + 1):data.shape[0]]

    cur_outdir = os.path.join(
        outdir, '{}_percentile_{}'.format(dataset_name, percentile))
    if not os.path.exists(cur_outdir):
      os.makedirs(cur_outdir)

    print('training_shape={} test_shape={}'.format(tr.shape, ts.shape))
    print(cur_outdir)
    with open(os.path.join(cur_outdir, 'tr.p'), 'wb') as outfile:
      pickle.dump(tr, outfile)

    with open(os.path.join(cur_outdir, 'ts.p'), 'wb') as outfile:
      pickle.dump(ts, outfile)
