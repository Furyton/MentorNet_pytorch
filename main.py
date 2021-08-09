import argparse
from ast import arg, parse
import training_mentornet.train as train
import training_mentornet.data_generator as data_generator

csv_file_path = "/data/lizongbu-slurm/furyton/mentornet/MentorNet_pytorch/fake_data.csv"

preprocess_data_path = "processed_data/fake_data_percentile_40"

def str2bool(v):
    return v.lower() in ('true')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--process_data', type=str2bool)
    parser.add_argument('--raw_csv', type=str, default=None, help="raw csv file path")
    parser.add_argument('--data_path', type=str, default=None, help="where you want to save the processed dataset")
    parser.add_argument('--processed_path', type=str, default=None)
    parser.add_argument('--train_dir', type=str, default='trial')
    parser.add_argument('--epoch',type=int, default=10)
    parser.add_argument('--device',type=str,default='cpu')
    parser.add_argument('--lr',type=float,default=0.1)
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--show_progress_bar',type=str2bool, default=False)

    config = parser.parse_args()

    if config.process_data:
        data_generator.generate_data_driven(config.raw_csv, config.data_path)
    else:
        tr = train.trainer(train_dir=config.train_dir, data_path=config.processed_path, show_progress_bar=config.show_progress_bar, epoch=config.epoch, mini_batch_size=config.batch_size, device=config.device)

        tr.train()