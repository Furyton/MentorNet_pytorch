import training_mentornet.train as train

if __name__ == '__main__':
    tr = train.trainer(train_dir="trial", data_path="/data/lizongbu-slurm/furyton/mentornet/MentorNet_pytorch/training_mentornet/processed_data/fake_data_percentile_40", show_progress_bar=False, epoch=10)

    tr.train()