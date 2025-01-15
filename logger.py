import torch

class Logger:
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        if len(result) != 3 or not (0 <= run < len(self.results)):
            raise ValueError("Invalid result format or run index.")
        self.results[run].append(result)

    def print_statistics(self, run=None):
        def calculate_statistics(data):
            train_max = data[:, 0].max().item()
            valid_max = data[:, 1].max().item()
            best_index = data[:, 1].argmax()
            final_train = data[best_index, 0].item()
            final_test = data[best_index, 2].item()
            return train_max, valid_max, final_train, final_test

        if run is not None:
            data = 100 * torch.tensor(self.results[run])
            train_max, valid_max, final_train, final_test = calculate_statistics(data)
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {train_max:.2f}')
            print(f'Highest Valid: {valid_max:.2f}')
            print(f'  Final Train: {final_train:.2f}')
            print(f'   Final Test: {final_test:.2f}')
        else:
            all_results = 100 * torch.tensor(self.results)
            stats = [calculate_statistics(run_data) for run_data in all_results]
            stats_tensor = torch.tensor(stats)

            def print_mean_std(idx, label):
                values = stats_tensor[:, idx]
                print(f'{label}: {values.mean():.2f} ± {values.std():.2f}')

            print(f'All runs:')
            print_mean_std(0, 'Highest Train')
            print_mean_std(1, 'Highest Valid')
            print_mean_std(2, 'Final Train')
            print_mean_std(3, 'Final Test')
