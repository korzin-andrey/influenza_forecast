from dataclasses import dataclass
import numpy as np

from bootstrapping.error_structures import generate_synth_data_neg_bin_trunc_predict


@dataclass
class PredictGate:
    x: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    title: str
    column: str
    length: int
    week_begin: int

    def draw(self, plt, model_fit, sample_size, show_title=True, **kwargs):
        x = self.x[sample_size-1:sample_size+self.length]
        y1 = self.y_min[sample_size-1:sample_size+self.length]
        y2 = self.y_max[sample_size-1:sample_size+self.length]

        y1[0] = y2[0] = model_fit[self.column][self.week_begin+sample_size - 1]

        if show_title:
            plt.fill_between(x, y1, y2, label=self.title, **kwargs)
        else:
            plt.fill_between(x, y1, y2, **kwargs)


class PredictGatesGenerator:
    def __init__(self, original_data, model_fit, datasets_amount=200,
                 sample_size=8, inflation_parameter=1.0, end=None):

        self.outbreak_begin, self.outbreak_end = original_data.index.min(), original_data.index.max() + 1
        self.simulated_datasets = generate_synth_data_neg_bin_trunc_predict(original_data,
                                                                            model_fit,
                                                                            datasets_amount,
                                                                            sample_size=0,
                                                                            inflation_parameter=inflation_parameter,
                                                                            begin=sample_size,
                                                                            end=end)
        self.original_data = original_data
        self.model_fit = model_fit

        def restructurate(simul_weekly_bs):
            begin, end = self.outbreak_begin, self.outbreak_end
            ds_columns = dict()
            for column in self.model_fit.columns:
                new_ds = dict()
                for i in range(begin, end):
                    new_val = []
                    for j in range(len(simul_weekly_bs)):
                        new_val.append(simul_weekly_bs[j].loc[i, column])
                    new_ds[i] = new_val
                ds_columns[column] = new_ds
            return ds_columns

        self.simulated_datasets_restructurated = restructurate(self.simulated_datasets)
        self.sample_size = sample_size

    def generate_predict_gate(self, percentile_lower, percentile_upper, column=None, length=None, week_begin=None):
        if column:  # generate for specific column
            x = []
            y_lower = []
            y_upper = []
            for key in self.simulated_datasets_restructurated[column].keys():
                x.append(key)
                y_lower.append(np.percentile(self.simulated_datasets_restructurated[column][key], percentile_lower))
                y_upper.append(np.percentile(self.simulated_datasets_restructurated[column][key], percentile_upper))
            return PredictGate(np.array(x), np.array(y_lower), np.array(y_upper),
                               f'"{column}" {percentile_lower}th - {percentile_upper}th percentile',
                               column,
                               length if length is not None else self.outbreak_end-self.outbreak_begin,
                               week_begin if week_begin is not None else self.outbreak_begin)

        # generate for all columns
        return [self.generate_predict_gate(percentile_lower, percentile_upper, col, length, week_begin)
                for col in self.model_fit.columns]

    def draw_gate(self, plt, predict_gate: PredictGate, show_title=True, **kwargs):
        predict_gate.draw(plt, self.model_fit,
                          self.sample_size,
                          show_title, **kwargs)
