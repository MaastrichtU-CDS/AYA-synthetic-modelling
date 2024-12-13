import os
import sdv
import datetime
import pytz


class DataGenerator:
    def __init__ (self):

        self.Data_Generators = input("Data Generator (fast_ml, gc, ctgan): ")

        amsterdam_tz = pytz.timezone('Europe/Amsterdam') # Set timezone to Amsterdam
        self.now = datetime.datetime.now(amsterdam_tz) # Get the current date and time in Amsterdam timezone
        self.date = self.now.strftime("%d%m") # Format the date and time as DDMM-HHMM
        print(self.date)

        if self.Data_Generators == 'fast_ml':
            self.model_path = '/root/fast_ml/generator/survaya_03062022_clean_n_output_model_FAST_ML.pk1'
            self.model = sdv.tabular.GaussianCopula.load(self.model_path)
            os.makedirs('/root/fast_ml/synthetic_data', exist_ok=True)
        elif self.Data_Generators == 'gc':
            self.model_path = '/root/gc/generator/survaya_03062022_clean_n_output_model_GaussianCopula.pk1'
            self.model = sdv.tabular.GaussianCopula.load(self.model_path)
            os.makedirs('/root/gc/synthetic_data', exist_ok=True)
        elif self.Data_Generators == 'ctgan':
            self.model_path = '/root/ctgan/generator/survaya_03062022_clean_n_output_model_CTGAN.pk1'
            self.model = sdv.tabular.CTGAN.load(self.model_path)
            os.makedirs('/root/ctgan/synthetic_data', exist_ok=True)

        elif self.Data_Generators == 'dp-cgan':
            # Prevent loading dp-cgans if not specified because of version incompatibility
            from dp_cgan import load_dp_cgan
            self.model_path = '/root/dp-cgan/generator/survaya_03062022_clean_n_output_model_DP_CGAN.pk1'
            self.model = load_dp_cgan(self.model_path)
            os.makedirs('/root/dp-cgan/synthetic_data', exist_ok=True)


        self.sample_size = 3735

        for run in range(1, 4): # (, n+1)
            self.samples = self.model.sample(self.sample_size)

            if self.Data_Generators == 'fast_ml':
                self.output_csv_path =f'/root/fast_ml/synthetic_data/run_{run}_{self.date}_FAST_ML_{self.sample_size}_samples.csv'
                self.samples.to_csv(self.output_csv_path, index=False)   
                print(f"Generated samples saved to {self.output_csv_path}")
            elif self.Data_Generators == 'gc':
                self.output_csv_path =f'/root/gc/synthetic_data/run_{run}_{self.date}_Gaussian_Copula_{self.sample_size}_samples.csv'
                self.samples.to_csv(self.output_csv_path, index=False)
                print(f"Generated samples saved to {self.output_csv_path}")
            elif self.Data_Generators == 'ctgan':
                self.output_csv_path =f'/root/ctgan/synthetic_data/run_{run}_{self.date}_CTGAN_{self.sample_size}_samples.csv'
                self.samples.to_csv(self.output_csv_path, index=False)
                print(f"Generated samples saved to {self.output_csv_path}")
            elif self.Data_Generators == 'dp-cgan':
                self.output_csv_path =f'/root/dp-cgan/synthetic_data/run_{run}_{self.date}_DP_CGAN_{self.sample_size}_samples.csv'
                self.samples.to_csv(self.output_csv_path, index=False)
                print(f"Generated samples saved to {self.output_csv_path}")


if __name__ == "__main__":
    DataGenerator()
