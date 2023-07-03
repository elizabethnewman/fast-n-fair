
import pandas as pd

def load_ucla():
    data = pd.read_csv('https://stats.idre.ucla.edu/stat/data/binary.csv')
    return data


if __name__ == "__main__":
    df = load_ucla()

