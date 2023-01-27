from src.data_preparation import DataPreparation
from pyhocon import ConfigFactory


config = ConfigFactory.parse_file('config/COMMON.conf')


if __name__ == '__main__':
    data_preparation = DataPreparation(config=config)
    nlp_df, existing_annotations_df = data_preparation.run()
    print("ok")

    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
