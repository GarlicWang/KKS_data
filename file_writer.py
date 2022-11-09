import logging
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class FileWriter:
    def __init__(
        self
    ):
        pass

    def remap(
        self,
        training_df,
        testing_df,
        kg_data,
        output_dir_path,
    ):
        logger.info(f"remapping data...")
        output_dir_path.mkdir(parents=True, exist_ok=True)
        kg_data_array = np.array(kg_data)
        usr_encoder, interaction_encoder, item_encoder, rel_encoder, ent_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder(), LabelEncoder()
        usr_encoder.fit(list(set(training_df['user_id']) | set(testing_df['user_id'])))
        interaction_encoder.fit(list(set(training_df['interaction']) | set(testing_df['interaction'])))
        item_encoder.fit(list(set(kg_data_array[:,0]) | set(training_df['item_id']) | set(testing_df['item_id'])))
        rel_encoder.fit(kg_data_array[:,1])
        ent_encoder.fit(kg_data_array[:,2])
        
        self.usr_encoder = usr_encoder
        self.interaction_encoder = interaction_encoder
        self.item_encoder = item_encoder
        self.rel_encoder = rel_encoder
        self.ent_encoder = ent_encoder
        
        logger.info(f"writing encoder to {output_dir_path}/usr_classes.npy")
        np.save(output_dir_path / 'usr_classes.npy', usr_encoder.classes_)
        logger.info(f"writing encoder to {output_dir_path}/interaction_classes.npy")
        np.save(output_dir_path / 'interaction_classes.npy', interaction_encoder.classes_)
        logger.info(f"writing encoder to {output_dir_path}/item_classes.npy")
        np.save(output_dir_path / 'item_classes.npy', item_encoder.classes_)
        logger.info(f"writing class data to {output_dir_path}/rel_classes.npy")
        np.save(Path(output_dir_path) / 'rel_classes.npy', rel_encoder.classes_)
        logger.info(f"writing class data to {output_dir_path}/ent_classes.npy")
        np.save(Path(output_dir_path) / 'ent_classes.npy', ent_encoder.classes_)

    def write_triple(
        self,
        training_df,
        testing_df,
        output_dir_path,
    ):
        logger.info(f"writing triple data...")
        output_dir_path.mkdir(parents=True, exist_ok=True)
        _training_df, _testing_df = training_df.copy(), testing_df.copy()

        logger.info(f"writing triple data to {output_dir_path}/train.txt")
        with open(output_dir_path / "train.txt", "w") as f:
            for usr, rel, item in _training_df[["user_id", "interaction", "item_id"]].values:
                f.write(str(usr) + '\t' + str(rel) + '\t' + str(item) + '\n')
        logger.info(f"writing triple data to {output_dir_path}/test.txt")
        with open(output_dir_path / "test.txt", "w") as f:
            for usr, rel, item in _testing_df[["user_id", "interaction", "item_id"]].values:
                f.write(str(usr) + '\t' + str(rel) + '\t' + str(item) + '\n')

        # remap
        _training_df['user_id'] = self.usr_encoder.transform(_training_df['user_id'])
        _training_df['interaction'] = self.interaction_encoder.transform(_training_df['interaction'])
        _training_df['item_id'] = self.item_encoder.transform(_training_df['item_id'])
        _testing_df['user_id'] = self.usr_encoder.transform(_testing_df['user_id'])
        _testing_df['interaction'] = self.interaction_encoder.transform(_testing_df['interaction'])
        _testing_df['item_id'] = self.item_encoder.transform(_testing_df['item_id'])

        logger.info(f"writing triple data to {output_dir_path}/train.encoded.txt")
        with open(output_dir_path / "train.encoded.txt", "w") as f:
            for usr, rel, item in _training_df[["user_id", "interaction", "item_id"]].values:
                f.write(str(usr) + '\t' + str(rel) + '\t' + str(item) + '\n')
        logger.info(f"writing triple data to {output_dir_path}/test.encoded.txt")
        with open(output_dir_path / "test.encoded.txt", "w") as f:
            for usr, rel, item in _testing_df[["user_id", "interaction", "item_id"]].values:
                f.write(str(usr) + '\t' + str(rel) + '\t' + str(item) + '\n')


    def write_tuple(
        self,
        training_df,
        testing_df,
        output_dir_path,
    ):
        logger.info(f"writing tuple data...")
        output_dir_path.mkdir(parents=True, exist_ok=True)
        _training_df, _testing_df = training_df.copy(), testing_df.copy()

        logger.info(f"writing tuple data to {output_dir_path}/train.txt")
        with open(output_dir_path / "train.txt", "w") as f:
            for usr, item in training_df[["user_id", "item_id"]].values:
                f.write(str(usr) + '\t' + str(item) + '\n')
        logger.info(f"writing tuple data to {output_dir_path}/test.txt")
        with open(output_dir_path / "test.txt", "w") as f:
            for usr, item in testing_df[["user_id", "item_id"]].values:
                f.write(str(usr) + '\t' + str(item) + '\n')
        
        # remap
        _training_df['user_id'] = self.usr_encoder.transform(_training_df['user_id'])
        _training_df['interaction'] = self.interaction_encoder.transform(_training_df['interaction'])
        _training_df['item_id'] = self.item_encoder.transform(_training_df['item_id'])
        _testing_df['user_id'] = self.usr_encoder.transform(_testing_df['user_id'])
        _testing_df['interaction'] = self.interaction_encoder.transform(_testing_df['interaction'])
        _testing_df['item_id'] = self.item_encoder.transform(_testing_df['item_id'])

        logger.info(f"writing tuple data to {output_dir_path}/train.encoded.txt")
        with open(output_dir_path / "train.encoded.txt", "w") as f:
            for usr, item in _training_df[["user_id", "item_id"]].values:
                f.write(str(usr) + '\t' + str(item) + '\n')
        logger.info(f"writing tuple data to {output_dir_path}/test.encoded.txt")
        with open(output_dir_path / "test.encoded.txt", "w") as f:
            for usr, item in _testing_df[["user_id", "item_id"]].values:
                f.write(str(usr) + '\t' + str(item) + '\n')


    def write_userwise(
        self,
        training_df,
        testing_df,
        output_dir_path,
    ):
        logger.info(f"writing tuple data...")
        output_dir_path.mkdir(parents=True, exist_ok=True)
        _training_df, _testing_df = training_df.copy(), testing_df.copy()

        train_usr_dict, test_usr_dict = {}, {}
        for usr, item in _training_df[["user_id", "item_id"]].values:
            if usr not in train_usr_dict:
                train_usr_dict[usr] = set()
            train_usr_dict[usr].add(item)
        for usr, item in _testing_df[["user_id", "item_id"]].values:
            if usr not in test_usr_dict:
                test_usr_dict[usr] = set()
            test_usr_dict[usr].add(item)

        logger.info(f"writing userwise data to {output_dir_path}/train.txt")
        with open(output_dir_path / "train.txt", "w") as f:
            for usr in train_usr_dict:
                items = list(sorted(train_usr_dict[usr]))
                items = [str(item) for item in items]
                items = " ".join(items)
                f.write(str(usr) + " " + items + '\n')
        logger.info(f"writing userwise data to {output_dir_path}/test.txt")
        with open(output_dir_path / "test.txt", "w") as f:
            for usr in test_usr_dict:
                items = list(sorted(test_usr_dict[usr]))
                items = [str(item) for item in items]
                items = " ".join(items)
                f.write(str(usr) + " " + items + '\n')
        
        # remap
        _training_df['user_id'] = self.usr_encoder.transform(_training_df['user_id'])
        _training_df['interaction'] = self.interaction_encoder.transform(_training_df['interaction'])
        _training_df['item_id'] = self.item_encoder.transform(_training_df['item_id'])
        _testing_df['user_id'] = self.usr_encoder.transform(_testing_df['user_id'])
        _testing_df['interaction'] = self.interaction_encoder.transform(_testing_df['interaction'])
        _testing_df['item_id'] = self.item_encoder.transform(_testing_df['item_id'])

        train_usr_dict, test_usr_dict = {}, {}
        for usr, item in _training_df[["user_id", "item_id"]].values:
            if usr not in train_usr_dict:
                train_usr_dict[usr] = set()
            train_usr_dict[usr].add(item)
        for usr, item in _testing_df[["user_id", "item_id"]].values:
            if usr not in test_usr_dict:
                test_usr_dict[usr] = set()
            test_usr_dict[usr].add(item)

        logger.info(f"writing userwise data to {output_dir_path}/train.encoded.txt")
        with open(output_dir_path / "train.encoded.txt", "w") as f:
            for usr in train_usr_dict:
                items = list(sorted(train_usr_dict[usr]))
                items = [str(item) for item in items]
                items = " ".join(items)
                f.write(str(usr) + " " + items + '\n')
        logger.info(f"writing userwise data to {output_dir_path}/test.encoded.txt")
        with open(output_dir_path / "test.encoded.txt", "w") as f:
            for usr in test_usr_dict:
                items = list(sorted(test_usr_dict[usr]))
                items = [str(item) for item in items]
                items = " ".join(items)
                f.write(str(usr) + " " + items + '\n')


    def write_kgdata(
        self,
        kg_data,
        kg_data_text,
        output_dir_path,
    ):
        logger.info("writing kg data...")
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"writing kg data to {output_dir_path}/kg.txt")
        with open(output_dir_path / 'kg.txt', 'w') as f:
            for h, r, t in kg_data:
                f.write(str(h) + '\t' + str(r) + '\t' + str(t) +'\n')
                
        logger.info(f"writing kg data with name to {output_dir_path}/kg.name.txt")
        with open(output_dir_path / 'kg.name.txt', 'w') as f:
            for h, r, t in kg_data_text:
                f.write(str(h) + '\t' + str(r) + '\t' + str(t) +'\n')

        # remap
        kg_data_array = np.array(kg_data)
        kg_data_array[:,0] = self.item_encoder.transform(kg_data_array[:,0])
        item_num = len(self.item_encoder.classes_)
        logger.info(f"item num = {item_num}")
        kg_data_array[:,1] = self.rel_encoder.transform(kg_data_array[:,1])
        kg_data_array[:,2] = self.ent_encoder.transform(kg_data_array[:,2])
        for i in range(len(kg_data_array)):
            kg_data_array[i,2] = str(int(kg_data_array[i,2]) + item_num)
        kg_data = kg_data_array.tolist()

        logger.info(f"writing kg data to {output_dir_path}/kg.encoded.txt")
        with open(output_dir_path / 'kg.encoded.txt', 'w') as f:
            for h, r, t in kg_data:
                f.write(str(h) + '\t' + str(r) + '\t' + str(t) +'\n')

