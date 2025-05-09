import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

from flags import DATA_FOLDER

@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(DATA_FOLDER))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)


        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames

    # def read_data(self, classnames, split_dir):
        # split_dir = os.path.join(self.image_dir, split_dir)
        # folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        # items = []

        # for label, folder in enumerate(folders):
        #     imnames = listdir_nohidden(os.path.join(split_dir, folder))
        #     classname = classnames[folder]
        #     for imname in imnames:
        #         impath = os.path.join(split_dir, folder, imname)
        #         item = Datum(impath=impath, label=label, classname=classname)
        #         items.append(item)

        # return items
        
        
        #Nhitny
    def read_data(self, classnames, split_dir):
            split_dir = os.path.join(self.image_dir, split_dir)
            folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
            items = []

            for label, folder in enumerate(folders):
                imnames = listdir_nohidden(os.path.join(split_dir, folder))
                classname = classnames[folder]
                for imname in imnames:
                    impath = os.path.join(split_dir, folder, imname)
                    if not os.path.isfile(impath):
                        import warnings
                        warnings.warn(f"⚠️ Skipping missing image: {impath}")
                        continue
                    # item = Datum(impath=impath, label=label, classname=classname)
                    
                    #Nhitny
                    if not os.path.isfile(impath):
                        import warnings
                        warnings.warn(f"⚠️ Skipping missing image: {impath}")
                        continue
                    item = Datum(impath=impath, label=label, classname=classname)
                    items.append(item)

                    #Nhitny
                    items.append(item)

            return items

        #Nhitny

# import os
# import pickle
# from collections import OrderedDict

# from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from dassl.utils import listdir_nohidden, mkdir_if_missing
# from dassl.utils import read_json, read_image


# from .oxford_pets import OxfordPets

# from flags import DATA_FOLDER

# @DATASET_REGISTRY.register()
# class ImageNet(DatasetBase):

#     dataset_dir = "imagenet"

#     def __init__(self, cfg):
#         root = os.path.abspath(os.path.expanduser(DATA_FOLDER))
#         self.dataset_dir = os.path.join(root, self.dataset_dir)
#         self.image_dir = os.path.join(self.dataset_dir, "images")
#         text_file = os.path.join(self.dataset_dir, "classnames.txt")
#         train_file = os.path.join(self.dataset_dir, "split/train.json")
#         test_file = os.path.join(self.dataset_dir, "split/test.json")

#         classnames = self.read_classnames(text_file)
#         train = self.read_data(train_file, classnames)
#         test = self.read_data(test_file, classnames)

#         if cfg.subsample < 1:
#             train, test = self.subsample_classes(train, test, cfg.subsample)

#         super().__init__(train_x=train, test=test, classnames=classnames)

#     def read_classnames(self, text_file):
#         with open(text_file, "r") as f:
#             classnames = [line.strip() for line in f.readlines()]
#         return classnames

#     def read_data(self, file_path, classnames):
#         items = []
#         data = read_json(file_path)
#         for label, entries in data.items():
#             label = int(label)
#             classname = classnames[label]
#             for entry in entries:
#                 impath = os.path.join(self.image_dir, entry)
#                 items.append(Datum(impath=impath, label=label, classname=classname))
#         return items

#     def subsample_classes(self, train, test, subsample=0.5):
#         all_classes = list(set(item.label for item in train))
#         num_classes = int(len(all_classes) * subsample)
#         selected_classes = sorted(random.sample(all_classes, num_classes))
#         label_map = {old: new for new, old in enumerate(selected_classes)}

#         def filter_and_remap(data):
#             result = []
#             for item in data:
#                 if item.label in selected_classes and os.path.isfile(item.impath):
#                     new_label = label_map[item.label]
#                     result.append(Datum(impath=item.impath, label=new_label, classname=item.classname))
#             return result

#         return filter_and_remap(train), filter_and_remap(test)